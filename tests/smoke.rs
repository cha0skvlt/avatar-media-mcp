//! Smoke suite: every public entry point the server relies on, exercised
//! without models, network or ffmpeg. Each case asserts a contract, not a
//! happy-path print: config files parse, the consent gate holds, stubs return
//! errors instead of panicking, and the identity core answers deterministically.

use avatar_media_mcp::config::models::{ModelsConfig, Role, Tier};
use avatar_media_mcp::config::platforms::PlatformsConfig;
use avatar_media_mcp::config::settings::Settings;
use avatar_media_mcp::infra::avatar_store::{AvatarStore, AVATAR_FILE, MODEL_FILE, REFS_DIR};
use avatar_media_mcp::domain::avatar::{Avatar, Identity, IdentityModel, SCHEMA_VERSION};
use avatar_media_mcp::domain::script::{Script, Shot, ShotType};
use avatar_media_mcp::identity::clusters::{cluster_references, Reference};
use avatar_media_mcp::identity::embeddings::{cosine_similarity_checked, CosineScore, EmbeddingSource};
use avatar_media_mcp::identity::scoring::{stage1_score, IdentityThresholds, Stage1Verdict};
use avatar_media_mcp::identity::textpass::{
    channel_order, parse_agent_identity, parse_agent_verification, verification_round_trip,
    AgentChannel, NumericSummary, TextChannel,
};
use avatar_media_mcp::infra::cache::Cache;
use avatar_media_mcp::infra::ffmpeg::{check_available, probe, FfmpegOp};
use avatar_media_mcp::pipeline::{assemble, audio, keyframes, qc, shots};
use avatar_media_mcp::series;
use avatar_media_mcp::tools::avatar::{avatar_create, AvatarCreateArgs};
use avatar_media_mcp::tools::health::preflight;

const MANIFEST: &str = env!("CARGO_MANIFEST_DIR");

fn manifest_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(MANIFEST).join(relative)
}

fn script() -> Script {
    Script {
        title: "smoke".into(),
        language: "en".into(),
        platform: "youtube_shorts".into(),
        hook: "hook".into(),
        cast: vec!["smoke".into()],
        shots: vec![Shot {
            id: "s1".into(),
            shot_type: ShotType::Talk,
            vo: Some("hello".into()),
            visual: "close-up".into(),
            characters: vec!["smoke".into()],
            speaking: Some("smoke".into()),
            framing: None,
            wardrobe: None,
            motion: None,
            est_seconds: 4.0,
        }],
    }
}

#[test]
fn config_files_parse_and_resolve() {
    let models = ModelsConfig::from_path(&manifest_path("config/models.yaml")).unwrap();
    let entry = models.resolve(Role::Talk, Tier::Standard).unwrap();
    assert!(!entry.model_id.is_empty());
    // `avatar_canon` has no draft tier: the documented fallback must answer.
    assert!(models.resolve(Role::AvatarCanon, Tier::Draft).is_some());

    let platforms = PlatformsConfig::from_path(&manifest_path("config/platforms.yaml")).unwrap();
    assert!(platforms.export.ai_disclosure_default);
    assert!(!platforms.export.auto_publish);
    assert!(!platforms.platforms.is_empty());
}

#[test]
fn settings_load_from_the_real_file() {
    let settings = Settings::from_path(&manifest_path("config/settings.yaml")).unwrap();
    assert_eq!(settings.default_tier, Tier::Standard);
    assert!(settings.project_budget_usd > 0.0);
    assert!(settings.max_concurrency >= 1);
    // A missing file is a valid configuration, not an error.
    assert_eq!(
        Settings::load(Some(std::path::Path::new("/nonexistent/settings.yaml"))).unwrap(),
        Settings::default()
    );
}

#[test]
fn avatar_layout_is_canonical_and_self_contained() {
    let dir = tempfile::tempdir().unwrap();
    let store = AvatarStore::new(dir.path());
    std::fs::create_dir_all(store.refs_dir("smoke")).unwrap();
    std::fs::write(store.refs_dir("smoke").join("01.jpg"), b"jpg").unwrap();

    let avatar: Avatar = serde_json::from_value(serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "slug": "smoke",
        "display_name": "smoke",
        "consent_ref": "consent.json",
        "identity": {
            "apparent_age_range": "", "gender_presentation": "", "face_shape": "",
            "skin_tone": "", "eyes": "brown", "brows": "", "nose": "", "lips": "",
            "hair": { "color": "", "length": "", "style": "" },
            "facial_hair": null, "distinguishing_marks": [], "build": ""
        },
        "prompt_blocks": {
            "identity": "", "negative": "", "do_not_change": [],
            "ref_binding": { "gemini_image": null, "omni_video": null, "generic": "same person" }
        },
        "refs": { "primary": "refs/01.jpg", "set": ["refs/01.jpg"] },
        "created_at": "2026-11-12T00:00:00Z",
        "canon_approved_at": null
    }))
    .unwrap();
    let model = IdentityModel {
        face_embeddings: [("refs/01.jpg".to_string(), vec![1.0, 0.0])]
            .into_iter()
            .collect(),
        face_model: None,
        body_model: None,
        identity_threshold: Some(0.72),
    };

    store.save(&avatar, &model).unwrap();
    assert!(store.dir("smoke").join(AVATAR_FILE).is_file());
    assert!(store.dir("smoke").join(MODEL_FILE).is_file());
    assert!(store.dir("smoke").join(REFS_DIR).join("01.jpg").is_file());

    let loaded = store.load("smoke").unwrap();
    assert_eq!(loaded.refs.len(), 1);
    assert!(loaded.refs[0].is_absolute());
    assert_eq!(loaded.avatar.consent_ref, "consent.json");
    assert_eq!(store.list().unwrap(), vec!["smoke".to_string()]);
}

#[test]
fn verification_round_trip_is_explicit() {
    let identity = Identity {
        apparent_age_range: String::new(),
        gender_presentation: String::new(),
        face_shape: String::new(),
        skin_tone: String::new(),
        eyes: "brown".into(),
        brows: String::new(),
        nose: String::new(),
        lips: String::new(),
        hair: avatar_media_mcp::domain::avatar::Hair {
            color: String::new(),
            length: String::new(),
            style: String::new(),
        },
        facial_hair: None,
        distinguishing_marks: vec![],
        build: String::new(),
    };
    let prompt = verification_round_trip(&identity, &["refs/01.jpg".into()]);
    assert!(prompt.contains("verification_json"));
    assert!(prompt.contains("refs/01.jpg"));
    assert!(prompt.contains("brown"));

    let verdict = parse_agent_verification("{\"match_score\": 0.55, \"mismatches\": [\"hair\"]}").unwrap();
    assert_eq!(verdict.match_score, 0.55);
    assert!(parse_agent_verification("{\"match_score\": 2.0, \"mismatches\": []}").is_err());
}

#[test]
fn parsers_never_panic_on_garbage() {
    // Deterministic xorshift: 500 pseudo-random payloads through both parsers.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..500 {
        let len = (next() % 64) as usize;
        let payload: String = (0..len)
            .map(|_| char::from((next() % 96) as u8 + 32))
            .collect();
        let _ = parse_agent_identity(&payload);
        let _ = parse_agent_verification(&payload);
    }
    // Structured near-misses.
    for payload in ["{}", "{\"match_score\":}", "{\"match_score\":null}", "{\"eyes\":1}"] {
        let _ = parse_agent_identity(payload);
        let _ = parse_agent_verification(payload);
    }
}

#[test]
fn preflight_reports_the_real_configuration() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let report = runtime.block_on(preflight());
    assert!(report.models.ok, "models.yaml: {}", report.models.detail);
    assert!(report.platforms.ok, "platforms.yaml: {}", report.platforms.detail);
    assert!(report.settings.ok, "settings.yaml: {}", report.settings.detail);
    assert_eq!(report.keys.len(), 8);
    // The avatars root is created on demand; the report must not lie about it.
    assert!(report.avatars.writable);
}

#[test]
fn migration_declares_the_state_tables() {
    let sql = std::fs::read_to_string(manifest_path("migrations/0001_init.sql")).unwrap();
    for table in [
        "projects",
        "shots",
        "jobs",
        "ledger",
        "identity_scores",
        "m0_results",
    ] {
        assert!(
            sql.contains(&format!("CREATE TABLE IF NOT EXISTS {table}")),
            "migration is missing table {table}"
        );
    }
    assert!(sql.contains("vector(512)"), "identity_scores must stay pgvector-backed");
}

#[test]
fn avatar_json_roundtrips_through_serde() {
    let raw = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "slug": "smoke",
        "display_name": "smoke",
        "consent_ref": "avatars/smoke/consent.json",
        "identity": {
            "apparent_age_range": "30-35",
            "gender_presentation": "androgynous",
            "face_shape": "oval",
            "skin_tone": "olive",
            "eyes": "brown",
            "brows": "straight",
            "nose": "narrow",
            "lips": "medium",
            "hair": { "color": "black", "length": "short", "style": "crop" },
            "facial_hair": null,
            "distinguishing_marks": [],
            "build": "medium"
        },
        "prompt_blocks": {
            "identity": "",
            "negative": "",
            "do_not_change": [],
            "ref_binding": { "gemini_image": null, "omni_video": null, "generic": "same person" }
        },
        "refs": { "primary": "a.png", "set": ["a.png"] },
        "created_at": "2026-11-12T00:00:00Z",
        "canon_approved_at": null
    });
    let avatar: Avatar = serde_json::from_value(raw).unwrap();
    let encoded = serde_json::to_value(&avatar).unwrap();
    let decoded: Avatar = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.identity.eyes, "brown");
    assert_eq!(decoded.schema_version, SCHEMA_VERSION);

    let model = IdentityModel {
        face_embeddings: [("a.png".to_string(), vec![1.0, 0.0])].into_iter().collect(),
        face_model: None,
        body_model: None,
        identity_threshold: Some(0.72),
    };
    let roundtrip: IdentityModel =
        serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
    assert_eq!(roundtrip.identity_threshold, Some(0.72));
}

#[test]
fn consent_gate_blocks_before_touching_models() {
    let args = AvatarCreateArgs {
        photos: vec!["does-not-exist.png".into()],
        slug: "smoke".into(),
        consent: "borrowed".into(),
        text_channel: None,
        identity_json: None,
        verification_json: None,
        force: false,
    };
    let error = avatar_create(&args).unwrap_err().to_string();
    assert!(error.contains("consent"), "unexpected error: {error}");
}

#[test]
fn slug_traversal_is_refused() {
    let args = AvatarCreateArgs {
        photos: vec!["does-not-exist.png".into()],
        slug: "../../etc".into(),
        consent: "self".into(),
        text_channel: None,
        identity_json: None,
        verification_json: None,
        force: false,
    };
    let error = avatar_create(&args).unwrap_err().to_string();
    assert!(error.contains("slug"), "unexpected error: {error}");
}

#[test]
fn stubs_return_errors_instead_of_panicking() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let platforms = PlatformsConfig::from_path(&manifest_path("config/platforms.yaml")).unwrap();
    let profile = platforms.platforms.get("youtube_shorts").unwrap().clone();
    let script = script();

    runtime.block_on(async {
        assert!(audio::synthesize_script(&script, Tier::Standard).await.is_err());
        assert!(audio::align_words_local("a.wav".into(), "text").await.is_err());
        assert!(keyframes::generate_keyframes(&script, &profile, Tier::Standard).await.is_err());
        assert!(shots::render_shots("p1", Tier::Standard).await.is_err());
        assert!(assemble::assemble("p1", "youtube_shorts").await.is_err());
        assert!(assemble::export_package("p1", "youtube_shorts").await.is_err());
        assert!(probe(std::path::Path::new("a.mp4")).await.is_err());
        assert!(qc::qc_keyframe("f.png".into(), &[]).await.is_err());
        assert!(qc::qc_shot("c.mp4".into(), 4.0).await.is_err());
        assert!(qc::qc_final("e.mp4".into()).await.is_err());
    });

    let series_doc = avatar_media_mcp::domain::series::Series {
        id: "s1".into(),
        title: "smoke".into(),
        cast: vec!["smoke".into()],
        style_rules: vec![],
        budget_cap_usd: 10.0,
        episodes: vec![],
        created_at: "2026-11-12T00:00:00Z".into(),
    };
    assert!(series::create_series(series_doc).is_err());
    assert!(series::status("s1").is_err());
    let op = FfmpegOp::Trim {
        input: "a.mp4".into(),
        start_s: 0.0,
        end_s: 1.0,
        output: "b.mp4".into(),
    };
    assert!(op.argv().is_err(), "a bare ffmpeg argv must not look executable");
}

#[test]
fn ffmpeg_availability_is_reported_not_assumed() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let result = runtime.block_on(check_available());
    // The server must boot without ffmpeg; this only asserts the probe answers.
    if let Err(error) = result {
        eprintln!("ffmpeg not available in this environment: {error}");
    }
}

#[test]
fn identity_core_is_deterministic() {
    let thresholds = IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx);
    assert!(matches!(
        stage1_score(CosineScore(0.90), &thresholds),
        Stage1Verdict::Pass(_)
    ));
    assert!(matches!(
        stage1_score(CosineScore(0.65), &thresholds),
        Stage1Verdict::Borderline(_)
    ));
    assert!(matches!(
        stage1_score(CosineScore(0.40), &thresholds),
        Stage1Verdict::Fail(_)
    ));

    let a = vec![1.0f32, 0.0];
    let b = vec![1.0f32, 0.0];
    assert_eq!(cosine_similarity_checked(&a, &b).unwrap().get(), 1.0);
    assert!(cosine_similarity_checked(&a, &[1.0]).is_err());
}

#[test]
fn clustering_separates_two_appearance_states() {
    let unit = |x: f32, y: f32| {
        let norm = (x * x + y * y).sqrt();
        vec![x / norm, y / norm]
    };
    let references = vec![
        Reference { path: "a1".into(), vector: unit(1.0, 0.1) },
        Reference { path: "a2".into(), vector: unit(0.9, 0.2) },
        Reference { path: "b1".into(), vector: unit(0.1, 1.0) },
        Reference { path: "b2".into(), vector: unit(0.2, 0.9) },
    ];
    let summary = cluster_references(&references, 0.85).unwrap();
    assert_eq!(summary.clusters.len(), 2);
    assert!(summary.avatar_threshold.is_some());
    assert!(summary.inter_cluster_mean < 0.5);
}

#[test]
fn text_channel_contract_is_agent_first() {
    assert_eq!(channel_order("agent"), &["agent", "cloud"]);
    assert_eq!(channel_order("cloud"), &["cloud", "agent"]);
    assert_eq!(channel_order("anything"), &["agent", "cloud"]);

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let summary = NumericSummary {
        cluster_count: 1,
        clusters: vec![("a.png".into(), 2, 0.9)],
        inter_cluster_mean: f32::NAN,
        avatar_threshold: Some(0.86),
    };
    let error = runtime
        .block_on(AgentChannel.describe(&["a.png".into()], &summary))
        .unwrap_err()
        .to_string();
    assert!(error.contains("round-trip"));
    assert!(error.contains("a.png"), "the round-trip must name the representatives");
}

#[test]
fn cache_roundtrip_is_atomic() {
    let dir = tempfile::tempdir().unwrap();
    let cache = Cache::new(dir.path().to_path_buf());
    let key = Cache::key("fal", "model", &serde_json::json!({"w": 1}), &["h".into()]);
    assert!(cache.get(&key).is_none());
    let path = cache.put(&key, b"bytes").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"bytes");
    assert_eq!(cache.get(&key).unwrap(), path);
    let leftovers = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
        .count();
    assert_eq!(leftovers, 0);
}

#[test]
fn per_avatar_threshold_reaches_the_qc_gate() {
    let model = IdentityModel {
        face_embeddings: Default::default(),
        face_model: None,
        body_model: None,
        identity_threshold: Some(0.80),
    };
    let thresholds = qc::thresholds_for_avatar(&model, EmbeddingSource::ArcFaceOnnx).unwrap();
    assert_eq!(thresholds.pass.0, 0.80);
    assert!(thresholds.fail.0 <= thresholds.pass.0);
    // A score between the global default (0.72) and this avatar's 0.80 is
    // Borderline for this avatar, not an auto-pass.
    assert!(matches!(
        stage1_score(CosineScore(0.75), &thresholds),
        Stage1Verdict::Borderline(_)
    ));
}

#[test]
fn qc_gates_map_to_actions() {
    let thresholds = IdentityThresholds::for_source(EmbeddingSource::ArcFaceOnnx);
    assert_eq!(
        qc::qc1_score_gate(CosineScore(0.95), &thresholds),
        qc::QcAction::AutoPass
    );
    assert_eq!(
        qc::qc1_score_gate(CosineScore(0.65), &thresholds),
        qc::QcAction::VisionJudge
    );
    assert_eq!(
        qc::qc1_score_gate(CosineScore(0.10), &thresholds),
        qc::QcAction::AutoFail
    );
}

#[test]
fn empty_identity_is_a_valid_document() {
    let identity = Identity {
        apparent_age_range: String::new(),
        gender_presentation: String::new(),
        face_shape: String::new(),
        skin_tone: String::new(),
        eyes: String::new(),
        brows: String::new(),
        nose: String::new(),
        lips: String::new(),
        hair: avatar_media_mcp::domain::avatar::Hair {
            color: String::new(),
            length: String::new(),
            style: String::new(),
        },
        facial_hair: None,
        distinguishing_marks: vec![],
        build: String::new(),
    };
    let encoded = serde_json::to_string(&identity).unwrap();
    let decoded: Identity = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.eyes, identity.eyes);
    assert_eq!(decoded.hair.style, identity.hair.style);
    assert!(decoded.facial_hair.is_none());
}
