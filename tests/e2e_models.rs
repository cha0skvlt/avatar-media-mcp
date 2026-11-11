//! Full `avatar_create` on real photos. Ignored by default: it needs the ONNX
//! weights and the ORT dylib.
//!
//! Run:
//! ```text
//! AVATAR_E2E_PHOTOS=/path/to/photos \
//! ORT_DYLIB_PATH=models/ort/lib/libonnxruntime.1.31.0.dylib \
//! cargo test --features local-identity --test e2e_models -- --ignored --nocapture
//! ```
//!
//! cwd must be the crate root: the pipeline resolves `models/` and `avatars/`
//! relative to it.

use avatar_media_mcp::infra::avatar_store::AvatarStore;
use avatar_media_mcp::tools::avatar::{AvatarCreateArgs, avatar_create};

fn photos() -> Vec<String> {
    let dir = std::env::var("AVATAR_E2E_PHOTOS")
        .expect("set AVATAR_E2E_PHOTOS to a directory with 2+ photos of one person");
    let mut paths: Vec<String> = std::fs::read_dir(&dir)
        .expect("read AVATAR_E2E_PHOTOS")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
        })
        .map(|path| path.display().to_string())
        .collect();
    paths.sort();
    paths.truncate(6);
    assert!(paths.len() >= 2, "need at least two photos in {dir}");
    paths
}

fn args(slug: &str, identity: Option<&str>, verification: Option<&str>) -> AvatarCreateArgs {
    AvatarCreateArgs {
        photos: photos(),
        slug: slug.into(),
        consent: "self".into(),
        text_channel: Some("agent".into()),
        identity_json: identity.map(str::to_string),
        verification_json: verification.map(str::to_string),
        force: true,
    }
}

#[test]
#[ignore = "needs models/ + ORT_DYLIB_PATH + AVATAR_E2E_PHOTOS"]
fn avatar_create_three_call_flow_writes_a_self_contained_avatar() {
    let slug = format!("e2e-{}", std::process::id());
    let store = AvatarStore::default();
    let _ = std::fs::remove_dir_all(store.dir(&slug));

    // Call 1: no identity → the describe round-trip names the representatives.
    let error = avatar_create(&args(&slug, None, None)).unwrap_err().to_string();
    assert!(error.contains("identity_json"), "unexpected error: {error}");
    assert!(error.contains("refs/"), "the round-trip must name the representatives: {error}");

    // Call 2: identity but no verification → the verification round-trip, no write.
    let identity = r#"{"apparent_age_range":"30-35","gender_presentation":"androgynous",
        "face_shape":"oval","skin_tone":"olive","eyes":"brown","brows":"straight",
        "nose":"narrow","lips":"medium","hair":{"color":"black","length":"short","style":"crop"},
        "facial_hair":null,"distinguishing_marks":[],"build":"medium"}"#;
    let error = avatar_create(&args(&slug, Some(identity), None))
        .unwrap_err()
        .to_string();
    assert!(error.contains("verification_json"), "unexpected error: {error}");
    assert!(!store.dir(&slug).join("avatar.json").exists(), "nothing may be written before verification");

    // Call 3: both artifacts → the avatar is written and loads back.
    let report = avatar_create(&args(
        &slug,
        Some(identity),
        Some(r#"{"match_score":0.9,"mismatches":[]}"#),
    ))
    .expect("avatar_create");
    assert!(report.clusters >= 1);
    assert!(!report.verification_flagged);
    assert!(report.avatar_threshold.is_some(), "warnings: {:?}", report.warnings);

    let loaded = store.load(&slug).expect("load");
    assert_eq!(loaded.avatar.slug, slug);
    assert!(!loaded.refs.is_empty());
    for reference in &loaded.refs {
        assert!(reference.is_file(), "ref {} missing", reference.display());
        assert!(reference.starts_with(store.dir(&slug)), "ref outside the avatar dir");
    }
    assert!(loaded.avatar.consent_ref == "consent.json");
    assert!(!loaded.avatar.prompt_blocks.identity.is_empty());

    std::fs::remove_dir_all(store.dir(&slug)).expect("cleanup");
}
