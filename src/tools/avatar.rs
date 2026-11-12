//! avatar_* tool group: the M1 end-to-end path
//! `avatar_create → normalize → quality → embed → cluster → text → verify → swap in`.

use std::path::Path;

use anyhow::Context;

use crate::identity::clusters::{self, Reference};
use crate::identity::textpass::{self, AgentChannel, CloudChannel, NumericSummary, TextChannel};
use crate::infra::avatar_store::AvatarStore;

/// Arguments of `avatar_create`.
#[derive(Debug, Clone, serde::Deserialize, rmcp::schemars::JsonSchema)]
pub struct AvatarCreateArgs {
    /// Absolute paths of reference photos (1–30).
    pub photos: Vec<String>,
    /// Avatar slug (filesystem-safe, e.g. `eugene`).
    pub slug: String,
    /// Text channel: `agent` (default), `cloud` (M0/M2).
    #[serde(default)]
    pub text_channel: Option<String>,
    /// Agent round-trip 1: the identity JSON the agent filled in by inspecting
    /// the cluster representatives (from the first round-trip error).
    #[serde(default)]
    pub identity_json: Option<String>,
    /// Agent round-trip 2: `{"match_score": 0..1, "mismatches": [...]}` — the
    /// agent's own check of that description against the same representatives.
    /// Without it nothing is written: an unverified description never becomes
    /// a canon.
    #[serde(default)]
    pub verification_json: Option<String>,
    /// Replace an avatar whose canon was already approved by the operator.
    #[serde(default)]
    pub force: bool,
    /// Operator override: write the avatar even when the verification score is
    /// below the pass threshold. The agent must not set this on its own; the
    /// default path rejects a failed verification and writes nothing.
    #[serde(default)]
    pub accept_flagged: bool,
}

/// A normal step of the agent round-trip, not a failure.
///
/// `avatar_create` is a multi-call protocol: the server cannot look at the
/// pictures, so it hands the agent a prompt and waits for the next call. It
/// travels through `anyhow::Error` (so `avatar_create`'s signature and every
/// existing `.unwrap_err().to_string()` caller keep working), and the MCP layer
/// downcasts it into an ordinary successful tool result instead of a JSON-RPC
/// error. `Display` prints only `message`, exactly the text the old `bail!` had.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStep {
    /// `needs_identity` | `needs_verification` | `verification_failed`.
    pub status: &'static str,
    /// What the server needs from the agent, or why the write was refused.
    pub message: String,
    /// The exact next call to make.
    pub next: &'static str,
}

impl AgentStep {
    pub fn needs_identity(message: String) -> Self {
        Self {
            status: "needs_identity",
            message,
            next: "Inspect the representative images, then call avatar_create again with the same arguments plus identity_json.",
        }
    }

    pub fn needs_verification(message: String) -> Self {
        Self {
            status: "needs_verification",
            message,
            next: "Compare the description with the representative images, then call avatar_create again with the same arguments plus verification_json.",
        }
    }

    pub fn verification_failed(score: f32, mismatches: &[String]) -> Self {
        let detail = if mismatches.is_empty() {
            "no mismatches were listed".to_string()
        } else {
            mismatches.join("; ")
        };
        Self {
            status: "verification_failed",
            message: format!(
                "verification score {score:.2} is below {:.2}; nothing was written. Mismatches: {detail}",
                textpass::VERIFY_PASS_THRESHOLD
            ),
            next: "Fix identity_json to address the mismatches, then call avatar_create again WITHOUT verification_json to get a fresh verification round-trip. Only the operator may pass accept_flagged=true to write anyway.",
        }
    }
}

impl std::fmt::Display for AgentStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for AgentStep {}

/// The verification gate. Returns `Ok(flagged)`; a score under the pass
/// threshold is an error unless the operator passed `accept_flagged`.
/// The score itself is the agent's own claim (the server has no eyes), so this
/// gate enforces the protocol, not the truth of the claim.
fn gate_verification(
    verdict: &textpass::VerificationVerdict,
    accept_flagged: bool,
    warnings: &mut Vec<String>,
) -> Result<bool, AgentStep> {
    if verdict.match_score >= textpass::VERIFY_PASS_THRESHOLD {
        return Ok(false);
    }
    if !accept_flagged {
        return Err(AgentStep::verification_failed(
            verdict.match_score,
            &verdict.mismatches,
        ));
    }
    warnings.push(format!(
        "verification {:.2} is below {:.2}; written by operator override (accept_flagged=true): {}",
        verdict.match_score,
        textpass::VERIFY_PASS_THRESHOLD,
        verdict.mismatches.join("; ")
    ));
    Ok(true)
}

/// Result of one `avatar_create` call.
#[derive(Debug, serde::Serialize)]
pub struct AvatarCreateReport {
    pub slug: String,
    pub avatar_path: String,
    pub clusters: usize,
    pub cluster_reps: Vec<String>,
    pub avatar_threshold: Option<f32>,
    pub skipped_images: Vec<String>,
    pub duplicate_images: Vec<String>,
    pub text_channel_used: String,
    pub verification: textpass::VerificationVerdict,
    pub verification_flagged: bool,
    pub warnings: Vec<String>,
}

/// The whole M1 pipeline for one avatar. Blocking (ONNX) work runs here;
/// the MCP layer wraps it in `spawn_blocking`.
pub fn avatar_create(args: &AvatarCreateArgs) -> anyhow::Result<AvatarCreateReport> {
    AvatarStore::validate_slug(&args.slug)?;
    anyhow::ensure!(
        !args.photos.is_empty() && args.photos.len() <= 30,
        "photos must be 1..=30 paths"
    );

    let store = AvatarStore::default();
    let live_dir = store.dir(&args.slug);

    // An approved canon is the operator's decision: replacing it must be explicit.
    if !args.force {
        let avatar_path = live_dir.join(crate::infra::avatar_store::AVATAR_FILE);
        if avatar_path.is_file() {
            let existing: crate::domain::avatar::Avatar =
                serde_json::from_str(&std::fs::read_to_string(&avatar_path)?)
                    .with_context(|| format!("parse {}", avatar_path.display()))?;
            anyhow::ensure!(
                existing.canon_approved_at.is_none(),
                "avatar {} has an approved canon ({}); pass force=true to replace it",
                args.slug,
                existing.canon_approved_at.unwrap_or_default()
            );
        }
    }

    // Phase 1a: normalize into a staging directory — the canon must live inside
    // the avatar directory, and a failed rebuild must not touch the live one.
    let staging = store.begin_staging(&args.slug)?;
    let refs_dir = staging.refs_dir();
    let mut normalized = Vec::new();
    let mut skipped = Vec::new();
    let mut warnings = Vec::new();
    for photo in &args.photos {
        let target = refs_dir.join(format!("{:02}.jpg", normalized.len() + 1));
        if let Err(error) = crate::pipeline::avatar_build::normalize_photo(Path::new(photo), &target)
        {
            skipped.push(format!("{photo}: {error}"));
            continue;
        }
        // The quality gate is not advisory: a 300px or near-black photo cannot
        // carry an identity lock.
        match crate::pipeline::avatar_build::check_photo(&target) {
            Ok(remarks) if remarks.is_empty() => normalized.push(target),
            Ok(remarks) => {
                let _ = std::fs::remove_file(&target);
                skipped.push(format!("{photo}: {}", remarks.join("; ")));
            }
            Err(error) => {
                let _ = std::fs::remove_file(&target);
                skipped.push(format!("{photo}: {error}"));
            }
        }
    }
    anyhow::ensure!(
        !normalized.is_empty(),
        "no reference photo passed normalization + quality; skipped: {skipped:?}"
    );

    // Phase 1b: numeric lock. embed → dedupe → clusters.
    let (vectors, unreadable) =
        futures::executor::block_on(crate::identity::embeddings::compute_embeddings(&normalized))?;
    skipped.extend(unreadable);
    let (vectors, duplicates) = dedupe_vectors(vectors);
    let references: Vec<Reference> = vectors
        .into_iter()
        .map(|(path, vector)| Reference { path, vector })
        .collect();
    let threshold = clusters::auto_threshold(&references);
    let summary =
        clusters::cluster_references(&references, threshold).context("cluster references")?;
    if summary.avatar_threshold.is_none() {
        warnings.push(
            "no per-avatar threshold: every reference is its own appearance state; QC will fall back to defaults"
                .into(),
        );
    }

    // Phase 2–4: grounded text over cluster artifacts.
    let numeric = NumericSummary::from_clusters(&summary, &references);
    let representative_images: Vec<String> = numeric
        .clusters
        .iter()
        .map(|(path, _, _)| path.clone())
        .collect();

    let order = textpass::channel_order(args.text_channel.as_deref().unwrap_or("agent"));
    let mut identity = None;
    let mut text_channel_used = "none".to_string();
    let mut round_trip = None;

    for &channel_name in order {
        match channel_name {
            "agent" => {
                if let Some(json) = &args.identity_json {
                    identity = Some(textpass::parse_agent_identity(json)?);
                    text_channel_used = "agent".into();
                    break;
                }
                // Remember the request and let the next channel try: a missing
                // identity_json must not make the cloud transport unreachable.
                round_trip = Some(
                    futures::executor::block_on(
                        AgentChannel.describe(&representative_images, &numeric),
                    )
                    .unwrap_err()
                    .to_string(),
                );
            }
            "cloud" => {
                if let Ok(id) = futures::executor::block_on(
                    CloudChannel.describe(&representative_images, &numeric),
                ) {
                    identity = Some(id);
                    text_channel_used = "cloud".into();
                    break;
                }
            }
            _ => {}
        }
    }

    let Some(identity) = identity else {
        let error: anyhow::Error = match round_trip {
            Some(message) => AgentStep::needs_identity(message).into(),
            None => anyhow::anyhow!("no text channel produced an identity"),
        };
        return Err(error);
    };
    anyhow::ensure!(
        !identity.is_empty(),
        "the identity description is empty: an avatar without a description is not a canon"
    );
    if identity.filled_fields() < 3 {
        warnings.push(format!(
            "thin description: only {} identity fields are filled",
            identity.filled_fields()
        ));
    }

    // Phase 4: reverse verification. The agent transport cannot be verified by
    // the server, so the agent's own verdict is required before any write.
    let verification = match text_channel_used.as_str() {
        "agent" => match &args.verification_json {
            Some(raw) => textpass::parse_agent_verification(raw)?,
            None => {
                return Err(AgentStep::needs_verification(
                    textpass::verification_round_trip(&identity, &representative_images),
                )
                .into());
            }
        },
        "cloud" => futures::executor::block_on(
            CloudChannel.verify(&identity, &representative_images),
        )?,
        other => anyhow::bail!("unknown text channel {other}"),
    };
    // A failed verification must never become a canon: refuse before anything
    // is committed (the staging directory is dropped and removed on return).
    let verification_flagged =
        gate_verification(&verification, args.accept_flagged, &mut warnings)?;

    // Every path inside avatar.json is relative to the avatar directory.
    let relative = |path: &str| -> String {
        Path::new(path)
            .strip_prefix(staging.dir())
            .map(|rest| rest.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string())
    };
    let refs: Vec<String> = representative_images.iter().map(|p| relative(p)).collect();

    let avatar = crate::domain::avatar::Avatar {
        schema_version: crate::domain::avatar::SCHEMA_VERSION,
        slug: args.slug.clone(),
        display_name: args.slug.clone(),
        identity: identity.clone(),
        persona: None,
        voice: None,
        prompt_blocks: textpass::build_prompt_blocks(&identity, &summary),
        refs: crate::domain::avatar::Refs {
            primary: refs.first().cloned().unwrap_or_default(),
            set: refs,
        },
        created_at: chrono::Utc::now().to_rfc3339(),
        canon_approved_at: None,
    };
    let model = crate::domain::avatar::IdentityModel {
        face_embeddings: references
            .iter()
            .map(|reference| (relative(&reference.path), reference.vector.clone()))
            .collect(),
        face_model: Some(serde_json::json!({"name": "buffalo_l/w600k_r50"})),
        body_model: None,
        identity_threshold: summary.avatar_threshold,
    };
    let avatar_path = staging.commit(&avatar, &model)?;

    Ok(AvatarCreateReport {
        slug: args.slug.clone(),
        avatar_path: avatar_path.display().to_string(),
        clusters: summary.clusters.len(),
        cluster_reps: representative_images,
        avatar_threshold: summary.avatar_threshold,
        skipped_images: skipped,
        duplicate_images: duplicates,
        text_channel_used,
        verification,
        verification_flagged,
        warnings,
    })
}

/// Drop near-identical reference vectors (cosine > 0.9999) so a burst of
/// frames from one photo cannot inflate a cluster's spread.
fn dedupe_vectors(
    vectors: std::collections::BTreeMap<String, Vec<f32>>,
) -> (std::collections::BTreeMap<String, Vec<f32>>, Vec<String>) {
    let mut kept: std::collections::BTreeMap<String, Vec<f32>> = std::collections::BTreeMap::new();
    let mut duplicates = Vec::new();
    for (path, vector) in vectors {
        let duplicate = kept.values().any(|existing| {
            crate::identity::embeddings::cosine_similarity_checked(existing, &vector)
                .map(|score| score.get() > 0.9999)
                .unwrap_or(false)
        });
        if duplicate {
            duplicates.push(path);
        } else {
            kept.insert(path, vector);
        }
    }
    (kept, duplicates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(slug: &str) -> AvatarCreateArgs {
        AvatarCreateArgs {
            photos: vec!["missing.png".into()],
            slug: slug.into(),
            text_channel: None,
            identity_json: None,
            verification_json: None,
            force: false,
            accept_flagged: false,
        }
    }

    fn verdict(score: f32, mismatches: &[&str]) -> textpass::VerificationVerdict {
        textpass::VerificationVerdict {
            match_score: score,
            mismatches: mismatches.iter().map(|m| m.to_string()).collect(),
        }
    }

    #[test]
    fn gate_passes_at_or_above_the_threshold() {
        let mut warnings = Vec::new();
        let flagged =
            gate_verification(&verdict(textpass::VERIFY_PASS_THRESHOLD, &[]), false, &mut warnings)
                .unwrap();
        assert!(!flagged);
        assert!(warnings.is_empty());
    }

    #[test]
    fn gate_refuses_a_failed_verification_by_default() {
        let mut warnings = Vec::new();
        let step = gate_verification(&verdict(0.3, &["hair colour", "age"]), false, &mut warnings)
            .unwrap_err();
        assert_eq!(step.status, "verification_failed");
        assert!(step.message.contains("hair colour; age"), "{}", step.message);
        assert!(step.message.contains("nothing was written"));
        assert!(warnings.is_empty());
    }

    #[test]
    fn gate_explains_an_empty_mismatch_list() {
        let mut warnings = Vec::new();
        let step = gate_verification(&verdict(0.1, &[]), false, &mut warnings).unwrap_err();
        assert!(step.message.contains("no mismatches were listed"));
    }

    #[test]
    fn operator_override_writes_but_stays_flagged() {
        let mut warnings = Vec::new();
        let flagged = gate_verification(&verdict(0.3, &["age"]), true, &mut warnings).unwrap();
        assert!(flagged);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("accept_flagged"));
    }

    #[test]
    fn agent_step_survives_the_anyhow_round_trip() {
        // The MCP layer relies on this downcast to tell a protocol step from a failure.
        let error: anyhow::Error = AgentStep::needs_identity("describe me".into()).into();
        let step = error.downcast_ref::<AgentStep>().expect("downcast");
        assert_eq!(step.status, "needs_identity");
        assert_eq!(error.to_string(), "describe me");
    }

    #[test]
    fn slug_rejects_path_traversal() {
        assert!(AvatarStore::validate_slug("eugene").is_ok());
        assert!(AvatarStore::validate_slug("a-b_1").is_ok());
        for bad in ["", "../x", "a/b", "A", "a b", ".", "..", "x\\y"] {
            assert!(AvatarStore::validate_slug(bad).is_err(), "accepted {bad:?}");
        }
        assert!(AvatarStore::validate_slug(&"x".repeat(65)).is_err());
    }

    #[test]
    fn dedupe_drops_near_identical_vectors() {
        let mut vectors = std::collections::BTreeMap::new();
        vectors.insert("a".to_string(), vec![1.0, 0.0]);
        vectors.insert("b".to_string(), vec![1.0, 0.0]);
        vectors.insert("c".to_string(), vec![0.0, 1.0]);
        let (kept, duplicates) = dedupe_vectors(vectors);
        assert_eq!(kept.len(), 2);
        assert_eq!(duplicates, vec!["b".to_string()]);
    }

    #[test]
    fn photo_count_is_bounded() {
        let mut too_many = args("smoke");
        too_many.photos = (0..31).map(|i| format!("{i}.png")).collect();
        let error = avatar_create(&too_many).unwrap_err().to_string();
        assert!(error.contains("1..=30"), "unexpected error: {error}");
    }
}
