//! Text channel — ArcFace-first identity pipeline, phases 2–4: grounded
//! identity description over cluster artifacts, reverse-verified before it
//! reaches prompt blocks.
//! The channel contract stays; transports are the multimodal agent (default,
//! round-trip through `identity_json`) and cloud vision (M0/M2, fal keys).
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::avatar::{Identity, PromptBlocks, RefBinding};
use crate::identity::clusters::ClusterSummary;

/// Numeric summary handed to the channel together with representative images.
/// The channel must ground every claim in these numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericSummary {
    pub cluster_count: usize,
    /// (representative path, member count, min intra-cluster cosine; NaN for
    /// singletons) per cluster.
    pub clusters: Vec<(String, usize, f32)>,
    pub inter_cluster_mean: f32,
    pub avatar_threshold: Option<f32>,
}

impl NumericSummary {
    pub fn from_clusters(
        summary: &ClusterSummary,
        references: &[crate::identity::clusters::Reference],
    ) -> Self {
        Self {
            cluster_count: summary.clusters.len(),
            clusters: summary
                .clusters
                .iter()
                .map(|c| {
                    (
                        references[c.representative].path.clone(),
                        c.members.len(),
                        c.min_internal_cosine,
                    )
                })
                .collect(),
            inter_cluster_mean: summary.inter_cluster_mean,
            avatar_threshold: summary.avatar_threshold,
        }
    }
}

/// Result of reverse verification: does the description match the images?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationVerdict {
    pub match_score: f32,
    pub mismatches: Vec<String>,
}

/// One contract, two transports (agent / cloud). Both return the same
/// `Identity` schema and receive the same numeric summaries.
#[async_trait]
pub trait TextChannel: Send + Sync {
    fn name(&self) -> &'static str;

    /// Phase 2: grounded description. Input: 1–2 representative images per
    /// cluster + the numeric summary. Output: the `Identity` struct.
    async fn describe(
        &self,
        representative_images: &[String],
        summary: &NumericSummary,
    ) -> Result<Identity>;

    /// Phase 4: reverse verification — the same channel checks the
    /// description against the cluster representatives.
    async fn verify(
        &self,
        description: &Identity,
        representative_images: &[String],
    ) -> Result<VerificationVerdict>;
}

/// Channel selection order per config `identity.text_channel`.
pub fn channel_order(configured: &str) -> &'static [&'static str] {
    match configured {
        "cloud" => &["cloud", "agent"],
        // "agent" (default) and anything unknown: agent first.
        _ => &["agent", "cloud"],
    }
}

/// Agent channel: the multimodal IDE agent fills the profile (verified
/// working in this environment). The server cannot block on the operator —
/// `describe` fails with the exact prompt + numeric summary, the agent fills
/// `identity_json` on the follow-up `avatar_create` call, and `verify` is the
/// agent's own judgement recorded in the report.
pub struct AgentChannel;

#[async_trait]
impl TextChannel for AgentChannel {
    fn name(&self) -> &'static str {
        "agent"
    }

    async fn describe(
        &self,
        representative_images: &[String],
        summary: &NumericSummary,
    ) -> Result<Identity> {
        anyhow::bail!(
            "agent text channel requires a round-trip: inspect the representative images, then retry avatar_create with identity_json. Representatives: {}. Numeric summary: {}",
            representative_images.join(", "),
            serde_json::to_string(summary)?
        )
    }

    async fn verify(
        &self,
        _description: &Identity,
        representative_images: &[String],
    ) -> Result<VerificationVerdict> {
        // The server cannot verify on the agent's behalf: it has no eyes and
        // cannot block on the operator. Returning a synthetic 1.0 here would
        // make every description "verified" without anyone looking at it.
        anyhow::bail!(
            "{}",
            verification_round_trip(_description, representative_images)
        )
    }
}

/// The second round-trip: the agent checks its own description against the
/// same representatives and returns `verification_json`.
pub fn verification_round_trip(description: &Identity, representative_images: &[String]) -> String {
    format!(
        "agent text channel requires a verification round-trip: compare the description below against the representative images, then retry avatar_create with verification_json. Representatives: {}. Description: {}. Return {{\"match_score\": 0.0-1.0, \"mismatches\": [\"...\"]}}.",
        representative_images.join(", "),
        serde_json::to_string(description).unwrap_or_default()
    )
}

/// Cloud channel placeholder (fal vision) — M2 when fal keys land.
pub struct CloudChannel;

#[async_trait]
impl TextChannel for CloudChannel {
    fn name(&self) -> &'static str {
        "cloud"
    }

    async fn describe(
        &self,
        _representative_images: &[String],
        _summary: &NumericSummary,
    ) -> Result<Identity> {
        anyhow::bail!("cloud vision channel lands with fal keys (M0/M2)")
    }

    async fn verify(
        &self,
        _description: &Identity,
        _representative_images: &[String],
    ) -> Result<VerificationVerdict> {
        anyhow::bail!("cloud vision channel lands with fal keys (M0/M2)")
    }
}

/// Phase 4 gate: regenerate ≤2 times, then flag the operator.
pub const VERIFY_PASS_THRESHOLD: f32 = 0.8;
pub const MAX_REGENERATIONS: usize = 2;

/// Closed loop: describe → verify → regenerate ≤2 → flag. Used by transports
/// that can verify programmatically (cloud); the agent transport verifies by
/// construction.
pub async fn grounded_description(
    channel: &dyn TextChannel,
    representative_images: &[String],
    summary: &NumericSummary,
) -> Result<(Identity, VerificationVerdict, bool)> {
    let mut last_verdict = None;
    for _attempt in 0..=MAX_REGENERATIONS {
        let identity = channel.describe(representative_images, summary).await?;
        let verdict = channel.verify(&identity, representative_images).await?;
        if verdict.match_score >= VERIFY_PASS_THRESHOLD {
            return Ok((identity, verdict, false));
        }
        last_verdict = Some((identity, verdict));
    }
    let (identity, verdict) = last_verdict.context("verification loop produced no verdict")?;
    Ok((identity, verdict, true))
}

/// Traits that legitimately vary between appearance states. With more than one
/// cluster these are barred from `do_not_change`: pinning them would make the
/// prompt contradict half the reference set.
const VARIABLE_TRAITS: [&str; 6] = [
    "hair",
    "hairstyle",
    "beard",
    "facial hair",
    "makeup",
    "glasses",
];

pub fn filter_do_not_change(entries: &[String], summary: &ClusterSummary) -> Vec<String> {
    let multi_state = summary.clusters.len() > 1;
    entries
        .iter()
        .filter(|entry| {
            if !multi_state {
                return true;
            }
            let entry = entry.to_lowercase();
            !VARIABLE_TRAITS.iter().any(|trait_name| entry.contains(trait_name))
        })
        .cloned()
        .collect()
}

fn trait_line(label: &str, value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| format!("{label}: {value}"))
}

/// Assemble prompt blocks from a validated identity and the cluster summary.
/// The identity block is a deterministic rendering of the description — no
/// free prose, so every clause traces back to a field the channel filled.
/// `do_not_change` carries only cluster-stable traits; hair is dropped when
/// the reference set has more than one appearance state.
pub fn build_prompt_blocks(identity: &Identity, summary: &ClusterSummary) -> PromptBlocks {
    let mut lines = Vec::new();
    for (label, value) in [
        ("apparent age", identity.apparent_age_range.as_str()),
        ("gender presentation", identity.gender_presentation.as_str()),
        ("face shape", identity.face_shape.as_str()),
        ("skin tone", identity.skin_tone.as_str()),
        ("eyes", identity.eyes.as_str()),
        ("brows", identity.brows.as_str()),
        ("nose", identity.nose.as_str()),
        ("lips", identity.lips.as_str()),
        ("hair color", identity.hair.color.as_str()),
        ("hair length", identity.hair.length.as_str()),
        ("hair style", identity.hair.style.as_str()),
        ("facial hair", identity.facial_hair.as_deref().unwrap_or("")),
        ("build", identity.build.as_str()),
    ] {
        if let Some(line) = trait_line(label, value) {
            lines.push(line);
        }
    }
    if !identity.distinguishing_marks.is_empty() {
        lines.push(format!(
            "distinguishing marks: {}",
            identity.distinguishing_marks.join(", ")
        ));
    }
    let identity_block = if lines.is_empty() {
        String::new()
    } else {
        format!("The same person in every frame. {}", lines.join("; "))
    };

    let mut stable = Vec::new();
    for (label, value) in [
        ("face shape", identity.face_shape.as_str()),
        ("eyes", identity.eyes.as_str()),
        ("nose", identity.nose.as_str()),
        ("lips", identity.lips.as_str()),
        ("skin tone", identity.skin_tone.as_str()),
        ("hair color", identity.hair.color.as_str()),
        ("hair style", identity.hair.style.as_str()),
        ("build", identity.build.as_str()),
    ] {
        if let Some(line) = trait_line(label, value) {
            stable.push(line);
        }
    }
    stable.extend(
        identity
            .distinguishing_marks
            .iter()
            .map(|mark| format!("distinguishing mark: {mark}")),
    );

    PromptBlocks {
        identity: identity_block,
        negative: "different person, altered face shape, changed eye colour, beauty-filter skin, extra fingers, deformed hands".into(),
        do_not_change: filter_do_not_change(&stable, summary),
        ref_binding: RefBinding {
            gemini_image: None,
            omni_video: None,
            generic: "same person as in the reference image(s)".into(),
        },
    }
}

/// Minimal base64 (no external crate; alphabet + padding per RFC 4648).
pub fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(triple >> 18) as usize & 63] as char);
        out.push(ALPHABET[(triple >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(triple >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[triple as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn parse_identity_json(raw: &str) -> Result<Identity> {
    // The model may wrap the object in prose; take the outermost braces.
    let start = raw.find('{').context("no JSON object in channel response")?;
    let end = raw.rfind('}').context("no JSON object in channel response")?;
    // A payload like "} ... {" has end < start; slicing would panic.
    anyhow::ensure!(start < end, "no JSON object in channel response");
    let identity: Identity = serde_json::from_str(&raw[start..=end])?;
    Ok(identity)
}

/// Parse an agent-supplied identity JSON (the round-trip payload).
pub fn parse_agent_identity(raw: &str) -> Result<Identity> {
    parse_identity_json(raw)
}

/// Parse the agent's verification payload. `match_score` must be a probability:
/// a channel that answers 1.5 or NaN is rejected, not clamped.
pub fn parse_agent_verification(raw: &str) -> Result<VerificationVerdict> {
    let start = raw.find('{').context("no JSON object in verification response")?;
    let end = raw.rfind('}').context("no JSON object in verification response")?;
    anyhow::ensure!(start < end, "no JSON object in verification response");
    let verdict: VerificationVerdict = serde_json::from_str(&raw[start..=end])?;
    anyhow::ensure!(
        verdict.match_score.is_finite() && (0.0..=1.0).contains(&verdict.match_score),
        "verification match_score must be in [0, 1], got {}",
        verdict.match_score
    );
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn parse_identity_tolerates_wrapped_json() {
        let raw = "Here you go: {\"apparent_age_range\":\"30-35\",\"gender_presentation\":\"male\",\"face_shape\":\"oval\",\"skin_tone\":\"fair\",\"eyes\":\"brown\",\"brows\":\"thick\",\"nose\":\"straight\",\"lips\":\"medium\",\"hair\":{\"color\":\"dark\",\"length\":\"short\",\"style\":\"crop\"},\"facial_hair\":null,\"distinguishing_marks\":[],\"build\":\"medium\"} — done.";
        let identity = parse_identity_json(raw).unwrap();
        assert_eq!(identity.hair.color, "dark");
        assert!(identity.facial_hair.is_none());
    }

    #[test]
    fn do_not_change_drops_variable_traits_but_keeps_stable_ones() {
        let summary = crate::identity::clusters::ClusterSummary {
            // Two appearance states → hair entries are barred.
            clusters: vec![dummy_cluster(), dummy_cluster()],
            inter_cluster_mean: 0.0,
            intra_cluster_mean: 0.0,
            avatar_threshold: None,
        };
        let entries = vec![
            "brown eyes".into(),
            "dark hair".into(),
            "straight nose".into(),
        ];
        let filtered = filter_do_not_change(&entries, &summary);
        assert!(filtered.contains(&"brown eyes".to_string()));
        assert!(filtered.contains(&"straight nose".to_string()));
        assert!(!filtered.contains(&"dark hair".to_string()));
    }

    fn dummy_cluster() -> crate::identity::clusters::ReferenceCluster {
        crate::identity::clusters::ReferenceCluster {
            members: vec![0],
            representative: 0,
            centroid: vec![1.0],
            min_internal_cosine: f32::NAN,
        }
    }

    #[tokio::test]
    async fn agent_channel_errors_with_round_trip_message() {
        let channel = AgentChannel;
        let summary = NumericSummary {
            cluster_count: 1,
            clusters: vec![],
            inter_cluster_mean: 0.0,
            avatar_threshold: None,
        };
        let err = channel.describe(&[], &summary).await.unwrap_err();
        assert!(err.to_string().contains("round-trip"));
    }

    #[tokio::test]
    async fn agent_channel_refuses_to_verify_on_the_agents_behalf() {
        let channel = AgentChannel;
        let mut identity = empty_identity();
        identity.eyes = "brown".into();
        let error = channel
            .verify(&identity, &["x.jpg".to_string()])
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("verification round-trip"), "{error}");
        assert!(error.contains("verification_json"), "{error}");
        assert!(error.contains("x.jpg"), "{error}");
    }

    #[test]
    fn verification_payload_is_validated() {
        let ok = parse_agent_verification(
            "checked: {\"match_score\": 0.62, \"mismatches\": [\"hair colour\"]} done",
        )
        .unwrap();
        assert_eq!(ok.match_score, 0.62);
        assert_eq!(ok.mismatches, vec!["hair colour".to_string()]);
        assert!(parse_agent_verification("{\"match_score\": 1.5, \"mismatches\": []}").is_err());
        assert!(parse_agent_verification("{\"match_score\": -0.1, \"mismatches\": []}").is_err());
        assert!(parse_agent_verification("no json here").is_err());
    }

    fn empty_identity() -> Identity {
        Identity {
            apparent_age_range: String::new(),
            gender_presentation: String::new(),
            face_shape: String::new(),
            skin_tone: String::new(),
            eyes: String::new(),
            brows: String::new(),
            nose: String::new(),
            lips: String::new(),
            hair: crate::domain::avatar::Hair {
                color: String::new(),
                length: String::new(),
                style: String::new(),
            },
            facial_hair: None,
            distinguishing_marks: vec![],
            build: String::new(),
        }
    }

    fn summary_with(clusters: usize) -> ClusterSummary {
        ClusterSummary {
            clusters: (0..clusters).map(|_| dummy_cluster()).collect(),
            inter_cluster_mean: 0.0,
            intra_cluster_mean: 0.0,
            avatar_threshold: None,
        }
    }

    #[test]
    fn prompt_blocks_render_only_filled_fields() {
        let blocks = build_prompt_blocks(&empty_identity(), &summary_with(1));
        assert!(blocks.identity.is_empty(), "empty identity must not invent prose");
        assert!(blocks.do_not_change.is_empty());
        assert!(!blocks.negative.is_empty());

        let mut identity = empty_identity();
        identity.eyes = "brown".into();
        identity.hair.color = "dark".into();
        let single = build_prompt_blocks(&identity, &summary_with(1));
        assert!(single.identity.contains("eyes: brown"));
        assert!(single.identity.contains("hair color: dark"));
        assert!(single.do_not_change.iter().any(|entry| entry.contains("eyes")));
        assert!(single.do_not_change.iter().any(|entry| entry.contains("hair")));

        // Two appearance states: hair is the explicitly variable trait.
        let multi = build_prompt_blocks(&identity, &summary_with(2));
        assert!(multi.do_not_change.iter().any(|entry| entry.contains("eyes")));
        assert!(!multi.do_not_change.iter().any(|entry| entry.contains("hair")));
    }
}
