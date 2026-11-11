//! Two-stage QC orchestration.
//!
//! The pure helpers below implement QC-1/QC-2 decisions and are independent of
//! database, frame extraction, and model-provider I/O. Existing async entry
//! points are retained for the rest of the server; the supplied archive omits
//! the provider/decoder adapters required to implement those I/O boundaries.
//! They therefore return explicit errors instead of panicking or fabricating
//! a judge verdict.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::identity::embeddings::{
    CosineScore, FaceEmbedding, compare_embeddings,
};
use crate::identity::scoring::{
    AssignmentConfig, AssignmentResult, ClipAggregationConfig, ClipAggregationResult,
    ClipFrameScore, IdentityThresholds, Stage1Verdict, aggregate_clip_frames,
    assign_faces_to_canons, stage1_embedding,
};
use crate::identity::scoring::FaceViewAngle;

/// QC thresholds for one avatar: the stored per-avatar pass threshold applied
/// to the encoder space. This is the only consumer of
/// `IdentityModel.identity_threshold` — without it the field is written and
/// never read, and every avatar is judged by the global default.
pub fn thresholds_for_avatar(
    model: &crate::domain::avatar::IdentityModel,
    source: crate::identity::embeddings::EmbeddingSource,
) -> Result<IdentityThresholds> {
    IdentityThresholds::with_avatar_pass_threshold(source, model.identity_threshold)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QcAction {
    /// Local numerical stage is sufficiently confident; do not spend on judge.
    AutoPass,
    /// Local score is below the reject boundary with sufficient confidence.
    AutoFail,
    /// Borderline / conflicting / under-calibrated: invoke the vision judge.
    VisionJudge,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Qc1IdentityResult {
    pub similarity: CosineScore,
    pub stage1: Stage1Verdict,
    pub action: QcAction,
}

/// A timestamped, source-tagged face embedding extracted from a clip frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedFaceEmbedding {
    pub timestamp_seconds: f64,
    pub embedding: FaceEmbedding,
    pub view_angle: FaceViewAngle,
    pub marked_artifact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qc2IdentityResult {
    pub aggregation: ClipAggregationResult,
    pub action: QcAction,
}

fn action_for_stage1(verdict: Stage1Verdict) -> QcAction {
    match verdict {
        Stage1Verdict::Pass(_) => QcAction::AutoPass,
        Stage1Verdict::Borderline(_) => QcAction::VisionJudge,
        Stage1Verdict::Fail(_) => QcAction::AutoFail,
    }
}

/// QC-1 local stage: checked cosine, source compatibility, then calibrated
/// per-avatar thresholds. Judge invocation is a pure decision returned to the
/// outer orchestrator (not performed inside the math core).
pub fn qc1_identity_gate(
    candidate: &FaceEmbedding,
    canon: &FaceEmbedding,
    thresholds: &IdentityThresholds,
) -> Result<Qc1IdentityResult> {
    let similarity = compare_embeddings(candidate, canon)?;
    let stage1 = stage1_embedding(similarity, candidate.source, thresholds)?;
    Ok(Qc1IdentityResult {
        similarity,
        stage1,
        action: action_for_stage1(stage1),
    })
}

/// QC-1 multi-character stage. The Hungarian assignment result is retained in
/// full so the caller can pass missing/extra/twin-canon diagnostics to the
/// judge instead of collapsing them into a boolean.
pub fn qc1_multi_character_gate(
    detected: &[FaceEmbedding],
    canons: &[FaceEmbedding],
    thresholds: &[IdentityThresholds],
    config: AssignmentConfig,
) -> Result<AssignmentResult> {
    assign_faces_to_canons(detected, canons, thresholds, config)
}

/// QC-2 score aggregation. Every frame is compared to the same canon, so a
/// frame cannot silently switch encoder space mid-clip.
pub fn qc2_identity_gate(
    frames: &[TimedFaceEmbedding],
    canon: &FaceEmbedding,
    thresholds: &IdentityThresholds,
    config: &ClipAggregationConfig,
) -> Result<Qc2IdentityResult> {
    thresholds.validate_source(canon.source)?;
    let mut scores = Vec::with_capacity(frames.len());
    for frame in frames {
        let similarity = compare_embeddings(&frame.embedding, canon)?;
        scores.push(ClipFrameScore {
            timestamp_seconds: frame.timestamp_seconds,
            score: similarity,
            view_angle: frame.view_angle,
            marked_artifact: frame.marked_artifact,
        });
    }
    let aggregation = aggregate_clip_frames(&scores, thresholds, config)?;
    let action = match aggregation.verdict {
        crate::identity::scoring::ClipVerdict::Pass => QcAction::AutoPass,
        crate::identity::scoring::ClipVerdict::Borderline => QcAction::VisionJudge,
        crate::identity::scoring::ClipVerdict::Fail => QcAction::AutoFail,
    };
    Ok(Qc2IdentityResult { aggregation, action })
}

/// Convenience for consumers that already computed a validated similarity.
pub fn qc1_score_gate(score: CosineScore, thresholds: &IdentityThresholds) -> QcAction {
    action_for_stage1(crate::identity::scoring::stage1_score(score, thresholds))
}

/// Existing public API retained. Image decode + encoder inference + QC judge
/// adapters are intentionally not guessed: none of those interfaces/files is
/// included in the provided archive. The pure helpers above are ready to be
/// called by the complete server's provider/decoder orchestration.
pub async fn qc_keyframe(
    _frame: PathBuf,
    _references: &[PathBuf],
) -> Result<crate::providers::base::JudgeVerdict> {
    bail!("qc_keyframe requires the omitted image-decoder, embedding and vision-judge adapters; call qc1_identity_gate after inference")
}

/// Existing public API retained; extraction of 3 timestamped frames and the
/// provider judge are outside the source files supplied for this task.
pub async fn qc_shot(
    _clip: PathBuf,
    _expected_duration_s: f64,
) -> Result<crate::providers::base::JudgeVerdict> {
    bail!("qc_shot requires the omitted ffprobe/frame-extraction, embedding and vision-judge adapters; call qc2_identity_gate after extraction")
}

pub async fn qc_final(_export: PathBuf) -> Result<Vec<String>> {
    bail!("QC-3 media inspection adapters are outside the identity-core task/source set")
}
