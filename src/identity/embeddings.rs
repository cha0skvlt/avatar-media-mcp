//! Embedding-space contracts and cosine metrics (ARCHITECTURE.md 4.5).
//!
//! An embedding is meaningful only inside the space produced by its encoder.
//! ArcFace/ONNX and the fal vision encoder therefore carry distinct enum tags;
//! comparing them or mixing them during calibration is an error, never an
//! implicit conversion. This deliberately chooses separate calibrated spaces
//! over an unverified Procrustes transform: a transform needs paired, trusted
//! samples from the *same* faces and must itself be versioned and validated.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Identity of the model/space that emitted an embedding.
/// Add a new variant whenever a genuinely different encoder is introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingSource {
    /// Local ArcFace-compatible ONNX encoder.
    ArcFaceOnnx,
    /// Vision embedding returned by the fal provider.
    VisionFal,
    /// Compatibility tag for callers of the old untyped vector API only.
    /// Typed QC and calibration APIs reject mixing this with named sources.
    LegacyUnspecified,
}

/// A validated cosine score. The value is bounded by [-1, 1].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CosineScore(pub f32);

impl CosineScore {
    pub fn new(value: f32) -> Result<Self> {
        if !value.is_finite() || !(-1.000_01..=1.000_01).contains(&value) {
            bail!("cosine score must be finite and in [-1, 1], got {value}");
        }
        Ok(Self(value.clamp(-1.0, 1.0)))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

/// Embedding plus its explicit encoder-space provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceEmbedding {
    pub vector: Vec<f32>,
    pub source: EmbeddingSource,
}

impl FaceEmbedding {
    pub fn new(vector: Vec<f32>, source: EmbeddingSource) -> Result<Self> {
        validate_vector(&vector)?;
        Ok(Self { vector, source })
    }

    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

/// Validate a vector before it enters any metric/calibration calculation.
pub fn validate_vector(vector: &[f32]) -> Result<()> {
    if vector.is_empty() {
        bail!("embedding vector must not be empty");
    }
    if vector.iter().any(|value| !value.is_finite()) {
        bail!("embedding contains NaN or infinity");
    }
    let norm_squared: f64 = vector.iter().map(|value| f64::from(*value).powi(2)).sum();
    if norm_squared <= f64::EPSILON {
        bail!("embedding vector has zero norm");
    }
    Ok(())
}

/// Checked cosine metric used by all new identity-core code.
pub fn cosine_similarity_checked(a: &[f32], b: &[f32]) -> Result<CosineScore> {
    validate_vector(a)?;
    validate_vector(b)?;
    if a.len() != b.len() {
        bail!("embedding dimensions differ: {} vs {}", a.len(), b.len());
    }

    // Accumulate in f64: 512-dimensional f32 vectors otherwise lose avoidable
    // precision, especially around close decision boundaries.
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (left, right) in a.iter().zip(b) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    let cosine = dot / (norm_a.sqrt() * norm_b.sqrt());
    CosineScore::new(cosine as f32)
}

/// Compare two typed embeddings. Different encoders are intentionally rejected.
pub fn compare_embeddings(a: &FaceEmbedding, b: &FaceEmbedding) -> Result<CosineScore> {
    if a.source != b.source {
        bail!(
            "incompatible embedding sources: {:?} vs {:?}; calibrate or align spaces explicitly",
            a.source,
            b.source
        );
    }
    cosine_similarity_checked(&a.vector, &b.vector)
}

/// Backwards-compatible cosine API used by the original integration test and
/// older call sites. New code must use `cosine_similarity_checked` so invalid
/// shapes/vectors cannot silently become valid-looking scores.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    cosine_similarity_checked(a, b).map(CosineScore::get).unwrap_or(0.0)
}

/// Legacy inference boundary retained for API compatibility. The source
/// archive does not include either the ONNX runtime adapter or fal provider
/// implementation; fail explicitly instead of panicking with `unimplemented!`.
pub async fn compute_embeddings(
    _images: &[PathBuf],
) -> Result<BTreeMap<String, Vec<f32>>> {
    bail!(
        "embedding inference adapter is not part of the supplied source set; provide the licensed local ONNX or provider adapter"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_cosine_rejects_invalid_vectors() {
        assert!(cosine_similarity_checked(&[1.0, 0.0], &[1.0]).is_err());
        assert!(cosine_similarity_checked(&[0.0, 0.0], &[1.0, 0.0]).is_err());
        assert!(cosine_similarity_checked(&[f32::NAN], &[1.0]).is_err());
    }

    #[test]
    fn source_mismatch_is_an_error() {
        let arc = FaceEmbedding::new(vec![1.0, 0.0], EmbeddingSource::ArcFaceOnnx).unwrap();
        let vision = FaceEmbedding::new(vec![1.0, 0.0], EmbeddingSource::VisionFal).unwrap();
        assert!(compare_embeddings(&arc, &vision).is_err());
    }
}
