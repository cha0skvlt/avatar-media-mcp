//! Identity module: the mathematical identity lock (ARCHITECTURE.md 4.5).
//!
//! Three models computed once at avatar creation:
//! 1. ArcFace 512-d embeddings — cosine similarity = "same person" metric.
//! 2. FLAME 3DMM fit — face geometry passport (works where ArcFace is weak).
//! 3. SMPL/BLADE body fit — figure check for wide shots.
//!
//! Wired into the pipeline: reference ranking, two-stage QC (local score first,
//! vision judge only for borderline), drift monitoring in the ledger,
//! multi-character "each closer to own canon" check.
//!
//! Local inference is behind the `local-identity` feature (ort/ONNX). Without
//! it, embeddings go through a vision model via fal (fallback, 4.5 licenses).

pub mod embeddings;
pub mod scoring;
pub mod threed;
#[cfg(feature = "local-identity")]
pub mod arcface;
