//! Avatar build pipeline (ARCHITECTURE.md 4.3).
//!
//! Steps: normalize photos → quality check → dedupe → identity model
//! (embeddings + 3D fits) → canon sheet generation with ArcFace scoring →
//! voice clone.

/// Normalize a photo: sRGB, ≤2048px, EXIF stripped.
pub fn normalize_photo(_input: std::path::PathBuf, _output: std::path::PathBuf) -> anyhow::Result<()> {
    // M1: image crate — resize, strip metadata, convert to sRGB.
    unimplemented!("M1")
}

/// Quality check: resolution, sharpness, single face, angle variety.
/// Returns concrete remarks ("need 3/4 angle", "too dark").
pub fn check_photo_quality(_photos: &[std::path::PathBuf]) -> anyhow::Result<Vec<String>> {
    // M1.
    unimplemented!("M1")
}

/// Generate the canon sheet (~6–8 images, ~$1); each sheet is ArcFace-scored
/// against references before operator approval.
pub async fn build_canon(
    _slug: &str,
    _tier: crate::config::models::Tier,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    // M1.
    unimplemented!("M1")
}
