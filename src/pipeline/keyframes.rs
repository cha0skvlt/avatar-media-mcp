//! Keyframe stage.
//!
//! Image model + avatar references → one frame per shot, generated in the
//! target aspect ratio (never cropped later). QC-1 gates identity.

/// Generate keyframes for all shots; QC-1 with retries ≤2.
pub async fn generate_keyframes(
    _script: &crate::domain::script::Script,
    _platform: &crate::config::platforms::PlatformProfile,
    _tier: crate::config::models::Tier,
) -> anyhow::Result<Vec<std::path::PathBuf>> {
    // M3.
    anyhow::bail!("generate_keyframes lands with M3 (image provider + QC-1)")
}
