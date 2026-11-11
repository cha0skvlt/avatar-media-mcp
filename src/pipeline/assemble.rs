//! Assembly stage.
//!
//! Deterministic ffmpeg: concat, J/L cuts, word-level ASS subtitles in
//! safe-area, unified color grade + grain, music with sidechain ducking,
//! loudnorm to platform LUFS, platform profile (shorts/long).

/// Assemble the final export for a platform profile.
pub async fn assemble(
    _project_id: &str,
    _platform: &str,
) -> anyhow::Result<std::path::PathBuf> {
    // M2 (animatic) / M6 (full).
    anyhow::bail!("assemble lands with M2 (ffmpeg argv builders + concat)")
}

/// Export package: exports/<platform>.mp4 + metadata.json
/// (title, description, tags, ai_disclosure: true by default).
pub async fn export_package(
    _project_id: &str,
    _platform: &str,
) -> anyhow::Result<std::path::PathBuf> {
    // M6.
    anyhow::bail!("export_package lands with M6 (metadata.json + ai_disclosure)")
}
