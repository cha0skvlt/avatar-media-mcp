//! Shot render stage.
//!
//! TALK → avatar model (frame + audio segment); BROLL → image-to-video /
//! reference-to-video; SCREEN/TITLE → local (ffmpeg drawtext), $0.
//! Parallel with a concurrency ceiling; QC-2 per shot.

/// Render all shots; jobs tracked in Postgres, resumable by request_id.
pub async fn render_shots(
    _project_id: &str,
    _tier: crate::config::models::Tier,
) -> anyhow::Result<String> {
    // M3/M4: returns job_id.
    anyhow::bail!("render_shots lands with M3/M4 (avatar + video providers)")
}
