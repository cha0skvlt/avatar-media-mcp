//! Typed ffmpeg operations (ARCHITECTURE.md 9.4, 10).
//!
//! Receipts, not raw flags: each operation is a typed call producing a
//! deterministic command. ffmpeg/ffprobe are external binaries in PATH.

use std::process::Stdio;

/// Verify ffmpeg/ffprobe are available; called at server startup.
pub async fn check_available() -> anyhow::Result<()> {
    for bin in ["ffmpeg", "ffprobe"] {
        let ok = tokio::process::Command::new(bin)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
        anyhow::ensure!(ok, "{bin} not found in PATH");
    }
    Ok(())
}

/// Probe media: duration, resolution, fps.
pub async fn probe(_path: &std::path::Path) -> anyhow::Result<ProbeResult> {
    // M2: ffprobe -json.
    unimplemented!("M2")
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ProbeResult {
    pub duration_s: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

/// Typed operations (M2+): trim, concat, captions_ass, loudnorm,
/// sidechain_duck, color_grade_grain, normalize_fps.
pub enum FfmpegOp {
    Trim { input: std::path::PathBuf, start_s: f64, end_s: f64, output: std::path::PathBuf },
    Concat { inputs: Vec<std::path::PathBuf>, output: std::path::PathBuf },
    CaptionsAss { video: std::path::PathBuf, ass: std::path::PathBuf, output: std::path::PathBuf },
    Loudnorm { input: std::path::PathBuf, lufs: f64, output: std::path::PathBuf },
    SidechainDuck { voice: std::path::PathBuf, music: std::path::PathBuf, output: std::path::PathBuf },
    ColorGradeGrain { input: std::path::PathBuf, output: std::path::PathBuf },
    NormalizeFps { input: std::path::PathBuf, fps: u32, output: std::path::PathBuf },
}

impl FfmpegOp {
    /// Build the deterministic argv for this operation.
    pub fn argv(&self) -> Vec<String> {
        // M2: per-variant argv builders.
        match self {
            Self::Trim { .. } => vec!["ffmpeg".into()],
            _ => vec!["ffmpeg".into()],
        }
    }
}
