//! Provider traits.
//!
//! Every trait is mockable in tests; implementations are thin reqwest clients.

use async_trait::async_trait;

/// Image generation with character references (keyframes, canon sheets).
#[async_trait]
pub trait ImageProvider: Send + Sync {
    async fn generate_image(
        &self,
        prompt: &str,
        negative: Option<&str>,
        reference_images: &[std::path::PathBuf],
        width: u32,
        height: u32,
        seed: Option<u64>,
    ) -> anyhow::Result<Vec<u8>>;
}

/// Video generation without avatar (B-roll).
#[async_trait]
pub trait VideoProvider: Send + Sync {
    async fn generate_video(
        &self,
        prompt: &str,
        first_frame: Option<std::path::PathBuf>,
        duration_s: f64,
        with_audio: bool,
    ) -> anyhow::Result<Vec<u8>>;
}

/// Photo + audio segment → talking video (TALK shots).
#[async_trait]
pub trait AvatarVideoProvider: Send + Sync {
    async fn generate_talk(
        &self,
        frame: std::path::PathBuf,
        audio: std::path::PathBuf,
        motion_prompt: Option<&str>,
    ) -> anyhow::Result<Vec<u8>>;
}

/// Voice cloning and speech synthesis with word timings.
#[async_trait]
pub trait TtsProvider: Send + Sync {
    async fn synthesize(
        &self,
        voice_id: &str,
        text: &str,
        language: &str,
    ) -> anyhow::Result<TtsResult>;

    async fn clone_voice(&self, sample: std::path::PathBuf, name: &str) -> anyhow::Result<String>;
}

#[derive(Debug, Clone)]
pub struct TtsResult {
    pub audio: Vec<u8>,
    /// Word-level timings for karaoke subtitles and J/L cuts.
    pub word_timings: Vec<WordTiming>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WordTiming {
    pub word: String,
    pub start_s: f64,
    pub end_s: f64,
}

/// Vision judge for borderline identity scores and artifacts (QC-1/QC-2).
#[async_trait]
pub trait JudgeProvider: Send + Sync {
    async fn judge_identity(
        &self,
        candidate: std::path::PathBuf,
        references: &[std::path::PathBuf],
        checklist: &[&str],
    ) -> anyhow::Result<JudgeVerdict>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JudgeVerdict {
    pub pass: bool,
    pub score: f32,
    pub findings: Vec<String>,
}
