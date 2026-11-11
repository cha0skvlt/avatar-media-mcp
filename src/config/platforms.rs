//! Platform profiles loaded from config/platforms.yaml (ARCHITECTURE.md 10).
//!
//! youtube_shorts / youtube_long / vertical_generic. Export rules:
//! ai_disclosure default true, auto_publish false — always.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformsConfig {
    pub platforms: std::collections::BTreeMap<String, PlatformProfile>,
    pub export: ExportRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformProfile {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub max_duration_s: Option<u64>,
    pub loudness_lufs: f64,
    pub subtitles: SubtitleMode,
    #[serde(default)]
    pub subtitle_safe_area: Option<SafeArea>,
    #[serde(default)]
    pub subtitle_font_size: Option<String>,
    #[serde(default)]
    pub chapters_from_script: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubtitleMode {
    WordLevelAss,
    OptionalBottom,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SafeArea {
    pub top: f64,
    pub bottom: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportRules {
    pub ai_disclosure_default: bool,
    pub auto_publish: bool,
}
