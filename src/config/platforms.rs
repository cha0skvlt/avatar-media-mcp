//! Platform profiles loaded from config/platforms.yaml.
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

impl PlatformsConfig {
    /// Parse `config/platforms.yaml`.
    pub fn from_yaml_str(raw: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(raw)
    }

    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| anyhow::anyhow!("read {}: {error}", path.display()))?;
        Self::from_yaml_str(&raw)
            .map_err(|error| anyhow::anyhow!("parse {}: {error}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_rules_are_conservative() {
        let config = PlatformsConfig::from_path(std::path::Path::new("config/platforms.yaml")).unwrap();
        assert!(config.export.ai_disclosure_default, "AI disclosure must default on");
        assert!(!config.export.auto_publish, "the server never publishes");
        let shorts = config.platforms.get("youtube_shorts").unwrap();
        assert_eq!((shorts.width, shorts.height), (1080, 1920));
        assert_eq!(shorts.subtitles, SubtitleMode::WordLevelAss);
    }
}
