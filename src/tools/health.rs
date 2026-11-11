//! Preflight (`providers_check`): what is configured, what is missing.
//!
//! No network calls, no model loads — this answers "why did the last call
//! fail" before the operator spends money or waits for ONNX.

use std::path::Path;

use serde::Serialize;

use crate::config::models::ModelsConfig;
use crate::config::platforms::PlatformsConfig;
use crate::config::settings::{ProviderKeys, Settings};
use crate::infra::avatar_store::AvatarStore;

#[derive(Debug, Serialize)]
pub struct PreflightReport {
    pub models: ConfigStatus,
    pub platforms: ConfigStatus,
    pub settings: ConfigStatus,
    pub keys: Vec<KeyStatus>,
    pub runtime: RuntimeStatus,
    pub avatars: AvatarStatus,
    pub database: DatabaseStatus,
}

#[derive(Debug, Serialize)]
pub struct ConfigStatus {
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Serialize)]
pub struct KeyStatus {
    pub name: String,
    pub present: bool,
}

#[derive(Debug, Serialize)]
pub struct RuntimeStatus {
    pub ort_dylib_env: bool,
    pub insightface_models: bool,
    pub faceswap_models: bool,
    pub ffmpeg: bool,
    pub ffmpeg_detail: String,
}

#[derive(Debug, Serialize)]
pub struct AvatarStatus {
    pub root: String,
    pub writable: bool,
    pub avatars: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DatabaseStatus {
    pub dsn_configured: bool,
    pub detail: String,
}

pub async fn preflight() -> PreflightReport {
    let models = match ModelsConfig::from_path(Path::new("config/models.yaml")) {
        Ok(config) => ConfigStatus {
            ok: true,
            detail: format!("{} roles, default tier {:?}", config.roles.len(), config.defaults.tier),
        },
        Err(error) => ConfigStatus {
            ok: false,
            detail: error.to_string(),
        },
    };

    let platforms = match PlatformsConfig::from_path(Path::new("config/platforms.yaml")) {
        Ok(config) => ConfigStatus {
            ok: true,
            detail: format!(
                "{} profiles: {}",
                config.platforms.len(),
                config.platforms.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        },
        Err(error) => ConfigStatus {
            ok: false,
            detail: error.to_string(),
        },
    };

    let settings = match Settings::load(Some(Path::new("config/settings.yaml"))) {
        Ok(settings) => ConfigStatus {
            ok: true,
            detail: format!(
                "tier {:?}, budget ${:.2}, concurrency {}",
                settings.default_tier, settings.project_budget_usd, settings.max_concurrency
            ),
        },
        Err(error) => ConfigStatus {
            ok: false,
            detail: error.to_string(),
        },
    };

    let keys = ProviderKeys::from_env();
    let keys = vec![
        KeyStatus { name: "FAL_KEY".into(), present: keys.fal_key.is_some() },
        KeyStatus { name: "ELEVENLABS_API_KEY".into(), present: keys.elevenlabs_api_key.is_some() },
        KeyStatus { name: "FISH_API_KEY".into(), present: keys.fish_api_key.is_some() },
        KeyStatus { name: "HEDRA_API_KEY".into(), present: keys.hedra_api_key.is_some() },
        KeyStatus { name: "BYTEPLUS_API_KEY".into(), present: keys.byteplus_api_key.is_some() },
        KeyStatus { name: "ANTHROPIC_API_KEY".into(), present: keys.anthropic_api_key.is_some() },
        KeyStatus { name: "KLING_ACCESS_KEY".into(), present: keys.kling_access_key.is_some() },
        KeyStatus { name: "KLING_SECRET_KEY".into(), present: keys.kling_secret_key.is_some() },
    ];

    let ffmpeg = crate::infra::ffmpeg::check_available().await;
    let runtime = RuntimeStatus {
        ort_dylib_env: std::env::var("ORT_DYLIB_PATH").is_ok_and(|v| !v.is_empty()),
        insightface_models: Path::new("models/insightface/buffalo_l").is_dir(),
        faceswap_models: Path::new("models/faceswap").is_dir(),
        ffmpeg: ffmpeg.is_ok(),
        ffmpeg_detail: match ffmpeg {
            Ok(()) => "ffmpeg + ffprobe in PATH".into(),
            Err(error) => error.to_string(),
        },
    };

    let store = AvatarStore::default();
    let root = store.root().to_path_buf();
    let writable = std::fs::create_dir_all(&root).is_ok();
    let avatars = store.list().unwrap_or_default();
    let avatars = AvatarStatus {
        root: root.display().to_string(),
        writable,
        avatars,
    };

    let database = match std::env::var("DATABASE_URL") {
        Ok(dsn) if !dsn.is_empty() => DatabaseStatus {
            dsn_configured: true,
            detail: "DATABASE_URL is set; the server connects lazily (no tool needs it yet)".into(),
        },
        _ => DatabaseStatus {
            dsn_configured: false,
            detail: "DATABASE_URL is not set: stateful tools (M2) will refuse to run".into(),
        },
    };

    PreflightReport {
        models,
        platforms,
        settings,
        keys,
        runtime,
        avatars,
        database,
    }
}
