//! Runtime settings from .env / process env (ARCHITECTURE.md 6.3).
//!
//! Minimum for MVP: FAL_KEY, ELEVENLABS_API_KEY. The server checks .env file
//! permissions at startup and never writes keys to logs or state.sqlite.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub default_tier: crate::config::models::Tier,
    pub project_budget_usd: f64,
    pub max_concurrency: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            default_tier: crate::config::models::Tier::Standard,
            project_budget_usd: 25.0,
            max_concurrency: 4,
        }
    }
}

/// Provider API keys, resolved from env at startup. Never serialized to logs.
#[derive(Debug, Default, Clone)]
pub struct ProviderKeys {
    pub fal_key: Option<String>,
    pub elevenlabs_api_key: Option<String>,
    pub fish_api_key: Option<String>,
    pub hedra_api_key: Option<String>,
    pub byteplus_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub kling_access_key: Option<String>,
    pub kling_secret_key: Option<String>,
}

impl ProviderKeys {
    pub fn from_env() -> Self {
        let get = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        Self {
            fal_key: get("FAL_KEY"),
            elevenlabs_api_key: get("ELEVENLABS_API_KEY"),
            fish_api_key: get("FISH_API_KEY"),
            hedra_api_key: get("HEDRA_API_KEY"),
            byteplus_api_key: get("BYTEPLUS_API_KEY"),
            anthropic_api_key: get("ANTHROPIC_API_KEY"),
            kling_access_key: get("KLING_ACCESS_KEY"),
            kling_secret_key: get("KLING_SECRET_KEY"),
        }
    }
}
