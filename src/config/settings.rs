//! Runtime settings.
//!
//! One source of truth for the three runtime knobs: `config/settings.yaml`,
//! overridden by `CHA0S_MEDIA_*` environment variables. Provider keys are not
//! settings — they come from the process env (the MCP `context_servers.env`
//! block), are never written to disk, and are never logged.

use std::path::Path;

use anyhow::{Context, Result};
use figment::providers::{Env, Format, Serialized, Yaml};
use figment::Figment;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

impl Settings {
    /// `CHA0S_MEDIA_DEFAULT_TIER`, `CHA0S_MEDIA_PROJECT_BUDGET_USD`,
    /// `CHA0S_MEDIA_MAX_CONCURRENCY`.
    pub const ENV_PREFIX: &'static str = "CHA0S_MEDIA_";

    /// Defaults → optional YAML file → environment. A missing file is not an
    /// error: the defaults are a valid configuration.
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut figment = Figment::from(Serialized::defaults(Settings::default()));
        if let Some(path) = path
            && path.is_file()
        {
            figment = figment.merge(Yaml::file(path));
        }
        let settings: Self = figment
            .merge(Env::prefixed(Self::ENV_PREFIX))
            .extract()
            .with_context(|| {
                format!(
                    "load settings (file {}, env prefix {})",
                    path.map(|p| p.display().to_string())
                        .unwrap_or_else(|| "-".into()),
                    Self::ENV_PREFIX
                )
            })?;
        settings.validate()?;
        Ok(settings)
    }

    pub fn from_path(path: &Path) -> Result<Self> {
        Self::load(Some(path))
    }

    /// A budget of zero or a concurrency of zero would silently disable the
    /// spend gate or deadlock the render queue: refuse them at load time.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.project_budget_usd.is_finite() && self.project_budget_usd > 0.0,
            "project_budget_usd must be a positive number, got {}",
            self.project_budget_usd
        );
        anyhow::ensure!(
            self.max_concurrency >= 1,
            "max_concurrency must be >= 1, got {}",
            self.max_concurrency
        );
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Settings::default().validate().unwrap();
    }

    #[test]
    fn file_overrides_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.yaml");
        std::fs::write(
            &path,
            "default_tier: premium\nproject_budget_usd: 120.0\nmax_concurrency: 2\n",
        )
        .unwrap();
        let settings = Settings::from_path(&path).unwrap();
        assert_eq!(settings.default_tier, crate::config::models::Tier::Premium);
        assert_eq!(settings.project_budget_usd, 120.0);
        assert_eq!(settings.max_concurrency, 2);
    }

    #[test]
    fn missing_file_falls_back_to_defaults() {
        let settings = Settings::load(Some(Path::new("/nonexistent/settings.yaml"))).unwrap();
        assert_eq!(settings, Settings::default());
    }

    #[test]
    fn invalid_values_are_refused() {
        let zero_budget = Settings {
            project_budget_usd: 0.0,
            ..Default::default()
        };
        assert!(zero_budget.validate().is_err());
        let nan_budget = Settings {
            project_budget_usd: f64::NAN,
            ..Default::default()
        };
        assert!(nan_budget.validate().is_err());
        let zero_concurrency = Settings {
            max_concurrency: 0,
            ..Default::default()
        };
        assert!(zero_concurrency.validate().is_err());
    }
}
