//! Registry: role + tier → provider instance from config/models.yaml.

use std::sync::Arc;

use crate::config::models::{ModelsConfig, Role, Tier};
use crate::config::settings::ProviderKeys;

/// Built provider instances keyed by provider name.
pub struct ProviderRegistry {
    pub models: ModelsConfig,
    pub keys: ProviderKeys,
    http: reqwest::Client,
}

impl ProviderRegistry {
    pub fn new(models: ModelsConfig, keys: ProviderKeys) -> anyhow::Result<Self> {
        Ok(Self {
            models,
            keys,
            http: reqwest::Client::builder()
                .user_agent("avatar-media-mcp/0.1")
                .build()?,
        })
    }

    /// Resolve the model entry for a role/tier (falls back to default tier).
    pub fn entry(&self, role: Role, tier: Tier) -> Option<&crate::config::models::ModelEntry> {
        self.models.resolve(role, tier)
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Construct a concrete provider by name. M1+: fal, elevenlabs, fish,
    /// byteplus constructors; unknown provider is a config error.
    pub fn provider(&self, name: &str) -> anyhow::Result<Arc<dyn std::any::Any>> {
        anyhow::bail!("provider `{name}` has no constructor yet (M1/M2)")
    }
}
