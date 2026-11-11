//! Model registry types loaded from config/models.yaml.
//!
//! role → tier → {provider, model_id, params, price_per_unit}.
//! Tiers: draft / standard / premium (ARCHITECTURE.md 6).
//! M0 fills m0_results: watermark_check, moderation_threshold, blind_test_winner.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    AvatarCanon,
    Keyframe,
    Talk,
    Broll,
    BrollReference,
    Tts,
    WordTiming,
    QcJudge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    Draft,
    Standard,
    Premium,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub provider: String,
    pub model_id: String,
    #[serde(default)]
    pub params: serde_json::Value,
    /// Price unit depends on role: per image / per second / per 1K chars / per call.
    #[serde(default, flatten)]
    pub price: PriceSpec,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PriceSpec {
    pub price_per_image: Option<f64>,
    pub price_per_second: Option<f64>,
    pub price_per_1k_chars: Option<f64>,
    pub price_per_call: Option<f64>,
    pub price_per_unit: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    pub defaults: Defaults,
    pub roles: std::collections::BTreeMap<String, std::collections::BTreeMap<String, ModelEntry>>,
    #[serde(default)]
    pub m0_results: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Defaults {
    pub tier: Tier,
    pub budget_cap_usd: f64,
    pub max_concurrency: u32,
}

impl ModelsConfig {
    /// Resolve role+tier → model entry; falls back to defaults.tier.
    pub fn resolve(&self, role: Role, tier: Tier) -> Option<&ModelEntry> {
        let role_key = serde_json::to_value(role).ok()?;
        let role_key = role_key.as_str()?;
        let tier_key = serde_json::to_value(tier).ok()?;
        let tier_key = tier_key.as_str()?;
        self.roles.get(role_key)?.get(tier_key)
    }
}
