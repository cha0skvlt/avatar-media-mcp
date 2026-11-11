//! Model registry types loaded from config/models.yaml.
//!
//! role → tier → {provider, model_id, params, price_per_unit}.
//! Tiers: draft / standard / premium.
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

/// Registry defaults. Budget and concurrency are runtime settings, not model
/// registry data — they live in `config/settings.yaml`, one source only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Defaults {
    pub tier: Tier,
}

impl ModelsConfig {
    /// Parse `config/models.yaml`. The file is the registry the server prices
    /// against; nothing else in the crate reads it, so this is the only entry.
    pub fn from_yaml_str(raw: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(raw)
    }

    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| anyhow::anyhow!("read {}: {error}", path.display()))?;
        Self::from_yaml_str(&raw)
            .map_err(|error| anyhow::anyhow!("parse {}: {error}", path.display()))
    }

    /// Resolve role+tier → model entry; falls back to `defaults.tier` when the
    /// role has no entry for the requested tier (e.g. `avatar_canon` has no
    /// `draft` model). Returns `None` only when neither key exists.
    pub fn resolve(&self, role: Role, tier: Tier) -> Option<&ModelEntry> {
        let role_key = serde_json::to_value(role).ok()?;
        let role_key = role_key.as_str()?;
        let tiers = self.roles.get(role_key)?;
        let tier_key = serde_json::to_value(tier).ok()?;
        let tier_key = tier_key.as_str()?;
        if let Some(entry) = tiers.get(tier_key) {
            return Some(entry);
        }
        let default_key = serde_json::to_value(self.defaults.tier).ok()?;
        tiers.get(default_key.as_str()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const YAML: &str = r#"
 defaults:
   tier: standard
 roles:
   avatar_canon:
     standard:
       provider: fal
       model_id: model-standard
       price_per_image: 0.03
   talk:
     draft:
       provider: fal
       model_id: model-draft
     standard:
       provider: fal
       model_id: model-standard-talk
 m0_results: {}
 "#;

    #[test]
    fn resolve_falls_back_to_default_tier() {
        let config = ModelsConfig::from_yaml_str(YAML).unwrap();
        // `avatar_canon` has no draft entry: the documented fallback applies.
        let entry = config.resolve(Role::AvatarCanon, Tier::Draft).unwrap();
        assert_eq!(entry.model_id, "model-standard");
        // An explicit entry wins over the fallback.
        let draft = config.resolve(Role::Talk, Tier::Draft).unwrap();
        assert_eq!(draft.model_id, "model-draft");
        // Unknown role is still None.
        assert!(config.resolve(Role::QcJudge, Tier::Standard).is_none());
    }

    #[test]
    fn price_spec_flattens_into_the_entry() {
        let config = ModelsConfig::from_yaml_str(YAML).unwrap();
        let entry = config.resolve(Role::AvatarCanon, Tier::Standard).unwrap();
        assert_eq!(entry.price.price_per_image, Some(0.03));
        assert_eq!(entry.price.price_per_second, None);
    }
}
