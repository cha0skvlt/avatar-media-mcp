//! Configuration: models.yaml registry, platforms.yaml profiles, .env keys.
//!
//! models.yaml: role → tier → {provider, model_id, params, price_per_unit}
//! (ARCHITECTURE.md 6.1–6.3). providers_check verifies liveness daily.
//! .env is never logged and never written to state.sqlite (6.3).

pub mod models;
pub mod platforms;
pub mod settings;
