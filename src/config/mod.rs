//! Configuration: models.yaml registry, platforms.yaml profiles, .env keys.
//!
//! models.yaml: role → tier → {provider, model_id, params, price_per_unit}.
//! Provider keys are never logged and never written to the database.

pub mod models;
pub mod platforms;
pub mod settings;
