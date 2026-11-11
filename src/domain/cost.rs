//! Cost accounting.
//!
//! Ledger entry: provider, model, units (seconds/images/chars), fact vs estimate.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct LedgerEntry {
    pub project_id: String,
    pub shot_id: Option<String>,
    pub provider: String,
    pub model_id: String,
    pub role: String,
    pub tier: String,
    /// Seconds / images / 1K chars / calls, depending on role.
    pub units: f64,
    pub unit_name: String,
    pub estimated_usd: f64,
    pub actual_usd: f64,
    pub request_id: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EstimateTier {
    Draft,
    Standard,
    Premium,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Estimate {
    pub tier: EstimateTier,
    pub total_usd: f64,
    /// Breakdown by role: talk seconds, broll seconds, keyframes, tts chars.
    pub breakdown: Vec<EstimateLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct EstimateLine {
    pub role: String,
    pub units: f64,
    pub unit_name: String,
    pub usd: f64,
}
