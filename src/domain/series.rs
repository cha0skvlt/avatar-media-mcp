//! Series layer for format=film.
//!
//! series.json over projects: shared cast, style rules, series-wide budget.
//! Each episode is a normal project referencing the series.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Series {
    pub id: String,
    pub title: String,
    /// Shared cast across episodes.
    pub cast: Vec<String>,
    /// Style rules applied to every episode (color, grain, framing language).
    #[serde(default)]
    pub style_rules: Vec<String>,
    /// Series-wide budget cap spanning all episodes.
    pub budget_cap_usd: f64,
    #[serde(default)]
    pub episodes: Vec<String>,
    pub created_at: String,
}
