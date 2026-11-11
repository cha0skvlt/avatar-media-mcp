//! Series serialization for format=film (M7).
//!
//! series.json over projects: shared cast, style rules, series-wide budget.
//! Each episode is a normal project referencing the series.

/// Create a series; subsequent project_create with series_id links episodes.
pub fn create_series(_series: crate::domain::series::Series) -> anyhow::Result<String> {
    // M7.
    anyhow::bail!("series.create lands with M7 (film serialization)")
}

/// Series status: episodes, per-episode and total spend, continuity state.
pub fn status(_series_id: &str) -> anyhow::Result<serde_json::Value> {
    // M7.
    anyhow::bail!("series.status lands with M7 (film serialization)")
}
