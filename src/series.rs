//! Series serialization for format=film (ARCHITECTURE.md 5.1, M7).
//!
//! series.json over projects: shared cast, style rules, series-wide budget.
//! Each episode is a normal project referencing the series.

/// Create a series; subsequent project_create with series_id links episodes.
pub fn create_series(_series: crate::domain::series::Series) -> anyhow::Result<String> {
    // M7.
    unimplemented!("M7")
}

/// Series status: episodes, per-episode and total spend, continuity state.
pub fn status(_series_id: &str) -> anyhow::Result<serde_json::Value> {
    // M7.
    unimplemented!("M7")
}
