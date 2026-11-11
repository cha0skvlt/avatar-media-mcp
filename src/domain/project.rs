//! Project brief and state machine (ARCHITECTURE.md 5.1, 5.3).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectFormat {
    Photo,
    PortraitSet,
    VideoShort,
    VideoLong,
    Film,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProjectState {
    Draft,
    ScriptSubmitted,
    Estimated,
    Approved,
    Rendering,
    Assembled,
    Done,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Brief {
    /// Avatar slugs; one or more (multi-character is the normal mode).
    pub cast: Vec<String>,
    pub format: ProjectFormat,
    pub platform: String,
    /// Default "en"; other languages are explicit (operator decision #2).
    pub language: String,
    pub goal: String,
    pub budget_cap_usd: f64,
    /// Series reference for format=film.
    #[serde(default)]
    pub series_id: Option<String>,
    #[serde(default)]
    pub materials: Vec<String>,
}
