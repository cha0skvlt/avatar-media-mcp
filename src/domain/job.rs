//! Render job state (ARCHITECTURE.md 9.2).
//!
//! Long operations: render_start → job_id; job_status polls stages, percent,
//! spent, links. State in SQLite; restart marks stale jobs and resumes by
//! provider request_id.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Paused,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Job {
    pub id: String,
    pub project_id: String,
    pub state: JobState,
    /// Pipeline stage: audio / keyframes / shots / assemble.
    pub stage: String,
    pub percent: f64,
    pub spent_usd: f64,
    /// Provider request ids for resumability.
    #[serde(default)]
    pub provider_requests: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub error: Option<String>,
    pub started_at: String,
    #[serde(default)]
    pub updated_at: Option<String>,
}
