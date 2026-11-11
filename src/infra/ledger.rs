//! Cost ledger and budget gates (ARCHITECTURE.md 8).
//!
//! Estimate before run, project cap, per-shot cap, stop + ask on overrun.
//! Retries ≤2 per shot, then tier escalation or operator flag.

pub struct Ledger {
    store: super::store::Store,
}

impl Ledger {
    pub fn new(store: super::store::Store) -> Self {
        Self { store }
    }

    /// Record a spend; refuses if the project cap would be exceeded.
    /// Data lands in the shared Postgres `ledger` table — the agent reads it
    /// through kb/database directly (ARCHITECTURE.md 9.5).
    pub async fn record(
        &self,
        _entry: crate::domain::cost::LedgerEntry,
        _project_cap_usd: f64,
    ) -> anyhow::Result<()> {
        // M2: INSERT INTO ledger; check SUM(actual_usd) + entry <= cap.
        unimplemented!("M2")
    }

    /// Total spent on a project.
    pub async fn spent(&self, _project_id: &str) -> anyhow::Result<f64> {
        // M2.
        unimplemented!("M2")
    }
}
