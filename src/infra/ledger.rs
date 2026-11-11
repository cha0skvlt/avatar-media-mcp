//! Cost ledger and budget gates.
//!
//! Estimate before run, project cap, per-shot cap, stop + ask on overrun.
//! Retries ≤2 per shot, then tier escalation or operator flag.
//!
//! The budget promise is enforced here: `record` locks the project row, sums
//! the ledger inside the same transaction and refuses the insert when the cap
//! would be exceeded. Two concurrent records cannot both slip under the cap.

use anyhow::Context;

use crate::domain::cost::LedgerEntry;

pub struct Ledger {
    store: super::store::Store,
}

impl Ledger {
    pub fn new(store: super::store::Store) -> Self {
        Self { store }
    }

    /// Record a spend; refuses if the project cap would be exceeded.
    /// Data lands in the shared Postgres `ledger` table — the agent reads it
    /// through kb/database directly.
    pub async fn record(
        &self,
        entry: LedgerEntry,
        project_cap_usd: f64,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            entry.actual_usd.is_finite() && entry.actual_usd >= 0.0,
            "actual_usd must be a non-negative number, got {}",
            entry.actual_usd
        );
        anyhow::ensure!(
            project_cap_usd.is_finite() && project_cap_usd > 0.0,
            "project cap must be a positive number, got {project_cap_usd}"
        );

        let mut tx = self.store.pool().begin().await?;
        // Lock the project row: the sum below is only meaningful if no other
        // writer can insert between the check and the insert.
        let project: Option<(String,)> =
            sqlx::query_as("SELECT id FROM projects WHERE id = $1 FOR UPDATE")
                .bind(&entry.project_id)
                .fetch_optional(&mut *tx)
                .await?;
        anyhow::ensure!(
            project.is_some(),
            "project {} does not exist",
            entry.project_id
        );

        let spent: (Option<f64>,) =
            sqlx::query_as("SELECT SUM(actual_usd) FROM ledger WHERE project_id = $1")
                .bind(&entry.project_id)
                .fetch_one(&mut *tx)
                .await?;
        let spent = spent.0.unwrap_or(0.0);
        let projected = spent + entry.actual_usd;
        anyhow::ensure!(
            projected <= project_cap_usd + 1.0e-9,
            "budget cap exceeded: spent {spent:.4} + {:.4} = {projected:.4} > cap {project_cap_usd:.4}",
            entry.actual_usd
        );

        sqlx::query(
            "INSERT INTO ledger (project_id, shot_id, provider, model_id, role, tier, units, \
             unit_name, estimated_usd, actual_usd, request_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
        )
        .bind(&entry.project_id)
        .bind(&entry.shot_id)
        .bind(&entry.provider)
        .bind(&entry.model_id)
        .bind(&entry.role)
        .bind(&entry.tier)
        .bind(entry.units)
        .bind(&entry.unit_name)
        .bind(entry.estimated_usd)
        .bind(entry.actual_usd)
        .bind(&entry.request_id)
        .execute(&mut *tx)
        .await
        .context("insert ledger entry")?;

        tx.commit().await?;
        Ok(())
    }

    /// Total spent on a project.
    pub async fn spent(&self, project_id: &str) -> anyhow::Result<f64> {
        self.store.spent_usd(project_id).await
    }
}
