//! Postgres state store.
//!
//! One contour with the agent: the same database the agent reaches through
//! `kb`/`database` in CHA0S AOE. DSN comes from `DATABASE_URL`; the server does
//! not connect at startup, so faceswap/avatar_create work without a database.
//! The first stateful tool (M2) makes the DSN mandatory.
//!
//! Tables: projects, shots, jobs, ledger, identity_scores, m0_results.
//! Provider keys are never written here. Artifacts stay on disk; the DB holds
//! paths and metadata only.

use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub struct Store {
    pool: PgPool,
}

impl Store {
    /// Connect and run migrations.
    pub async fn connect(database_url: &str) -> anyhow::Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        sqlx::migrate!("./migrations").run(&pool).await?;
        Ok(Self { pool })
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Liveness probe for `providers_check`.
    pub async fn health(&self) -> anyhow::Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    /// Total actual spend on a project.
    pub async fn spent_usd(&self, project_id: &str) -> anyhow::Result<f64> {
        let row: (Option<f64>,) =
            sqlx::query_as("SELECT SUM(actual_usd) FROM ledger WHERE project_id = $1")
                .bind(project_id)
                .fetch_one(&self.pool)
                .await?;
        Ok(row.0.unwrap_or(0.0))
    }

}
