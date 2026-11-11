//! Postgres state store (ARCHITECTURE.md 9.5).
//!
//! One contour with the agent: the same database the agent reaches through
//! `kb`/`database` in CHA0S AOE. DSN comes from DATABASE_URL env; the server
//! refuses to start without it — state must not live in two places.
//!
//! Tables: projects, shots, jobs, ledger, identity_scores, m0_results.
//! Provider keys are never written here. Artifacts stay on disk; the DB holds
//! paths and metadata only.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

pub struct Store {
    pool: PgPool,
}

impl Store {
    /// Connect and run migrations. DATABASE_URL is mandatory.
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
}
