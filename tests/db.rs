//! Postgres contour tests. Skipped unless `DATABASE_URL` is set, so the suite
//! stays green on a machine without a database.
//!
//! Run: `DATABASE_URL=postgres://… cargo test --features local-identity --test db`

use avatar_media_mcp::domain::cost::LedgerEntry;
use avatar_media_mcp::infra::ledger::Ledger;
use avatar_media_mcp::infra::store::Store;

fn entry(project_id: &str, usd: f64, request_id: Option<&str>) -> LedgerEntry {
    LedgerEntry {
        project_id: project_id.into(),
        shot_id: None,
        provider: "fal".into(),
        model_id: "test-model".into(),
        role: "talk".into(),
        tier: "standard".into(),
        units: 1.0,
        unit_name: "call".into(),
        estimated_usd: usd,
        actual_usd: usd,
        request_id: request_id.map(str::to_string),
        created_at: "2026-11-12T00:00:00Z".into(),
    }
}

#[tokio::test]
async fn ledger_enforces_the_project_cap_and_is_idempotent_per_request() {
    let Ok(dsn) = std::env::var("DATABASE_URL") else {
        eprintln!("skipped: DATABASE_URL is not set");
        return;
    };
    let store = Store::connect(&dsn).await.expect("connect");
    store.health().await.expect("health");

    let project_id = format!("qa-ledger-{}", std::process::id());
    sqlx::query(
        "INSERT INTO projects (id, brief, budget_cap_usd) VALUES ($1, '{}'::jsonb, 1.0) \
         ON CONFLICT (id) DO UPDATE SET budget_cap_usd = 1.0",
    )
    .bind(&project_id)
    .execute(store.pool())
    .await
    .expect("insert project");

    let ledger = Ledger::new(store);
    ledger
        .record(entry(&project_id, 0.40, Some("req-1")), 1.0)
        .await
        .expect("first spend fits");
    assert!((ledger.spent(&project_id).await.unwrap() - 0.40).abs() < 1e-9);

    // A retried provider request must not be charged twice.
    assert!(
        ledger
            .record(entry(&project_id, 0.40, Some("req-1")), 1.0)
            .await
            .is_err(),
        "duplicate request_id must be rejected by the unique index"
    );

    // 0.40 + 0.70 > 1.00: the cap is enforced inside the transaction.
    let error = ledger
        .record(entry(&project_id, 0.70, Some("req-2")), 1.0)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("budget cap exceeded"), "{error}");
    assert!((ledger.spent(&project_id).await.unwrap() - 0.40).abs() < 1e-9);

    // Unknown project is refused before any insert.
    assert!(
        ledger
            .record(entry("qa-missing-project", 0.01, None), 1.0)
            .await
            .is_err()
    );

    // Cleanup: the cascade added in 0002 removes the ledger rows.
    let cleanup = Store::connect(&dsn).await.expect("reconnect");
    sqlx::query("DELETE FROM projects WHERE id = $1")
        .bind(&project_id)
        .execute(cleanup.pool())
        .await
        .expect("cleanup");
    let orphan: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM ledger WHERE project_id = $1")
        .bind(&project_id)
        .fetch_one(cleanup.pool())
        .await
        .expect("count");
    assert_eq!(orphan.0, 0, "ON DELETE CASCADE must remove ledger rows");
}
