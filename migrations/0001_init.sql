-- avatar-media-mcp schema. One contour with the agent:
-- the agent reads these tables through kb/database directly.

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    brief JSONB NOT NULL,
    state TEXT NOT NULL DEFAULT 'draft',
    script JSONB,
    script_hash TEXT,
    approved_at TIMESTAMPTZ,
    budget_cap_usd DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shots (
    id TEXT NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(id),
    script_hash TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    tier TEXT NOT NULL,
    retries INTEGER NOT NULL DEFAULT 0,
    identity_score DOUBLE PRECISION,
    output_path TEXT,
    PRIMARY KEY (project_id, id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    state TEXT NOT NULL DEFAULT 'queued',
    stage TEXT NOT NULL,
    percent DOUBLE PRECISION NOT NULL DEFAULT 0,
    spent_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    provider_requests JSONB NOT NULL DEFAULT '{}',
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ledger (
    seq BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    shot_id TEXT,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    role TEXT NOT NULL,
    tier TEXT NOT NULL,
    units DOUBLE PRECISION NOT NULL,
    unit_name TEXT NOT NULL,
    estimated_usd DOUBLE PRECISION NOT NULL,
    actual_usd DOUBLE PRECISION NOT NULL,
    request_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ledger_project_idx ON ledger(project_id);

-- Identity scores (pgvector): drift monitoring and multi-character checks.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS identity_scores (
    seq BIGSERIAL PRIMARY KEY,
    project_id TEXT NOT NULL,
    shot_id TEXT,
    subject_slug TEXT NOT NULL,
    embedding vector(512) NOT NULL,
    cosine_to_canon DOUBLE PRECISION NOT NULL,
    verdict TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS identity_scores_vec_idx
    ON identity_scores USING hnsw (embedding vector_cosine_ops);

-- M0 spike results: watermark check, moderation threshold, blind test winner.
CREATE TABLE IF NOT EXISTS m0_results (
    model_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    watermark TEXT,
    moderation_threshold TEXT,
    blind_test_winner BOOLEAN DEFAULT false,
    notes JSONB,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
