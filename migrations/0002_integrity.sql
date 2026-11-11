-- 0002: integrity fixes from the QA pass. 0001 is left untouched because sqlx
-- validates the checksum of every applied migration.

-- 1. A retried provider request must not be charged twice. Partial index:
--    request_id is optional (local $0 operations have none).
CREATE UNIQUE INDEX IF NOT EXISTS ledger_request_unique
    ON ledger (request_id)
    WHERE request_id IS NOT NULL;

-- 2. identity_scores must not outlive its project (0001 declared the column
--    without a foreign key).
ALTER TABLE identity_scores
    DROP CONSTRAINT IF EXISTS identity_scores_project_fk;
ALTER TABLE identity_scores
    ADD CONSTRAINT identity_scores_project_fk
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE;

-- 3. Deleting a project must not fail on its children.
ALTER TABLE shots DROP CONSTRAINT IF EXISTS shots_project_id_fkey;
ALTER TABLE shots
    ADD CONSTRAINT shots_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE;

ALTER TABLE jobs DROP CONSTRAINT IF EXISTS jobs_project_id_fkey;
ALTER TABLE jobs
    ADD CONSTRAINT jobs_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE;

ALTER TABLE ledger DROP CONSTRAINT IF EXISTS ledger_project_id_fkey;
ALTER TABLE ledger
    ADD CONSTRAINT ledger_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE;

-- 4. Drift queries filter by project and subject.
CREATE INDEX IF NOT EXISTS identity_scores_project_subject_idx
    ON identity_scores (project_id, subject_slug);
