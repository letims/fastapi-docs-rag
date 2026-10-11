CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id           BIGSERIAL PRIMARY KEY,
  source       TEXT NOT NULL UNIQUE,          -- e.g. tutorial/dependencies/index.md
  title        TEXT NOT NULL,
  content_hash TEXT NOT NULL,                 -- sha256 of normalized content: skip unchanged docs
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id           BIGSERIAL PRIMARY KEY,
  document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index  INT NOT NULL,
  heading      TEXT NOT NULL,                 -- breadcrumb: Title > Section > Subsection
  content      TEXT NOT NULL,
  content_hash TEXT NOT NULL,                 -- sha256(heading + content): reuse embeddings
  embedding    vector({dim}) NOT NULL,
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', heading || ' ' || content)) STORED,
  UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin(tsv);
-- The HNSW index is created AFTER bulk load (see app/db.py: build_vector_index); building it
-- while inserting slows writes down.

-- Postgres as the job queue: claim with FOR UPDATE SKIP LOCKED.
CREATE TABLE IF NOT EXISTS ingest_jobs (
  id          BIGSERIAL PRIMARY KEY,
  source      TEXT NOT NULL,
  title       TEXT NOT NULL,
  content     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'pending'
              CHECK (status IN ('pending','running','done','failed')),
  attempts    INT NOT NULL DEFAULT 0,
  run_after   TIMESTAMPTZ NOT NULL DEFAULT now(),   -- backoff: next time it may be claimed
  locked_at   TIMESTAMPTZ,
  last_error  TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- At most one live job per source: enqueueing the same doc twice is a no-op.
CREATE UNIQUE INDEX IF NOT EXISTS ingest_jobs_live_source
  ON ingest_jobs(source) WHERE status IN ('pending','running');
CREATE INDEX IF NOT EXISTS ingest_jobs_claim_idx ON ingest_jobs(status, run_after);

-- Client-side daily request counter (see app/quota.py). One row per (Pacific day, api).
CREATE TABLE IF NOT EXISTS api_usage (
  day      DATE NOT NULL,
  api      TEXT NOT NULL,            -- 'embed' | 'llm'
  requests INT  NOT NULL DEFAULT 0,
  PRIMARY KEY (day, api)
);
