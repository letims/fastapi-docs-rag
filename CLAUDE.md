# ragnarok

Agentic RAG over the FastAPI docs. Built as a prototype for a take-home: ingestion at scale,
hybrid retrieval, an agent loop that notices when retrieval was not enough, and an eval harness
that proves each step helped.

## Stack
FastAPI (async) · Postgres 16 + pgvector (vectors, full-text, and the job queue) · psycopg 3 ·
Gemini free tier behind provider interfaces · LangGraph (Phase 3) · uv · pytest · ruff.

## Commands
- `make up` / `make down`: Postgres via docker compose (host port 5433)
- `make corpus`: fetch FastAPI docs into `corpus/` (gitignored)
- `make enqueue` then `make worker`: queue and embed the corpus
- `make api`: run the API · `make eval`: retrieval metrics · `make lint`
- `make test`: full suite + coverage. **Fails under 90% (line+branch)**; HTML report in `htmlcov/index.html`
  (`make cov-open`). `make test-fast`: no coverage, stops at first failure.
- Offline dev with no API key: `PROVIDER=fake`. The fake embedder is bag-of-words hashing:
  good for plumbing, NOT a quality baseline. Never report eval numbers from it.
- DB tests use `RAGNAROK_TEST_DB=<conninfo>` if set (DROPS the app tables there), otherwise a throwaway
  Postgres+pgvector from the `pgserver` dev dependency (no Docker needed). They fail loudly, never skip.
- New code needs tests in the same change; do not lower `fail_under` or add `omit`s to pass the gate.

## Conventions
- Rest of the app depends on `app/providers/base.py` interfaces only. Model IDs live in config.
- Every external call goes through `app/retry.py` (backoff + full jitter). Retry only 429/5xx.
- Ingestion must stay idempotent: doc hash, chunk hash, one transaction per doc. Any change to
  chunking or the embed text changes chunk hashes, which re-embeds everything: say so in the PR.
- Never hold a DB connection across a network call.
- The LLM must answer only from retrieved context, cite as `[n]`, and emit `INSUFFICIENT_CONTEXT`
  when it cannot. Abstention is a feature with its own metric.
- Every retrieval or agent change ships with an eval run saved in `evals/results/` and a row in
  the README ablation table. No number in the README without a saved run behind it.
- Secrets only in `.env` (gitignored). Free-tier Gemini data may be used by Google for product
  improvement: public docs only.

## Roadmap (see README for status)
1. Baseline vector RAG + eval  2. Hybrid (tsvector + pgvector) + RRF + reranker
3. LangGraph loop: grade → rewrite/decompose → retrieve → generate, caps, abstain
4. Tracing + cost/latency per query  5. Throughput measurement + write-up
