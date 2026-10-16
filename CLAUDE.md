# fastapi-docs-rag

Agentic RAG over the FastAPI docs. Built as a prototype for a take-home: ingestion at scale,
hybrid retrieval, an agent loop that notices when retrieval was not enough, and an eval harness
that proves each step helped.

## Stack
FastAPI (async) · Postgres 16 + pgvector (vectors, full-text, and the job queue) · psycopg 3 ·
Voyage (embeddings) + Groq (chat) free tiers, Gemini optional, all behind provider interfaces · LangGraph (agent loop) · uv · pytest · ruff.

## Commands
- `make up` / `make down`: Postgres via docker compose (host port 5433)
- `make corpus`: fetch FastAPI docs into `corpus/` (gitignored)
- `make enqueue` then `make worker`: queue and embed the corpus
- `make check`: lint + format check + tests (what the pre-commit hook and CI run); `make fmt` fixes style
- `make install-hooks`: enable the pre-commit hook on a fresh clone
- `make api`: run the API · `make eval`: retrieval metrics · `make lint`
- `make test`: full suite + coverage. **Fails under 90% (line+branch)**; HTML report in `htmlcov/index.html`
  (`make cov-open`). `make test-fast`: no coverage, stops at first failure.
- Offline dev with no API key: `PROVIDER=fake`. The fake embedder is bag-of-words hashing:
  good for plumbing, NOT a quality baseline. Never report eval numbers from it.
- DB tests use `TEST_DATABASE_URL=<conninfo>` if set (DROPS the app tables there), otherwise a throwaway
  Postgres+pgvector from the `pgserver` dev dependency (no Docker needed). They fail loudly, never skip.
- New code needs tests in the same change; do not lower `fail_under` or add `omit`s to pass the gate.

## API quotas (Gemini free tier: 60 RPM / 100K TPM / 100 RPD)
- Never call a provider directly; go through `get_llm()` / `get_embedder()` (Cached -> Governed -> Gemini).
- The Gemini adapter must NOT retry 429s (its 5xx retry is separate). `app/quota.py` owns 429 handling:
  Google's 429 does not say which quota was hit, so we count RPM/TPM in process and RPD in Postgres.
- `QuotaExhausted` means pause, not fail: release the job (`queue.release`, no attempt burned), stop, resume later.
- `EMBED_QUOTA_UNIT=text` is the safe default; `call` only after verifying on ai.dev/rate-limit.
- Dev loop: `make enqueue-dev`, retrieval-only evals (free after first run), `make quota`. Do not delete `.cache/`.

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
- New vendor = one adapter + a branch in `app/providers/__init__.py` + a `QUOTA_PRESETS` row + mocked-HTTP tests.
  Daily counters are keyed `role:provider`.
- Secrets only in `.env` (gitignored). Free-tier Gemini data may be used by Google for product
  improvement: public docs only.

## Docs layout
README is deliberately short (context, setup, how to read the results). Detail lives in `docs/`:
architecture, ingestion, quotas, commands, evaluation, testing. Put new detail there, not in the README.

## Retrieval and agent notes
- `search(mode, ...)` in `app/rag/retrieve.py` is the one retrieval entry point (vector | hybrid | rerank).
  `RERANK_MAX_PER_SOURCE` (default 0 = off) caps chunks per document after reranking.
- The agent loop (`app/rag/agent.py`) must stay bounded: max rounds, token budget, no-progress rule.
  A grader reply it cannot parse fails open (answer with what was found), never crashes or retries forever.
- The API is stateless (no conversation memory). Do not describe it as a multi-turn chatbot.

## Roadmap (see README for status)
1. Baseline vector RAG + eval (done)  2. Hybrid + RRF + reranker (done, measured)
3. LangGraph loop: grade → rewrite/decompose → retrieve → generate, caps, abstain (built; 3-question runs only)
4. Tracing + cost/latency per query (not built)  5. Throughput measurement (pipeline measured with `make load-test`)
6. Chat layer: sessions, follow-up rewriting, persona and guardrails (designed, not built)
