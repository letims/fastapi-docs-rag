# ragnarok

An agentic RAG system over the FastAPI documentation, built to answer one question honestly:
**what happens when the first retrieval pass is not enough, and how do you know it got better?**

> Status: Phase 1 (ingestion + baseline vector RAG + eval harness). Numbers below are filled in
> only from saved runs in `evals/results/`; empty cells mean "not measured yet".

## Quickstart

```bash
cp .env.example .env            # add GEMINI_API_KEY (free key: aistudio.google.com/apikey)
make up                         # Postgres 16 + pgvector on :5433
make corpus                     # fetch FastAPI docs (markdown) into ./corpus
make enqueue && make worker     # queue every doc, embed with N workers, build the HNSW index
make api                        # http://localhost:8000/docs
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How do I run code at startup and shutdown?"}' | jq
make eval                       # retrieval metrics on evals/questions.jsonl
```

No API key? `PROVIDER=fake` runs everything offline (plumbing only, not quality).

## Architecture

```
 docs ──► POST /documents or scripts.enqueue_corpus
              │
              ▼
        ingest_jobs (Postgres queue, FOR UPDATE SKIP LOCKED)
              │  claim / retry with backoff+jitter / dead-letter
              ▼
        N async workers ─► chunk (markdown-aware) ─► embed (batched) ─► one txn per doc
              │                                                       │
              ▼                                                       ▼
        documents (content hash)                      chunks (hash, vector, tsvector)
                                                              │
 question ─► embed ─► vector search ─► top-k ─► LLM (cite or abstain) ─► answer + citations
```

Provider interfaces (`app/providers/base.py`) isolate the model vendor: Gemini free tier today,
Claude or anything else is one adapter file and a config change.

## Ingestion design (and why)

**Idempotency, three layers.** Unchanged document hash → skip with zero embedding calls.
Changed document → chunks are matched by content hash and only new chunks are embedded.
Each document is written in a single transaction (upsert doc, replace chunks), so a crash never
leaves half a document. Re-running the corpus is cheap and safe.

**Queue in Postgres.** `SELECT … FOR UPDATE SKIP LOCKED` gives safe concurrent claiming with one
less service to run, and jobs are transactional with the data they produce. Failures retry with
exponential backoff and full jitter (`run_after`), then dead-letter after `JOB_MAX_ATTEMPTS`.
A job whose lock outlives the visibility timeout is reclaimed (crashed worker). A partial unique
index allows one live job per source, so double-enqueueing is a no-op.
*When I would change it:* above a few hundred jobs/s or with cross-service fan-out, SQS / Cloud
Tasks / Redis streams; the claim/ack/fail interface in `app/ingest/queue.py` is the seam.

**The queue is not the bottleneck; embedding is.** Measured on this corpus (147 docs after
excluding changelog/people pages):

| | |
|---|---|
| chunks | 1,161 (mean 7.9 per doc, max 39) |
| chunk size | mean 619 chars, p50 512, p90 1,228, max 1,896 |
| tokens (≈ chars/4) | ≈ 180k total, ≈ 155 per chunk |

Scaling to the sample problem's target of 50,000 docs in an hour (≈ 14 docs/s) at this shape
is ≈ 110 chunks/s ≈ 17k embedding tokens/s ≈ 1M tokens/min. That is a provider quota question,
not a queue question. Levers, in order: batch many chunks per embedding call, rate-limit with
backoff on 429 (done), bulk-insert with `COPY` instead of `executemany` (not done yet), build the
HNSW index after load (done), request a higher quota or spread across keys/providers.
*Measured ingest throughput on the real run:* _pending_.

## Retrieval and the agent loop

| Phase | Change | Status |
|---|---|---|
| 1 | Vector search over pgvector, cited answers, abstain token | built |
| 2 | Hybrid: Postgres full-text + vector, Reciprocal Rank Fusion, cross-encoder rerank | planned |
| 3 | LangGraph loop: retrieve → grade sufficiency → (rewrite \| decompose \| widen) → retry → answer or abstain | planned |

When a pass is not enough, the plan is to diagnose, not just retry: a cheap score check first
(no LLM call), then a structured sufficiency grader that names the missing sub-questions, then a
targeted action (rewrite for vocabulary mismatch, decompose for multi-part questions, widen
filters for over-narrow retrieval), under hard caps on iterations and tokens. If it still cannot
answer, it answers what it can with citations and says what it could not find.

## Evaluation

`evals/questions.jsonl` has 21 questions grounded in the corpus: 13 single-fact, 3 multi-hop
(need two documents), 1 partial (docs only touch the topic), 4 unanswerable (verified absent by
grep, so the correct behavior is to abstain).

Metrics: hit@k, full_recall@k (all expected sources, matters for multi-hop), MRR, cited_ok,
false_abstain, abstain rate on unanswerable, tokens, latency.

| Configuration | hit@5 | full_recall@5 | MRR | cited_ok | abstain (unanswerable) | tokens/query | latency |
|---|---|---|---|---|---|---|---|
| Vector only | | | | | | | |
| + hybrid (RRF) | | | | | | | |
| + rerank | | | | | | | |
| + agent loop | | | | | | | |

Caveat to keep honest: 21 questions is a smoke test, not a benchmark. Differences of one
question are noise; the table is for direction and for finding failure modes, which I list here
as they are found.

### Known failure modes
_to be filled from real runs_

## Limits and what I would do next
- Free-tier Gemini: rate limits apply, and Google may use free-tier data to improve its products
  (public docs only here).
- Chunks under 200 chars are 12% of the corpus; merging tiny sections into neighbors may help.
- Contextual retrieval (LLM-written context prepended to each chunk) is a likely next gain.
- Production: tracing (Langfuse/LangSmith), drift monitoring on the eval set in CI, per-query
  cost budgets, and a real queue.
