# Ingestion design (and why)

The sequence diagrams of a worker running a job (happy path and failure paths) are in the
[README](../README.md#how-ingestion-works).

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
