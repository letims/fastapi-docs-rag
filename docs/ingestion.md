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
A job whose lock outlives the visibility timeout is reclaimed (crashed worker). A live worker renews its lock in the background (`queue.heartbeat`, every third of the timeout), so a job that is only waiting on a rate limit is not mistaken for a dead one. A partial unique
index allows one live job per source, so double-enqueueing is a no-op.
*When I would change it:* above a few hundred jobs/s or with cross-service fan-out, SQS / Cloud
Tasks / Redis streams; the claim/ack/fail interface in `app/ingest/queue.py` is the seam.

**Not built:** removing documents. Re-submitting a changed document replaces its chunks, but a source file
that disappears from the corpus leaves its document and chunks in the database. Deleting them needs a
sweep that compares stored sources with the current list.

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

## Pipeline throughput (load test)

`make load-test` measures the pipeline with the embedding provider taken out: the offline fake
embedder, a throwaway Postgres (no Docker, no API, no quota) and real FastAPI docs repeated and
rewritten so every chunk is unique (otherwise chunk-hash reuse would skip the embedding step and
the test would measure the shortcut). `--latency-ms` adds a fixed delay per embedding call to
imitate an API round trip. Other shapes:

```bash
uv run python -m scripts.load_test --docs 2000 --concurrency 4,16,64 --latency-ms 200
uv run python -m scripts.load_test --database-url postgresql://... --drop-tables   # your own server
```

Measured on a 4 vCPU Linux VM (one Python process, throwaway Postgres, 1,000 docs ≈ 7,900 chunks,
1024-d vectors, 7.9 chunks per doc). Absolute numbers depend on the machine; the shape does not.

| Embedding call latency | Workers | docs/s | chunks/s |
|---|---|---|---|
| 0 ms | 1 | 39 | 307 |
| 0 ms | 4 | 66 | 524 |
| 0 ms | 8 | 68 | 543 |
| 0 ms | 16 | 72 | 568 |
| 200 ms | 4 | 15.7 | 124 |
| 200 ms | 16 | 40.8 | 323 |
| 200 ms | 64 | 55.6 | 441 |

What it says:

- **The pipeline is not the bottleneck.** One process tops out near 70 docs/s (about 550
  chunks/s), where the limit is Python and Postgres CPU, not the queue. The sample problem's target
  of 14 docs/s needs about 4 workers when each embedding call takes 200 ms (15.7 docs/s measured).
  50,000 docs would take about 12 minutes of pipeline time at the plateau.
- **Workers pay off while calls are slow.** At 0 ms, going from 4 to 16 workers adds little; at
  200 ms it multiplies throughput, until the same CPU ceiling is reached. Past that point you add
  processes (`make worker` more than once), not tasks.
- **One embedding request per document.** A document averages 7.9 chunks and the batch size is 64,
  so each document costs one API request. That makes the provider's requests-per-minute limit the
  real ceiling: 3 RPM without a payment method is 3 docs/min; 2,000 RPM would not be the limit.
  Batching chunks from several documents into one request would raise the ceiling when the limit is
  on requests rather than tokens.
- **Index build:** about 3 to 6 seconds for 7,900 chunks. It runs once, after the load.

What it does not say: throughput with a 395k-chunk table (inserts into a large table, autovacuum,
a larger working set), real network behavior, or what a managed Postgres would do. The tables here
are small and everything is on one machine.

*Measured ingest on the free tier (the actual run):* roughly one to two documents per minute, limited by the
3 requests per minute embedding quota, as described in [Quotas](quotas.md).
