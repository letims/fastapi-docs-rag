# Logging

Every entry point (`make worker`, `make api`, `make eval*`, the enqueue script) calls
`setup_logging()` in `app/logging_config.py`. Set `LOG_LEVEL` in `.env` (`DEBUG`, `INFO` default,
`WARNING`). The worker also prints a progress line every `PROGRESS_INTERVAL_S` seconds (default 30).

| Level | What you see |
|---|---|
| `INFO` | Decisions and milestones: what each document did and why, retries and backoff, queue state, retrieval results, every step of the agent loop, quota thresholds |
| `DEBUG` | Per-item detail: each claimed job, embedding batches, cache hits and misses, hybrid fusion overlap |
| `WARNING` | Something is degraded or will be retried: a failed job, a 429, 80% of the daily budget, an unusable grader reply |

Secrets never appear in logs. Question text (first 60 characters) is logged at `INFO`; use
`LOG_LEVEL=WARNING` where that is not acceptable.

## Sentry (errors and traces)

`setup_sentry()` in `app/observability.py` runs when the API module loads. It does nothing unless
`SENTRY_DSN` is set, so tests and `PROVIDER=fake` runs never send events. `SENTRY_ENVIRONMENT`
(default `development`) and `SENTRY_TRACES_SAMPLE_RATE` (default `0.1`) are settings too; set the
release with the `SENTRY_RELEASE` env var at deploy time. Request bodies, local variables and PII
are off, because they can hold prompts and document text.

## What a worker run looks like

```
worker starting 4 workers (embedding batch 64, lock timeout 300s, max 5 attempts per job)
worker queue at start: {'pending': 147} | already stored: 0 documents, 0 chunks
app.ingest.pipeline tutorial/cors.md: new document: 9 chunks, 0 reuse a stored embedding, 9 to embed
worker w0 ingested tutorial/cors.md: 9 chunks (9 embedded, 0 reused) in 2.1s
app.ingest.pipeline tutorial/bigger-applications.md: content changed: 14 chunks, 12 reuse a stored embedding, 2 to embed, 1 old chunks dropped
worker w1 skipped tutorial/first-steps.md (unchanged)
app.ingest.queue job 31: attempt 1 of 5 failed; retry in 8s (exponential backoff with jitter)
app.quota daily budget embed:voyage: 80 of 100 requests used (80%); it resets in 6.2 h
worker progress: 52 done this run, 90 pending, 4 running, 1 failed | 11.3 docs/min | about 8 min left
worker embedding cache: 40 texts served from disk (free), 120 sent to the API
app.db vector index: building HNSW over 1161 chunks (created after the bulk load)
```

(The lines above are illustrative of the format; the numbers are made up.)

## The decisions that are logged

- **Enqueue:** how many documents were received, how many were new, how many already queued.
- **Claim:** a job on its second attempt or later says so, because that means a failure or a dead
  worker; the first attempt is `DEBUG`.
- **Ingest:** new document, content changed (how many chunks reuse an embedding, how many are
  embedded, how many old ones were dropped) or unchanged (skipped with no embedding calls).
- **Failure:** the retry delay from the exponential backoff, or giving up after the last attempt.
- **Quota:** a job put back without using an attempt, the 80% daily-budget warning, the limiter
  waiting for the minute window, and a 429 cool-down.
- **Retrieval:** per query, mode, number of chunks and documents, and the top sources; when the
  per-document cap replaced results, which documents it pushed out.
- **Answer:** passages given, passages cited, answered or abstained, tokens, and a warning when the
  model neither abstains nor cites.
- **Agent loop:** each round's queries, new chunks and new documents; the grader's verdict
  (including when the parser overrode "sufficient"); the plan chosen; why it stopped.
