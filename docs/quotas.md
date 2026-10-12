# Working within API quotas

Every provider here has a free tier with hard limits, and the project is built to stay inside them
instead of reacting to errors. Current presets (`QUOTA_PRESETS` in `app/config.py`, override in `.env`):

| Role | Provider | Requests/min | Tokens/min | Requests/day | Notes |
|---|---|---|---|---|---|
| Embeddings | Voyage (no card) | 3 (assumed) | 10K (assumed) | none | A batch counts as one request. A run at `EMBED_RPM=2 EMBED_TPM=3000` finished with no 429s. Adding a payment method raises the limits to 2,000 RPM and 16M TPM for `voyage-4-lite` |
| Rerank | Voyage (no card) | 3 (assumed) | 10K (assumed) | none | **Not confirmed** for the rerank model; assumed to match embeddings. One request per question, about 3,000 tokens (20 candidates). A 21-question eval took about 20 minutes (about one request per minute: the local limiter's token estimate is high for 20 candidates). Counter `rerank:voyage` |
| Chat | Groq (gpt-oss) | 30 | 8K | 1,000 | A 200K tokens/day cap also exists and is not modeled |
| Both | Gemini free tier | 60 | 100K | 100 | Fallback, not benchmarked |

The rest of this page explains the mechanism. The worked numbers are from the Gemini free tier,
where the 100 requests/day limit made the design necessary; the layers are provider-independent.

Reacting to 429s does not work here: Google's quota 429 carries no field saying *which* limit was hit,
so "wait a minute" and "come back tomorrow" look identical. The project counts instead (`app/quota.py`):

| Layer | Does | Where |
|---|---|---|
| Sliding-window limiter | never exceeds RPM / TPM in any rolling 60 s: waits instead of failing | in process |
| Daily budget | atomic RPD counter, resets at midnight Pacific, shared by all workers | Postgres `api_usage` |
| Governor | reserve budget, wait, call. An unexpected 429 gets one 65 s cool-down; a second marks the day exhausted | per API (`embed`, `llm`) |
| Disk cache | identical embeddings / prompts are never paid for twice, even across DB resets | `.cache/api_cache.sqlite` |
| Pause/resume | on exhaustion workers put jobs back **without burning attempts**, API returns `429 + Retry-After`, eval saves partial results | worker, `/ask`, `evals` |

```bash
make enqueue-dev    # 40-doc dev corpus (every eval-expected doc + seeded distractors): 317 chunks
make worker         # runs until done OR the day's budget is used up; just re-run after the reset
make quota          # today's usage vs limits, time to reset
```

**Decide once: `EMBED_QUOTA_UNIT`.** Default `text` counts every text in a batch as a request, so it can
never overshoot, but 100 RPD then means ~100 chunks/day (dev subset: 4 days, full corpus: 12 days; the
cache makes the days additive, not repeated). If a batch call moves Google's counter by 1 instead of N
(check https://ai.dev/rate-limit before/after one `make worker` run, compare with `make quota`), set
`EMBED_QUOTA_UNIT=call`. Ingestion then costs about one request per document.

Budgeting the day: retrieval-only evals cost only query embeddings (cached after the first run, so every
later run and every retrieval ablation is free); `--generate` costs one LLM request per question
(21 per full pass, cached). Disable the cache (`API_CACHE=false`) for final numbers, since it hides run-to-run variance.

Not solved here (the scale lever): documents are embedded one call per document. At production volume
you would coalesce chunks from concurrent workers into full batches, or use the Batch API.
