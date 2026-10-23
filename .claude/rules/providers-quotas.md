---
paths:
  - "app/providers/**"
  - "app/quota.py"
  - "app/cache.py"
  - "app/retry.py"
  - "app/worker.py"
  - "app/ingest/queue.py"
  - "scripts/quota.py"
---

# Providers and quotas

- The app depends only on `app/providers/base.py`. Get clients via `get_llm()` / `get_embedder()` /
  `get_reranker()` (Cached → Governed → vendor). Never instantiate a vendor client elsewhere.
- New vendor = one adapter + a branch in `app/providers/__init__.py` + a `QUOTA_PRESETS` row in
  `app/config.py` + mocked-HTTP tests. Daily counters are keyed `role:provider`.
- `app/quota.py` owns 429 handling. Adapters must NOT retry 429s (5xx retry is separate, in `app/retry.py`).
  Gemini's 429 doesn't say which limit was hit, so RPM/TPM are counted in process and RPD in Postgres.
- `QuotaExhausted` means pause, not fail: `queue.release` the job (no attempt burned), stop, resume later.
- `EMBED_QUOTA_UNIT=text` is the safe default; `call` only after checking the vendor console.
- Free-tier reference: Voyage 3 RPM / 10K TPM (no card); Groq gpt-oss 30 RPM / 8K TPM / 1K RPD;
  Gemini 60 RPM / 100K TPM / 100 RPD. Confirm in each console before trusting these.
- Prices for `cost_usd` in evals live in `PRICES` in `app/config.py` (list price per 1M tokens).
  Update the row and its `as_of` date when a vendor changes pricing.
