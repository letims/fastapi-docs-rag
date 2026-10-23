---
name: retrieval-eval
description: Use when a change touches retrieval, reranking, chunking, embeddings, prompts or the agent loop in fastapi-docs-rag (app/rag/, app/ingest/chunker.py, evals/). Runs the eval, compares to the stored baseline, gets a subagent review and records the result. Not for provider/quota-only changes.
---

# Change retrieval → measure → compare → record

Every number this produces must come from a saved file in `evals/results/`.

1. **State the hypothesis** in one line: what should move (e.g. "cap per source fixes q16's
   full_recall without hurting MRR"). Note whether it changes chunk hashes (chunker, `embed_text`,
   `max_chars`, embed model) — if yes, it needs a re-index: stop and ask before `make enqueue`/`make worker`.
2. **Make the change** with tests. The Stop hook runs unit tests; `make check` before committing.
3. **Retrieval eval vs baseline** (cached, near-free):
   `make eval-compare MODE=<vector|hybrid|rerank>` (env knobs work: `RERANK_MAX_PER_SOURCE=2 make eval-compare MODE=rerank`).
   Exit 1 = regression beyond one question. Read the "Questions that moved" list, not just the totals.
4. **Answer-level eval only if step 3 changed what reaches the LLM, or the prompt changed.** Ask the
   user first (Groq daily quota): `make eval-gen MODE=…` or `make eval-judge MODE=…`, then
   `uv run python -m evals.compare --latest <mode>-gen`.
5. **Review**: send the diff (`git diff main...HEAD -- app evals`) and the compare output to the
   `rag-eval-reviewer` subagent. Fix `block` findings before going on.
6. **Record**: add a README ablation row citing the result file name (corpus size, LLM or not).
   21 questions is a smoke test: call a one-question difference noise.
7. **Baseline**: promote only when the change is merged and is the new default:
   `make eval-baseline RUN=evals/results/<file>.json`. Never promote to make a failing compare pass.

Report back: hypothesis, the compare table, questions that moved, reviewer verdict, and the files
added. If a step was skipped (no quota, no index), say which and why.
