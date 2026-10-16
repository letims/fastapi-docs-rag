# Evaluation method

`evals/questions.jsonl` has 21 questions grounded in the corpus: 13 single-fact, 3 multi-hop
(need two documents), 1 partial (docs only touch the topic), 4 unanswerable (verified absent by
grep, so the correct behavior is to abstain). Each answerable question lists the source documents
that contain the answer (`expected_sources`).

## Metrics

| Metric | Meaning | Needs the LLM |
|---|---|---|
| hit@k | any expected source appears among the top-k retrieved chunks' documents | no |
| full_recall@k | **all** expected sources appear (matters for multi-hop) | no |
| MRR | 1 / rank of the first expected source, averaged | no |
| cited_ok | the answer cites at least one expected source | yes |
| false_abstain | the model said `INSUFFICIENT_CONTEXT` although the answer exists | yes |
| abstain rate | on the 4 unanswerable questions; should be 100% | yes |
| tokens/query, latency | cost and speed of the full answer | yes |

Agent runs (`--agent`) add per-question `rounds`, `stop_reason`, the queries tried and the `trace`, and
the summary adds average rounds, tokens per question and a count of stop reasons.

hit@k, full_recall@k and MRR are computed over the 17 answerable questions. The top-k are chunks,
so several can come from one document and k=5 chunks may cover only 2 or 3 documents.

## Running it

```bash
make eval        # retrieval only: needs the embedding key, cheap; prints [i/N] progress and time left
make eval-gen    # + answers, citations, abstention: also calls the chat model (slow on Groq's free tier)

make eval MODE=hybrid      # vector + full-text, fused with RRF (no extra API calls: queries are cached)
make eval MODE=rerank      # + cross-encoder rerank (one rerank request per question; about 20 min for 21 questions on the free tier)
make eval-gen MODE=rerank  # the same retriever, with answers
make eval-agent MODE=rerank ONLY=q14,q16,q17   # Phase 3 loop, only the known failures (cheap first look)
make eval-agent MODE=rerank                    # all 21 questions (several LLM calls each: mind Groq's daily cap)
RERANK_MAX_PER_SOURCE=2 make eval MODE=rerank  # rerank with the per-document cap (settings are read from the environment)
```

`MODE` defaults to `vector`. Each mode saves its own file (`hybrid-<time>.json`, ...), so runs
compare side by side in the README table.

Both save a JSON file in `evals/results/` (summary plus one row per question with the retrieved
sources). Question embeddings and answers are cached on disk, so a rerun is nearly free. Disable the
cache (`API_CACHE=false`) for final numbers, because it hides run-to-run variance. If the daily
quota runs out mid-run, the script saves the finished questions and exits with status 2; rerun after
the reset.

## Rules for the results table

- A number goes in the README only if a saved run in `evals/results/` backs it, and the row says
  which corpus (40-doc dev subset or full 147 docs) and whether the LLM was involved.
- Every retrieval or agent change ships with a run and a row, so each step's effect is measured.
- 21 questions is a smoke test. A difference of one question is noise.
