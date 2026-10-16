# fastapi-docs-rag

A retrieval-augmented generation (RAG) system over the FastAPI documentation: it ingests the docs,
answers questions with citations to the pages it used, and says "I can't tell from these docs" when
the answer is not there. Its real subject is the failure case:
**what happens when the first retrieval pass is not enough, and how do you know it got better?**

## About

**Context.** Most RAG demos stop the moment a question gets a plausible answer. The interesting
engineering starts after that: loading tens of thousands of documents without duplicating work or
losing jobs, noticing that the retrieved passages do not actually answer the question, trying again
in a controlled way, and proving with numbers that each change helped. The FastAPI docs make a good
test bed: public, well written, small enough to run on a laptop, and familiar enough that I can tell a
good answer from a confident wrong one. Four of the eval questions are deliberately unanswerable from
those docs, so the correct behavior is to abstain.

**Goal.** Build a small but honest end-to-end system: queue-based, idempotent ingestion that could
scale to something like 50,000 documents an hour; retrieval with citations; an agent loop that grades
whether the evidence is sufficient and then rewrites, decomposes or widens the search under hard caps;
and an evaluation harness so every step is measured against the last one instead of judged by feel.

**Motivation.** I am a full-stack engineer learning agentic patterns by building one, and I wanted a
project where the design decisions (queues, retries, idempotency, cost and rate limits, evaluation)
are as visible as the model calls. It also runs entirely on free API tiers (Voyage for embeddings,
Groq for answers), which turned quota handling into part of the design instead of an afterthought.

> Status: Phase 1 is done (queue-based ingestion, baseline vector RAG, eval harness, quota-safe
> providers). Phase 2 is built and measured: reranking helped, hybrid search did not (see Phase 2
> results). Phase 3 (the agent loop, LangGraph) is built and was run on three questions, twice;
> that is too few to say whether it helps, and the full 21-question run is not done. The system answers
> any number of separate questions but keeps no conversation memory between turns. Numbers below come
> only from saved runs in `evals/results/`; empty cells mean "not measured yet".

## Documentation

If you want the big picture first, start with [architecture](docs/architecture.md): it has the system
diagram, explains how the embedding and chat providers can be swapped independently, and lays out the
three retrieval phases and how the agent loop works. The part I would most like you to read is
[ingestion design](docs/ingestion.md), which covers why a Postgres queue is enough at this scale, how
retries, dead-lettering and idempotency work, and the throughput math behind the 50,000-documents-an-hour
target (the sequence diagrams for it are further down this page). Because free tiers shape so much of
the design, [API quotas](docs/quotas.md) explains the rate limiter, the daily budget, the disk cache and
how a run pauses and resumes instead of failing.

When you want to run things, [make commands](docs/commands.md) walks through every `make` target in the
order you would use them, what each one needs first, and the pre-commit hook that blocks a commit on
lint errors, failing tests or a leaked key. To understand the numbers, [evaluation method](docs/evaluation.md)
describes the 21 questions, the metrics and the rules for what may appear in the results table, and
[tests and coverage](docs/testing.md) covers the test suite and the 90% coverage gate.

## Quickstart

You need Docker, [uv](https://docs.astral.sh/uv/), and free API keys: Voyage AI (embeddings) and
Groq (chat). Put the keys in `.env` only, never in chat, issues or commits.

```bash
cp .env.example .env            # then add VOYAGE_API_KEY and GROQ_API_KEY
make up                         # Postgres 16 + pgvector on :5433
make init                       # create tables (once per fresh database)
make corpus                     # fetch the FastAPI docs (markdown) into ./corpus
make enqueue-dev && make worker # queue a 40-doc subset and embed it (or `make enqueue` for all 147)
make api                        # http://localhost:8000/docs
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How do I run code at startup and shutdown?"}' | jq
make eval                       # retrieval metrics on the 21 eval questions
```

No API keys? `PROVIDER=fake` runs everything offline (plumbing only, not quality).
Providers and models are chosen in `.env`; see [Architecture > Providers](docs/architecture.md#providers).
Free tiers are slow: read [API quotas](docs/quotas.md) before a big ingest.

**Check your keys before ingesting** (two real calls, after `make up && make init`):

```bash
uv run python - <<'EOF'
import asyncio
from app.providers import get_embedder, get_llm
async def main():
    v = await get_embedder().embed(["hello"], kind="query")
    print("embed ok, dim:", len(v[0]))
    r = await get_llm().generate("Say OK.")
    print("llm ok:", r.text, r.input_tokens, r.output_tokens)
asyncio.run(main())
EOF
```

Troubleshooting:
- `EMBED_DIM ... got 768`: Voyage needs `EMBED_DIM=1024`. If `.env` already says so, a variable
  exported in your shell wins over `.env`: check `env | grep -E "EMBED|LLM_|PROVIDER"` and `unset` it.
- `relation "api_usage" does not exist`: run `make init`.
- Changing embedding model or `EMBED_DIM` needs a fresh database (`docker compose down -v`).
- A 400 from Voyage or Groq usually means a wrong model name; the response body is in the error.
- `make eval` looks stuck: with a low `EMBED_RPM` it waits between questions and logs the wait.

## How ingestion works

Workers pull jobs from a Postgres queue; each job is one document. The first diagram is the normal
path, the second shows what happens when something goes wrong. Why it is built this way, and the
throughput math, are in [Ingestion design](docs/ingestion.md).

### Happy path

Several workers run this loop in parallel against the same queue:

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker (w0..w3)
    participant Q as ingest_jobs (Postgres)
    participant P as ingest_document
    participant D as documents / chunks
    participant C as Cache
    participant G as Governor (quota)
    participant V as Voyage API

    loop until the queue is empty (--drain)
        W->>Q: claim(): oldest runnable job<br/>(FOR UPDATE SKIP LOCKED, attempts + 1)
        Q-->>W: job (source, title, content)
        W->>P: ingest_document()
        P->>D: document hash unchanged?
        alt unchanged
            D-->>P: yes, skip (zero API calls)
        else new or changed
            P->>P: chunk markdown, hash each chunk
            P->>D: reuse embeddings of unchanged chunks
            P->>C: embed only the new chunks
            alt cache hit
                C-->>P: vectors (no quota used)
            else cache miss
                C->>G: embed batch
                G->>Q: reserve daily budget (atomic, api_usage)
                G->>G: wait for RPM / TPM window
                G->>V: POST /embeddings (no DB connection held)
                V-->>G: vectors
                G-->>C: vectors (stored in cache)
                C-->>P: vectors
            end
            P->>D: ONE transaction: upsert document, replace chunks
        end
        P-->>W: result (ingested or skipped)
        W->>Q: complete()
    end
    W->>D: after the drain: build the HNSW index once
```

### Failure paths

Each one is handled per job, so one bad document never stops the run:

```mermaid
sequenceDiagram
    autonumber
    participant W as Worker
    participant Q as ingest_jobs (Postgres)
    participant G as Governor (quota)
    participant V as Voyage API

    rect rgb(0, 0, 0)
    Note over W,V: A. Transient error (5xx, timeout)
    W->>G: embed batch
    G->>V: POST /embeddings
    V--xG: 503
    G->>V: retry with exponential backoff + jitter
    V-->>G: vectors (the job carries on)
    end

    rect rgb(0, 0, 0)
    Note over W,Q: B. Job keeps failing
    W->>Q: fail(): status pending, run_after = now + backoff with jitter
    Q-->>W: another worker (or this one) picks it up later
    W->>Q: after JOB_MAX_ATTEMPTS: status failed (dead letter, last_error kept)
    end

    rect rgb(0, 0, 0)
    Note over W,Q: C. Worker crashes mid-job
    W--xQ: process dies, job stays running
    Q->>Q: lock older than the visibility timeout counts as claimable again
    Q-->>W: claim() hands it to a live worker (the attempt still counts)
    end

    rect rgb(0, 0, 0)
    Note over W,V: D. Quota hit: a pause, not a failure
    W->>G: embed batch
    G->>Q: reserve daily budget: limit reached
    G--xW: QuotaExhausted (no network call made)
    W->>Q: release(): back to pending until the reset, attempt NOT burned
    W->>W: set stop, all workers exit cleanly
    Note over G,V: An unexpected 429 waits out the minute window once (65 s),<br/>a second 429 marks the day exhausted and follows the same path
    end
```

## Reading the results

How the metrics are defined, and how to run an eval, is in [Evaluation method](docs/evaluation.md).
In short: **hit@5** asks "did retrieval find a right document", **full_recall@5** asks "did it find
*all* the documents a multi-part question needs", **MRR** rewards ranking the right one first, and the
LLM-based columns (cited, abstain, tokens, latency) need `make eval-gen`.

**Runs so far** (saved in `evals/results/`; Run 1 is kept locally, not pushed):

1. **Dev subset, retrieval only (40 docs).** The 40 docs are the ones the questions point at plus
   distractors, so this is the easy case.
2. **Full corpus, retrieval only (147 docs, 1,161 chunks).** `vector-1791813805.json`.
3. **Full corpus, retrieval + generation with Groq (21 questions).** `vector-gen-1791813768.json`, then again after the citation fix (`vector-gen-1791814944.json`).
4. **Full corpus, hybrid search, retrieval only.** `hybrid-1791820936.json`.
5. **Full corpus, hybrid candidates + Voyage rerank, retrieval only.** `rerank-1791820713.json`.
6. **Same, with at most 2 chunks per document (`RERANK_MAX_PER_SOURCE=2`), retrieval only.** `rerank-1791828982.json`.
7. **Rerank with generation, without and with the cap.** `rerank-gen-1791832103.json` and `rerank-gen-1791833929.json` (the two later `rerank-gen-*` files are cache re-runs of the same settings and give identical numbers).

| Configuration | hit@5 | full_recall@5 | MRR | cited_ok | false_abstain | abstain (unanswerable) | tokens/query | latency |
|---|---|---|---|---|---|---|---|---|
| Vector only, 40 docs (retrieval) | 1.00 (17/17) | 0.94 (16/17) | 0.97 | not run | not run | not run | not run | not run |
| Vector only, 147 docs (retrieval) | 1.00 (17/17) | 0.88 (15/17) | 0.97 | not run | not run | not run | not run | not run |
| Vector only, 147 docs + generation (before citation fix) | 1.00 (17/17) | 0.88 (15/17) | 0.97 | 0.76 (13/17) | 0.06 (1/17) | 1.00 (4/4) | ~1,440 | see note |
| Vector only, 147 docs + generation (after citation fix) | 1.00 (17/17) | 0.88 (15/17) | 0.97 | 0.88 (15/17) | 0.06 (1/17) | 1.00 (4/4) | ~1,600 | see note |
| + hybrid (RRF), retrieval | 1.00 (17/17) | 0.88 (15/17) | **0.73** | not run | not run | not run | not run | not run |
| + rerank (on hybrid candidates), retrieval | 1.00 (17/17) | **0.94 (16/17)** | 0.94 | not run | not run | not run | not run | see note |
| + rerank with at most 2 chunks per document, retrieval | 1.00 (17/17) | **1.00 (17/17)** | 0.94 | not run | not run | not run | not run | see note |
| + rerank, generation | 1.00 (17/17) | 0.94 (16/17) | 0.94 | 0.88 (15/17) | 0.06 (1/17) | 1.00 (4/4) | ~1,570 | see note |
| + rerank with cap of 2, generation | 1.00 (17/17) | **1.00 (17/17)** | 0.94 | 0.88 (15/17) | 0.06 (1/17) | 1.00 (4/4) | ~1,640 | see note |
| + agent loop (rerank), only q14, q16, q17 | see Phase 3 section | | | | | | ~4,000 | see note |

hit@5, full_recall@5, MRR, cited_ok and false_abstain are over the 17 answerable questions;
abstain is over the 4 unanswerable ones. "not run" means that run did not call the LLM. Tokens per
query is input plus output tokens of the answer call (21,000 in + 9,200 out over 21 questions).
The corpus is 156 markdown files; 9 (release notes, newsletter, contributing, translation banner
and similar) are excluded on purpose, which leaves 147.

**Latency note.** Average latency is 20 s per question, but it is bimodal: most answers take 1 to
2 s and about a third take around 58 s. The slow ones are the client waiting for the Groq
tokens-per-minute window to reopen (8K TPM on the free tier), not the model. Treat the average as a
measure of the free-tier limit, not of the system.

### Phase 2 results: reranking helped, hybrid search did not

Same 21 questions, same 147 documents, same 5 chunks per question; only the retriever changed.

| Retriever | hit@5 | full_recall@5 | MRR | Questions whose first right document moved |
|---|---|---|---|---|
| Vector (baseline) | 1.00 | 0.88 | 0.97 | none |
| Hybrid (vector + full-text, RRF) | 1.00 | 0.88 | **0.73** | 7 of 17 moved down: q02 (1 to 3), q05 (1 to 2), q07 (1 to 4), q11 (1 to 3), q12 (1 to 2), q15 (1 to 2), q17 (1 to 2) |
| Rerank (hybrid top 20, Voyage `rerank-2.5-lite`) | 1.00 | **0.94** | 0.94 | 1: q11 (1 to 2) |

- **Hybrid search made ranking worse and recovered nothing.** It found the same documents as
  the vector search (hit and full recall unchanged; q14 and q16 still miss) but pushed the right
  one down on 7 questions. My hypothesis, not yet tested: the keyword half ORs every word of the
  question, Postgres ranks without IDF, and so generic words like "use", "app" or "FastAPI" put
  broad pages (`versions.md`, `fastapi-cli.md`, `bigger-applications.md`) at the top of the keyword
  list. Reciprocal Rank Fusion gives that list the same weight as the vector list, so the noise
  outranks a correct vector hit. This is a flaw of this implementation, not proof that hybrid
  search cannot help; questions with exact identifiers are the case it is meant for, and this
  question set has few.
- **Reranking fixed q14 and cost one rank on q11.** q14 now returns both documents it needs
  (`testing-dependencies.md` and `testing.md`), so full recall rose from 15 to 16 of 17. On q11 the
  reranker put `reference/staticfiles.md` above `tutorial/static-files.md`; the right document is
  still in the top 5, so hit@5 is unaffected, but MRR dropped from 0.97 to 0.94. The reranker also
  recovered from the noisy hybrid ordering, because it rescored all 20 candidates rather than
  trusting their order.
- **q16 is still not fixed by either, and the reason is the reranker's choice, not the search.**
  `background-tasks.md` is among the 20 hybrid candidates (checked offline, see below), but the
  reranker scored five chunks from `advanced-dependencies.md` and `dependencies-with-yield.md`
  above it. One question about two topics gets answered with the more dominant topic. That is what
  the agent loop's decompose step is for: search each half separately.
- **Reranking is slow on the free tier.** The retrieval stage took about 60 s per question (21
  questions in about 20 minutes), because the local limiter waits between rerank requests; the model
  call itself is quick. The likely cause is the token estimate (3 characters per token, deliberately
  high) making each 20-candidate request look large against the assumed 10K tokens per minute. I
  have not confirmed Voyage's actual rerank limits.
- **Not measured:** answer quality with the new retrievers (`make eval-gen MODE=rerank`).

**Offline experiments (no API calls).** To tune the hybrid search without spending quota,
`experiments/phase2_offline.py` rebuilds the corpus in a throwaway Postgres using the real Voyage
vectors from the disk cache. It reproduces the saved runs exactly (vector 0.882 / 0.971, hybrid
0.882 / 0.730), so its other rows can be trusted. Reranking cannot be tested this way.

| Variant (top 5 chunks) | full_recall@5 | MRR |
|---|---|---|
| Vector | 0.88 | **0.97** |
| Hybrid, keyword list weight 1.0 (shipped) | 0.88 | 0.73 |
| Hybrid, keyword weight 0.5 / 0.15 | 0.88 | 0.79 / 0.84 |
| Hybrid, drop words found in more than 10% of chunks, weight 0.5 | 0.94 | 0.84 |
| Hybrid, drop words found in more than 2% of chunks | 0.82 | 0.93 |
| Hybrid, keywords only from code-like identifiers | 0.82 | 0.90 to 0.93 |

| Candidate pool for the reranker (top 20 chunks) | Both needed documents present |
|---|---|
| Vector | 16 of 17 (misses q16) |
| Hybrid, keyword weight 1.0 (shipped) | **17 of 17** |
| Hybrid, keyword weight 0.5 or lower | 16 of 17 |
| Hybrid at depth 10 or 15 | 16 of 17 |

What this decides: no hybrid variant beats plain vector search on ranking for these questions
(vector is already at 0.97 MRR), and the variants that raise MRR do it by dropping keywords, which
costs full recall. But the shipped hybrid is the best *candidate generator*: its top 20 contains
every needed document, which vector search alone does not. So hybrid stays as the first stage of
`rerank` and is not recommended as a standalone mode. The caveat is the same as everywhere here:
21 questions, so these are directions, not tunings I would defend to the second decimal. I did not
add the document-frequency filter: it needs corpus statistics that get expensive on a large table.
- **Reading the size of these effects.** One question is 0.06 of full recall, so +0.06 from
  reranking is a single question. Hybrid's MRR drop (0.97 to 0.73) is larger and consistent across
  7 questions, so I trust that one more than the recall gain.

**How much to trust this.** 21 questions is a smoke test, not a benchmark, and a difference of one
question is noise. Moving from 40 to 147 documents cost one question of full recall (0.94 to 0.88),
which is the effect you expect when distractors are added and the reason Phase 2 was built. hit@5 and MRR did not move, so the first right document is still found and
ranked first; what gets lost is the second document of a multi-part question.

### Per-document cap: fixed q16's retrieval, not its answer

Setting `RERANK_MAX_PER_SOURCE=2` (off by default) makes the reranker score all 20 candidates and
lets no document fill more than 2 of the 5 slots. Run: `rerank-1791828982.json`, compared row by row
with `rerank-1791820713.json`.

| | Rerank | Rerank + cap of 2 |
|---|---|---|
| hit@5 | 1.00 (17/17) | 1.00 (17/17) |
| full_recall@5 | 0.94 (16/17) | **1.00 (17/17)** |
| MRR | 0.94 | 0.94 |

- **q16 now retrieves `background-tasks.md`.** That is the whole gain: one question.
- **No rank changed for any question** (MRR is identical), and no expected source was lost. The
  only differences are in the tail: where a document had held 3 or more slots, the freed slots now go
  to the next-best other documents. Eleven questions differ this way (q16 is the fix; for example q01 went from 1 to 3
  distinct documents in the top 5).
- **What this does not show.** One question moved, and I picked the cap of 2 from the q16 failure
  before running anything, so this is a result on the question that motivated the change and not
  independent evidence. The tail chunks are extra material for the answer step: they could add
  useful context or noise, and generation has not been run with them. Unanswerable questions
  now see slightly different passages too (q19's fifth source changed), so abstention has to be
  rechecked.

**End to end (21 questions, with generation).** Without and with the cap the answer-level metrics are
identical: `cited_ok` 0.88 (15/17), `false_abstain` 0.06 (1/17), abstention on the unanswerable
questions 4/4. Tokens per question rose about 4% (roughly 1,570 to 1,640). The two failures are the
same in both runs:

- **q16:** with the cap, `background-tasks.md` is in the context (full recall true), but the answer
  still cites only the `yield` documents. Retrieval was fixed; the model did not use the extra
  document. The answer step is now the limit for this question, not search.
- **q17:** still a false abstain in both runs (the partial question that flips between runs).

So the cap is a retrieval improvement (one question, 16/17 to 17/17 full recall) that costs a little
and does not hurt answers or abstention, and it does not change the answer-level scores.

### Phase 3 first results: the loop works, but three questions cannot prove it helps

Run: `agent-rerank-gen-1791822263.json`, the rerank retriever plus the agent loop, on the three
known failures only (q14, q16, q17), to stay inside Groq's daily token cap. It is not comparable
to the 21-question rows above, and there is no unanswerable question in it, so ignore its abstain
figure.

| Question | Vector + generation (before) | Agent + rerank (after) | Rounds, stop reason | Tokens |
|---|---|---|---|---|
| q14 (multi: override a dependency + create the test client) | full recall missed, answer cited a right doc | both documents, cited | 1, sufficient | 2,642 (vector: 1,461) |
| q16 (multi: yield dependency + background tasks) | missed `background-tasks.md` | still missed `background-tasks.md` | 1, sufficient (the grader was wrong) | 4,348 (vector: 3,183) |
| q17 (partial: Celery) | **false abstain** | answered the part the docs cover, "Not found:" for the rest, cited | 3, max_rounds | 5,014 (vector: 1,813) |

Average 4,001 tokens per question against about 1,600 for Phase 1, and 1.67 rounds.

- **q14 is a retrieval win, not an agent win.** Reranking fixed it in Phase 2; the loop only
  confirmed the evidence was enough (one extra grading call).
- **q16 is the loop's failure, and the reason is informative.** The grader judged the evidence
  sufficient because the passages from `dependencies-with-yield.md` do discuss background tasks
  (the version note about when exit code runs). It could not know that a separate document is the
  authority on background tasks. The decompose step never ran, so it never got the chance to fix
  the case it was built for.
- **q17 went from a false abstain to a useful partial answer, but the loop was wasteful.** The
  grader asked for "Celery integration", "Celery worker setup", then "Celery FastAPI integration
  setup": rewordings of one query. The docs simply do not cover Celery setup, so the extra rounds
  found one and then two new chunks (the last two from `tutorial/index.md` and `deployment/docker.md`,
  unrelated) and cost about 3,300 more tokens and 3 minutes. I cannot say how much of the better answer
  comes from the loop and how much from the larger context (8 passages instead of 5); that would
  need a rerank-plus-generation run without the loop.
- **Not shown:** whether the loop hurts questions that were already fine. That needs the full
  21-question run.

#### Second run, after two changes to the loop

Run: `agent-rerank-gen-1791825324.json`. Changes: (1) a follow-up round must bring a document no
earlier round returned, otherwise the loop stops; (2) the grader must list the parts of the question
and the passage covering each, and the parser overrides "sufficient" when a part has none.

| Question | First run | Second run | Rounds (first, second) | Tokens (first, second) |
|---|---|---|---|---|
| q14 | both documents, cited | same answer (cached) | 1, 1 | 2,642 / 2,853 |
| q16 | missed `background-tasks.md` | same answer (cached), still missed | 1, 1 | 4,348 / 4,571 |
| q17 | partial answer with "Not found:", cited | **`INSUFFICIENT_CONTEXT` (false abstain)** | 3, 2 | 5,014 / 3,113 |

Tokens per question fell from 4,001 to 3,512 and the average from 1.67 to 1.33 rounds, but
`cited_ok` fell from 0.67 to 0.33 and `false_abstain` rose from 0 to 0.33. On these three questions
the changes bought nothing in quality.

- **Change 1 did what it was designed to do on cost.** q17 stopped after round 2 because the
  follow-up query returned one new chunk from a document already in hand (`new_sources: 0`), saving
  a round and about 1,900 tokens.
- **q17 is unstable, so its earlier improvement was not a result.** Both runs had
  `background-tasks.md`, the document with the Celery paragraph, in the context from round 1. The
  first run (8 passages) answered with "Not found:" for the setup steps; this one (6 passages, a
  different order) refused. The Phase 1 baseline (5 passages) also refused. Whether that is
  prompt sensitivity or sampling noise I cannot tell from three samples, but it means the loop is
  not what decides q17: the generation step is. I should not have credited the loop for it.
- **Change 2 had no effect on q16, and the trace now shows why (re-run from cache, no tokens
  spent).** The grader followed the format and split the question into its two *clauses*: "can a
  dependency with yield be used together with background tasks" (passage 1) and "when does its exit
  code run" (passage 2). Both passages are about `yield` dependencies, so every part had a passage and
  nothing was overridden. It treated "yield" and "background tasks" as one topic. So the failure is
  one of judgement about what the question needs, not of format, and more prompt wording would be
  tuning to this one question.
- **q14 and q16 answers were served from the cache** (same retrieval, same passages), so the
  second run only really exercised q17, and the grader on all three.

### Known failure modes

Status as of the last saved runs. "Measured" means a file in `evals/results/` backs it.

- **Multi-hop questions get only half the evidence (q16; q14 until reranking).**
  q16 needs `dependencies-with-yield.md` and `background-tasks.md`. Vector, hybrid and rerank all
  return chunks from `advanced-dependencies.md` and `dependencies-with-yield.md` and not the second
  document, even though `background-tasks.md` is among the 20 hybrid candidates (checked offline).
  Reasons: one query vector favors the dominant half of a two-part question, and top-k chunks
  cluster in a few documents. q14 (`testing-dependencies.md` plus `testing.md`) was fixed by
  reranking (measured). q16 was **not** fixed by the agent loop (measured, one run): the grader
  called the evidence sufficient, so decompose never ran. A cap of 2 chunks per document fixed it at
  the retrieval stage (retrieval only; see the per-document cap section).
- **The grader can be fooled by a passing mention (q16, measured).** The passages it saw talk about
  background tasks in the context of `yield`, so it judged that part covered. An LLM judging
  sufficiency from passages cannot know which document is the authority on a topic. The part-by-part
  check added after the first run did not change the verdict on q16: the grader split the question
  into two clauses, both answered by `yield` passages, and never treated "background tasks" as a
  topic that needs its own document.
- **Partial questions flip between runs (q17, measured three times).** "Set up Celery workers with
  FastAPI" is only touched on in the docs. Plain generation refused (false abstain). Rerank plus the
  loop answered with "Not found:" for the setup steps in one run and refused in the next, with the
  same relevant document in the context both times. So this is generation behaviour on a partial
  question, not something the loop controls. The loop's real cost here is separate: it cannot tell
  "the docs do not have this" from "I have not found it yet", and asked for rewordings of one query
  (3 rounds, about 5,000 tokens) until the new-document stop rule cut it to 2 rounds.
- **Unanswerable questions are refused correctly without the loop (q18 to q21, 4/4).** Retrieval
  returns something for every query (often `index.md`, `features.md`), so the refusal comes from
  the LLM saying `INSUFFICIENT_CONTEXT`. With four questions this says the prompt works, not that
  abstention is solved. **Not measured with the loop**: the agent run covered only q14, q16 and
  q17, so whether the loop keeps refusing, or burns rounds looking for something that is not there,
  is unknown.
- **Citation misses were mostly a format problem, fixed (q07, q10).** A lenient citation parser
  (accepts `【5】`, `[2, 5]`) and a tighter prompt lifted `cited_ok` from 0.76 to 0.88 with retrieval
  unchanged. What remains is q16 (it cites only the one of its two documents that was retrieved).

## Limits and what I would do next

### What the numbers can and cannot support

- **The eval is small.** 21 questions, 17 answerable: one question moves a rate by 0.06. Reranking's
  gain is one question (q14); the agent loop has been run on three. None of this is statistically
  significant, and the agent changes were designed after looking at those same three questions, so a
  good result on them proves little. A held-out set is needed before tuning further.
- **No control for the loop.** The agent run used rerank plus a larger context (up to 9 passages).
  There is no rerank-plus-generation run without the loop, so "the loop helped" and "more context
  helped" cannot be told apart.
- **The grader has been tested on one model.** OpenAI gpt-oss-20b on Groq. All four grader replies in
  the first run parsed as JSON, but four is not a reliability measure. The fail-open path (answer with
  what was found) exists for when it does not.
- **Cost and latency are inflated by the free tier.** The loop used about 4,000 tokens per question
  against about 1,600 without it. The 60 to 183 s per question is mostly the rate limiter waiting,
  so it says little about real latency. Groq's daily cap of about 200K tokens is not modelled by the
  quota governor; a full 21-question agent run is estimated at 85K or more.
- **Not verified against real services:** Voyage rerank rate limits (assumed 3 requests per minute),
  and the Gemini adapter (tested against mocked responses only).
- Free-tier APIs: rate limits apply, and providers may use free-tier data to improve their products
  (public docs only here).
- Ingest throughput on the free tier is limited by the embedding quota (3 requests per minute), not
  by the pipeline. `make load-test` measures the pipeline alone: about 70 docs/s per process with a
  fake embedder (details in [Ingestion](docs/ingestion.md#pipeline-throughput-load-test)). That
  excludes real network behavior and a 395k-chunk table, which are not measured.

### What I would do next, in order

1. **Make the answer step use the evidence it is given (q16).** The cap put `background-tasks.md` in
   the context and the model still cited only the `yield` documents. Passages already carry their
   source and heading, so the likely fix is a prompt instruction to answer each part of a multi-part
   question from the passages that cover it. A prompt change to measure on the same 21 questions;
   the agent loop is not needed for this case any more.
2. **Run the full agent evaluation.** The no-loop control now exists (`rerank-gen-*` above). Run
   `RERANK_MAX_PER_SOURCE=2 make eval-agent MODE=rerank` on all 21 questions, after the daily quota resets. Report the loop's
   effect on the questions that already passed too: the real risk is that it costs tokens and
   sometimes makes a good answer worse.
3. **Make it a conversation.** Today every `/ask` call is independent: it answers any question, and
   several parts in one message, but a follow-up such as "and how do I test it?" has no context. The
   design: store turns per session in Postgres, rewrite each follow-up into a standalone question
   before retrieval (one extra LLM call, skipped on the first turn), pass a window of recent turns to
   the answer step, and add a persona, style and guardrail block to the system prompt (scope to the
   docs, treat passage text as data, no personal data). It needs its own multi-turn eval (pronoun
   follow-ups, topic switches, unanswerable follow-ups) and a re-run of the 21 questions, because the
   prompt change can regress them. Not built, not measured.
4. **Grow the eval set before tuning anything else.** Roughly 40 questions with more multi-hop and
   more unanswerable ones, written before looking at results, with part of them held out. Without
   this, every later improvement is a guess about three questions.
5. **If the cap hurts answers or abstention,** the other LLM-free option is to always decompose
   questions that contain two clauses (a cheap planner call, or a rule on "and"), at the cost of
   more tokens on every multi-part question. The grader-prompt route is closed for now: it followed
   its instructions and still judged the evidence sufficient.
6. **Make the loop cheaper and faster.** Run sub-queries concurrently (they run one after another
   today), and use plain hybrid for follow-up rounds, reserving the reranker for the first one. Both
   change how it behaves, so each needs its own measurement.
7. **Unfinished housekeeping from earlier phases:** batch several documents per embedding request
   (up to about 3x more throughput on the free tier, needs a higher `EMBED_TPM`), make Voyage and Groq
   the built-in provider defaults, and confirm the real Voyage rerank limits.
8. Chunks under 200 chars are 12% of the corpus; merging tiny sections into neighbors may help.
   Contextual retrieval (LLM-written context prepended to each chunk) is a likely further gain.
9. **Production:** tracing (Langfuse or LangSmith; LangSmith is installed but off), drift monitoring
   on the eval set in CI, per-query cost budgets, and a real queue.
