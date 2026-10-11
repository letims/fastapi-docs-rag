# Makefile commands

Listed in the order you would normally use them. "Needs" says what must have happened first.

## 1. Set up

| Command | What it does | When to use it | Needs |
|---|---|---|---|
| `make up` | Starts Postgres 16 + pgvector in Docker (host port 5433) and waits until it is healthy | Once per work session, before anything that touches the database | Docker running |
| `make init` | Creates the tables (documents, chunks, job queue, API usage counter) | Once per fresh database, including after wiping the Docker volume | `make up` |
| `make corpus` | Downloads the FastAPI docs as markdown into `./corpus` (gitignored) | Once, or to refresh the docs | Internet access |
| `make down` | Stops the database container. Data stays in the volume | End of the day. To wipe the data, use `docker compose down -v` | |

## 2. Load the documents

| Command | What it does | When to use it | Needs |
|---|---|---|---|
| `make enqueue-dev` | Queues a 40-doc dev subset: every doc the eval questions point at, plus distractors | First runs, and whenever the API quota is tight | `make init`, `make corpus` |
| `make enqueue` | Queues the whole corpus (147 docs after exclusions) | Once you are ready to embed everything. Use it **instead of** `enqueue-dev`, or after it: finished docs are skipped | `make init`, `make corpus` |
| `make worker` | Starts workers that claim queued jobs, chunk and embed them, and write the results. Exits when the queue is empty, then builds the vector index and prints throughput | After every `enqueue*`. Safe to stop and rerun | `make up`, a queue filled by `enqueue*`, embedding keys in `.env` |
| `make quota` | Shows today's API usage per provider and the time until the reset | Before a big run, or after a "quota exhausted" pause | `make up`, `make init` |

## 3. Use it

| Command | What it does | When to use it | Needs |
|---|---|---|---|
| `make api` | Runs the FastAPI app on http://localhost:8000 (`/docs` for the UI, `POST /ask` for questions) | To ask questions by hand | A finished `make worker` (otherwise there is nothing to retrieve), plus embedding and chat keys |
| `make eval` | Runs the 21 eval questions through retrieval and prints hit rate and MRR, with a `[i/N]` progress line and time left. Saves the run in `evals/results/` | After any change to retrieval, so every change has a measured effect | A finished `make worker`, embedding key. It does not need `make api` |
| `make eval-gen` | Same, plus answers, citations and abstention (calls the LLM) | To measure answer quality, abstention and tokens per query | Same as `make eval`, plus the chat key (Groq limits make it slow) |

## 4. Check the code

| Command | What it does | When to use it | Needs |
|---|---|---|---|
| `make test` | Runs the full test suite with coverage and **fails below 90%**. Writes the HTML report to `htmlcov/` | Before every commit | Nothing: it starts its own throwaway Postgres (`pgserver`), or uses `RAGNAROK_TEST_DB` if set. No API keys |
| `make test-fast` | Runs the tests without coverage and stops at the first failure | While developing, for a quick loop | Same as `make test` |
| `make cov-open` | Opens the HTML coverage report in the browser (macOS `open`) | To see which lines are untested | A previous `make test` |
| `make lint` | Checks style and formatting with ruff (changes nothing) | Before pushing | Nothing |
| `make fmt` | Auto-fixes what ruff can fix and reformats the code | When `make lint` or the pre-commit hook complains | Nothing |
| `make check` | `make lint` then `make test`: exactly what the pre-commit hook runs | To confirm everything is green before committing or pushing | Same as `make test` |
| `make install-hooks` | Points git at `.githooks/` so the pre-commit hook runs on this clone | Once per clone (already done on this machine) | A git repository |

## Before you commit

`.githooks/pre-commit` runs on every `git commit` and **blocks the commit** if any check fails:

1. **No secrets:** `.env` must not be staged, and no added line may look like an API key
   (Voyage `pa-…`, Groq `gsk_…`, Google `AIza…`, `sk-…`).
2. **Lint:** `ruff check` (errors, unused code, import order, common bugs).
3. **Format:** `ruff format --check` (fix with `make fmt`).
4. **Tests and coverage:** the full suite, failing under 90% coverage.

Commits that change only docs (no `.py`, config, SQL, shell or workflow files) skip steps 2-4, so
editing the README is instant. The same lint, format and test checks run again in CI on every push,
and CI also fails if `uv.lock` is out of date. Emergency bypass: `git commit --no-verify`
(CI will still catch it).

## Typical sequences

```bash
# First time
make up && make init && make corpus
make enqueue-dev && make worker      # small and fast
make api                             # or: make eval

# Later, with a bigger quota
make enqueue && make worker          # adds the rest; finished docs are skipped

# Every day
make up                              # then work; make down at the end
```
