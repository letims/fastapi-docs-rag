.PHONY: up down corpus init enqueue enqueue-dev quota worker api test test-fast cov-open lint fmt check install-hooks validate-json eval eval-gen eval-agent eval-judge eval-compare eval-baseline load-test
up:        ; docker compose up -d --wait
down:      ; docker compose down
corpus:    ; ./scripts/fetch_corpus.sh
init:      ; uv run python -m app.db
enqueue:   ; uv run python -m scripts.enqueue_corpus corpus/fastapi
# Small dev corpus (all eval-expected docs + distractors): fits a tight API quota.
enqueue-dev: ; uv run python -m scripts.enqueue_corpus corpus/fastapi --subset 40
quota:     ; uv run python -m scripts.quota
worker:    ; uv run python -m app.worker --drain
api:       ; uv run uvicorn app.main:app --reload
# Full suite + coverage. FAILS if total coverage (line+branch) < 90% (see pyproject.toml).
# Writes the HTML report to htmlcov/index.html. DB: TEST_DATABASE_URL if set, else pgserver.
test:      ; uv run pytest -q --cov --cov-report=term-missing --cov-report=html
test-fast: ; uv run pytest -q -x
cov-open:  ; open htmlcov/index.html
lint:      ; uv run ruff check . && uv run ruff format --check .
# Auto-fix what ruff can fix and reformat. Run it when `make lint` or the pre-commit hook complains.
fmt:       ; uv run ruff check --fix . && uv run ruff format .
# JSON schema checks: Claude settings (SchemaStore, cached after first fetch), workflows (vendored
# schema, offline), eval runs + baselines (evals/schemas/). Syntax AND structure, not just "parses".
SETTINGS_SCHEMA ?= https://json.schemastore.org/claude-code-settings.json
validate-json:
	uv run check-jsonschema --schemafile $(SETTINGS_SCHEMA) .claude/settings.json
	uv run check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/*.yml
	uv run check-jsonschema --schemafile evals/schemas/eval-run.schema.json evals/baseline/*.json $(wildcard evals/results/*.json)
# Everything the pre-commit hook runs (lint + format check + tests with the coverage gate).
check:     ; $(MAKE) lint && $(MAKE) validate-json && $(MAKE) test
# Point git at .githooks/ so the pre-commit hook runs on this clone. Once per clone.
install-hooks: ; git config core.hooksPath .githooks
# MODE picks the retriever: vector (default), hybrid or rerank.  Example: make eval MODE=hybrid
MODE ?= vector
eval:      ; uv run python -m evals.run_eval --mode $(MODE)
# Also calls the LLM: answers, citations, abstention, tokens. Slow under Groq's free limits.
eval-gen:  ; uv run python -m evals.run_eval --mode $(MODE) --generate
# Phase 3 loop on top of MODE. Cheap first look: make eval-agent MODE=rerank ONLY=q14,q16,q17
eval-agent: ; uv run python -m evals.run_eval --mode $(MODE) --agent $(if $(ONLY),--only $(ONLY))
# Answers + LLM-judge groundedness (one extra LLM call per answer; spends Groq daily quota).
eval-judge: ; uv run python -m evals.run_eval --mode $(MODE) --judge
# Retrieval eval (no LLM), then diff against evals/baseline/$(MODE).json. Exit 1 on regression.
eval-compare: ; uv run python -m evals.run_eval --mode $(MODE) && uv run python -m evals.compare --latest $(MODE)
# Promote a saved run to the baseline for its tag:  make eval-baseline RUN=evals/results/rerank-<sha>-<ts>.json
eval-baseline: ; uv run python -m evals.compare --promote $(RUN)
# Pipeline-only throughput (fake embedder, throwaway Postgres, no API or quota). See docs/ingestion.md.
load-test: ; uv run python -m scripts.load_test --docs 1000 --concurrency 1,4,8,16
