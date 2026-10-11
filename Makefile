.PHONY: up down corpus init enqueue worker api test test-fast cov-open lint eval
up:        ; docker compose up -d --wait
down:      ; docker compose down
corpus:    ; ./scripts/fetch_corpus.sh
init:      ; uv run python -m app.db
enqueue:   ; uv run python -m scripts.enqueue_corpus corpus/fastapi
worker:    ; uv run python -m app.worker --drain
api:       ; uv run uvicorn app.main:app --reload
# Full suite + coverage. FAILS if total coverage (line+branch) < 90% (see pyproject.toml).
# Writes the HTML report to htmlcov/index.html. DB: RAGNAROK_TEST_DB if set, else pgserver.
test:      ; uv run pytest -q --cov --cov-report=term-missing --cov-report=html
test-fast: ; uv run pytest -q -x
cov-open:  ; open htmlcov/index.html
lint:      ; uv run ruff check . && uv run ruff format --check .
eval:      ; uv run python -m evals.run_eval --mode vector
