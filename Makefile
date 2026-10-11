.PHONY: up down corpus init enqueue worker api test lint eval
up:        ; docker compose up -d --wait
down:      ; docker compose down
corpus:    ; ./scripts/fetch_corpus.sh
init:      ; uv run python -m app.db
enqueue:   ; uv run python -m scripts.enqueue_corpus corpus/fastapi
worker:    ; uv run python -m app.worker --drain
api:       ; uv run uvicorn app.main:app --reload
test:      ; uv run pytest -q
lint:      ; uv run ruff check . && uv run ruff format --check .
eval:      ; uv run python -m evals.run_eval --mode vector
