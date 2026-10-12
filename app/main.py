from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import close_pool, init_schema, open_pool
from app.ingest import queue
from app.providers import get_embedder, get_llm, get_reranker
from app.quota import QuotaExhausted, seconds_until_reset
from app.rag.agent import run_agent
from app.rag.generate import generate_answer
from app.rag.retrieve import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown: open the DB pool and ensure the schema exists, close the pool on exit.

    The pool is stored on `app.state.pool` for the endpoints to use.
    """
    app.state.pool = await open_pool()
    await init_schema(app.state.pool)
    yield
    await close_pool()


app = FastAPI(title="fastapi-docs-rag", lifespan=lifespan)


@app.exception_handler(QuotaExhausted)
async def quota_exhausted_handler(_request, exc: QuotaExhausted):
    """Map an exhausted daily API budget to HTTP 429 with a Retry-After header (seconds to reset)."""
    retry_after = int(seconds_until_reset())
    return JSONResponse(
        status_code=429,
        content={"error": "quota_exhausted", "detail": str(exc), "retry_after_s": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


class DocIn(BaseModel):
    """One document submitted for ingestion."""

    source: str  # unique key (e.g. relative path); re-submitting the same source updates it
    title: str
    content: str  # markdown text


class AskIn(BaseModel):
    """Request body for `/ask`."""

    question: str = Field(min_length=3)
    k: int = Field(5, ge=1, le=20)  # chunks to retrieve and give the LLM as context
    agent: bool | None = None  # run the Phase 3 agent loop; None = the AGENT_DEFAULT setting


@app.get("/health")
async def health():
    """Liveness check. Does not touch the database.

    Returns:
        `{"ok": true}` whenever the process is serving requests.
    """
    return {"ok": True}


@app.get("/ingest/stats")
async def ingest_stats():
    """Report ingestion progress, for monitoring a bulk load.

    Returns:
        `{"jobs": {status: count}, "documents": int, "chunks": int}`, where status is one of
        pending / running / done / failed (only statuses that currently have jobs appear).
    """
    return await queue.stats(app.state.pool)


@app.post("/documents", status_code=202)
async def add_documents(docs: list[DocIn]):
    """Enqueue documents for ingestion; workers (python -m app.worker) do the embedding.

    This only writes job rows, so it returns fast (HTTP 202) and nothing is searchable until a
    worker has processed the job. A source that already has a pending or running job is skipped.

    Args:
        docs: Documents to ingest (source, title, markdown content).

    Returns:
        `{"received": <docs in the request>, "enqueued": <new jobs actually created>}`.
    """
    created = await queue.enqueue_many(
        app.state.pool, [(d.source, d.title, d.content) for d in docs]
    )
    return {"received": len(docs), "enqueued": created}


def _answer_payload(ans, hits) -> dict:
    """The parts of an /ask response shared by the plain and the agent path."""
    return {
        "answer": ans.text,
        "abstained": ans.abstained,
        "citations": [{"source": h.source, "heading": h.heading} for h in ans.citations],
        "retrieved": [
            {"source": h.source, "heading": h.heading, "score": round(h.score, 4)} for h in hits
        ],
    }


@app.post("/ask")
async def ask(body: AskIn):
    """Answer a question about the FastAPI docs with retrieval-augmented generation.

    Retrieves the top `k` chunks with the configured `RETRIEVAL_MODE` (vector, hybrid or rerank), and asks the LLM to
    answer only from those chunks. If the chunks do not contain the answer the LLM replies
    `INSUFFICIENT_CONTEXT` and `abstained` is true.

    Args:
        body: The question and how many chunks (`k`) to retrieve.

    Returns:
        A dict with:
          - answer: the LLM's text, with `[n]` citations pointing into the retrieved passages.
          - abstained: true if the model said the context was insufficient.
          - citations: source/heading of the passages actually cited in the answer.
          - retrieved: every retrieved passage (source, heading, score; the scale depends on the retrieval mode), in rank order.
          - usage: input/output token counts for the LLM call.
          - latency_s: LLM generation time in seconds (retrieval time is not included).
    """
    s = get_settings()
    reranker = get_reranker() if s.retrieval_mode == "rerank" else None
    if body.agent if body.agent is not None else s.agent_default:
        res = await run_agent(
            app.state.pool,
            get_embedder(),
            get_llm(),
            reranker,
            body.question,
            mode=s.retrieval_mode,
            k=body.k,
            candidates=s.rerank_candidates,
            max_rounds=s.agent_max_rounds,
            token_budget=s.agent_token_budget,
            per_query=s.agent_per_query,
            extra_context=s.agent_context_extra,
        )
        ans = res.answer
        return {
            **_answer_payload(ans, res.context),
            "usage": {"input_tokens": res.input_tokens, "output_tokens": res.output_tokens},
            "latency_s": round(res.latency_s, 2),
            "agent": {
                "rounds": res.rounds,
                "stop_reason": res.stop_reason,
                "queries": res.queries,
                "trace": res.trace,
            },
        }
    hits = await search(
        s.retrieval_mode,
        app.state.pool,
        get_embedder(),
        reranker,
        body.question,
        body.k,
        s.rerank_candidates,
    )
    ans = await generate_answer(get_llm(), body.question, hits)
    return {
        **_answer_payload(ans, hits),
        "usage": {"input_tokens": ans.input_tokens, "output_tokens": ans.output_tokens},
        "latency_s": round(ans.latency_s, 2),
    }
