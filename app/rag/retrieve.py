import re
from dataclasses import dataclass, replace

from psycopg_pool import AsyncConnectionPool

from app.db import vec_literal
from app.providers.base import Embedder, Reranker

RRF_K = 60  # the constant from the original RRF paper; larger flattens the rank weighting
MODES = ("vector", "hybrid", "rerank")


@dataclass
class Hit:
    """One retrieved chunk plus where it came from."""

    chunk_id: int
    source: str
    heading: str
    content: str
    score: float  # cosine similarity (vector), RRF score (hybrid) or relevance score (rerank)


async def vector_search(
    pool: AsyncConnectionPool, embedder: Embedder, question: str, k: int = 5
) -> list[Hit]:
    """Return the `k` chunks whose embeddings are closest to the question's.

    Embeds the question (as a query), then lets Postgres rank chunks by cosine distance using
    the HNSW index. The DB connection is only taken after the embedding call has finished.

    Args:
        pool: Open connection pool.
        embedder: Provider used to embed the question; must match the one used at ingestion.
        question: The user's question, in natural language.
        k: Number of chunks to return.

    Returns:
        Up to `k` hits, best first. `score` is cosine similarity (1.0 = identical direction).
    """
    [qvec] = await embedder.embed([question], kind="query")
    lit = vec_literal(qvec)
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT c.id, d.source, c.heading, c.content,
                      1 - (c.embedding <=> %s::vector) AS score
                 FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s""",
            (lit, lit, k),
        )
        rows = await cur.fetchall()
    return [Hit(r["id"], r["source"], r["heading"], r["content"], float(r["score"])) for r in rows]


def keyword_query(question: str) -> str:
    """Turn a question into an OR-style `tsquery` string: `word | word | ...`.

    Only letters and digits survive (underscores split, so `response_model` becomes two words,
    as the Postgres parser indexes it), which also makes the string safe to hand to `to_tsquery`.
    Stop words are dropped by Postgres itself. OR, not AND: a natural-language question rarely has
    every word in one chunk, and ranking decides what comes first.

    Returns:
        The query string, or "" if the question has no usable word.
    """
    words = dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 1)
    return " | ".join(words)


async def keyword_search(pool: AsyncConnectionPool, question: str, n: int = 20) -> list[Hit]:
    """Full-text search over chunk headings and text (Postgres `tsvector`, GIN index).

    Finds exact words and identifiers that embeddings blur, such as `HTTPException` or
    `response_model_exclude_unset`. Ranked with `ts_rank_cd`. Postgres has no IDF, so very common
    words weigh as much as rare ones; fusing with the vector ranking is what compensates.

    Args:
        pool: Open connection pool.
        question: The user's question.
        n: Maximum number of hits.

    Returns:
        Up to `n` hits, best first; `score` is the `ts_rank_cd` value. Empty if nothing matches.
    """
    q = keyword_query(question)
    if not q:
        return []
    async with pool.connection() as conn:
        cur = await conn.execute(
            """SELECT c.id, d.source, c.heading, c.content, ts_rank_cd(c.tsv, query) AS score
                 FROM chunks c JOIN documents d ON d.id = c.document_id,
                      to_tsquery('english', %s) query
                WHERE c.tsv @@ query
                ORDER BY score DESC, c.id
                LIMIT %s""",
            (q, n),
        )
        rows = await cur.fetchall()
    return [Hit(r["id"], r["source"], r["heading"], r["content"], float(r["score"])) for r in rows]


def rrf_fuse(rankings: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: merge several ranked lists into one.

    Each chunk scores the sum of `1 / (k + rank)` over the lists it appears in (rank starts at 1),
    so a chunk that two retrievers both like beats one that only a single retriever ranks first.
    Only ranks are used, never the raw scores, which is why cosine similarity and `ts_rank_cd`
    (different scales) can be combined without calibration.

    Args:
        rankings: Ranked hit lists, each best first.
        k: Damping constant.

    Returns:
        One list of unique chunks, best first, with `score` set to the fused RRF score. Ties keep
        the order in which the chunks were first seen.
    """
    scores: dict[int, float] = {}
    first: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking, 1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            first.setdefault(h.chunk_id, h)
    order = sorted(scores, key=lambda cid: -scores[cid])  # sorted() is stable: ties keep first-seen
    return [replace(first[cid], score=scores[cid]) for cid in order]


async def hybrid_search(
    pool: AsyncConnectionPool, embedder: Embedder, question: str, k: int = 5, candidates: int = 20
) -> list[Hit]:
    """Vector search and keyword search, fused with Reciprocal Rank Fusion.

    Each retriever contributes its top `candidates`; the fused list is cut to `k`.

    Args:
        pool: Open connection pool.
        embedder: Provider used to embed the question.
        question: The user's question.
        k: Number of chunks to return.
        candidates: How deep each retriever looks before fusing.

    Returns:
        Up to `k` hits, best first, `score` = RRF score.
    """
    vec = await vector_search(pool, embedder, question, candidates)
    kw = await keyword_search(pool, question, candidates)
    return rrf_fuse([vec, kw])[:k]


async def rerank_search(
    pool: AsyncConnectionPool,
    embedder: Embedder,
    reranker: Reranker,
    question: str,
    k: int = 5,
    candidates: int = 20,
) -> list[Hit]:
    """Hybrid search for `candidates` chunks, then a cross-encoder reorders them.

    The first stage is cheap and recall-oriented; the reranker reads the question and each
    candidate together, which is slower but much better at telling a passage that answers the
    question from one that merely shares its topic.

    Args:
        pool: Open connection pool.
        embedder: Provider used to embed the question.
        reranker: Cross-encoder used to rescore the candidates.
        question: The user's question.
        k: Number of chunks to return.
        candidates: How many first-stage hits the reranker sees.

    Returns:
        Up to `k` hits, best first, `score` = the reranker's relevance score.
    """
    pool_hits = await hybrid_search(pool, embedder, question, candidates, candidates)
    if not pool_hits:
        return []
    docs = [f"{h.heading}\n{h.content}" for h in pool_hits]
    ranked = await reranker.rerank(question, docs, top_k=k)
    return [replace(pool_hits[i], score=float(s)) for i, s in ranked]


async def search(
    mode: str,
    pool: AsyncConnectionPool,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    k: int = 5,
    candidates: int = 20,
) -> list[Hit]:
    """Run the retriever named by `mode`: "vector", "hybrid" or "rerank".

    Raises:
        ValueError: Unknown mode, or "rerank" without a reranker.
    """
    if mode == "vector":
        return await vector_search(pool, embedder, question, k)
    if mode == "hybrid":
        return await hybrid_search(pool, embedder, question, k, candidates)
    if mode == "rerank":
        if reranker is None:
            raise ValueError("mode 'rerank' needs a reranker")
        return await rerank_search(pool, embedder, reranker, question, k, candidates)
    raise ValueError(f"unknown retrieval mode {mode!r} (choose from {', '.join(MODES)})")
