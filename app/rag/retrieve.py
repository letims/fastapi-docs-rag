"""Retrieval: vector search, keyword search, Reciprocal Rank Fusion, reranking and the per-document
cap. `search(mode, ...)` is the single entry point used by the API, the agent and the evals."""

import logging
import re
from dataclasses import dataclass, replace

from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.db import vec_literal
from app.providers.base import Embedder, Reranker

log = logging.getLogger(__name__)

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
    fused = rrf_fuse([vec, kw])
    log.debug(
        "hybrid: vector %d + keyword %d candidates, %d found by both, %d after fusion, keeping %d",
        len(vec),
        len(kw),
        len({h.chunk_id for h in vec} & {h.chunk_id for h in kw}),
        len(fused),
        k,
    )
    return fused[:k]


def cap_per_source(ranked: list[Hit], k: int, cap: int) -> list[Hit]:
    """Take the best `k` hits while letting no document fill more than `cap` of the slots.

    Top-k chunks cluster in a few documents, so a question about two topics can be answered from
    only the dominant one. Hits over the cap are skipped in favour of the next-best hit from another
    document; if that leaves fewer than `k`, the skipped hits fill the remaining slots in order.
    `cap <= 0` turns the rule off.
    """
    if cap <= 0:
        return ranked[:k]
    kept: list[Hit] = []
    skipped: list[Hit] = []
    per_source: dict[str, int] = {}
    for h in ranked:
        if len(kept) == k:
            break
        if per_source.get(h.source, 0) < cap:
            per_source[h.source] = per_source.get(h.source, 0) + 1
            kept.append(h)
        else:
            skipped.append(h)
    return kept + skipped[: k - len(kept)]


async def rerank_search(
    pool: AsyncConnectionPool,
    embedder: Embedder,
    reranker: Reranker,
    question: str,
    k: int = 5,
    candidates: int = 20,
    max_per_source: int | None = None,
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
        max_per_source: At most this many of the `k` hits from one document (0 = no limit,
            the default from `RERANK_MAX_PER_SOURCE`). When set, the reranker scores every
            candidate instead of only the top `k`, so lower-ranked documents can take the slots.

    Returns:
        Up to `k` hits, best first, `score` = the reranker's relevance score.
    """
    pool_hits = await hybrid_search(pool, embedder, question, candidates, candidates)
    if not pool_hits:
        return []
    docs = [f"{h.heading}\n{h.content}" for h in pool_hits]
    cap = get_settings().rerank_max_per_source if max_per_source is None else max_per_source
    ranked = await reranker.rerank(question, docs, top_k=len(docs) if cap > 0 else k)
    scored = [replace(pool_hits[i], score=float(s)) for i, s in ranked]
    final = cap_per_source(scored, k, cap)
    if cap > 0:
        dropped = [h.source for h in scored[:k] if h not in final]
        if dropped:
            log.info(
                "rerank cap of %d per document replaced %d of the top %d: dropped chunks from %s",
                cap,
                len(dropped),
                k,
                sorted(set(dropped)),
            )
    log.debug("rerank: scored %d candidates, keeping %d", len(docs), len(final))
    return final


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
        hits = await vector_search(pool, embedder, question, k)
    elif mode == "hybrid":
        hits = await hybrid_search(pool, embedder, question, k, candidates)
    elif mode == "rerank":
        if reranker is None:
            raise ValueError("mode 'rerank' needs a reranker")
        hits = await rerank_search(pool, embedder, reranker, question, k, candidates)
    else:
        raise ValueError(f"unknown retrieval mode {mode!r} (choose from {', '.join(MODES)})")
    sources = list(dict.fromkeys(h.source for h in hits))
    log.info(
        "search %s k=%d %r: %d chunks from %d documents %s%s",
        mode,
        k,
        question[:60],
        len(hits),
        len(sources),
        sources[:4],
        "" if hits else " (nothing found)",
    )
    return hits
