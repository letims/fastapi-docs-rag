from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from app.db import vec_literal
from app.providers.base import Embedder


@dataclass
class Hit:
    """One retrieved chunk plus where it came from."""

    chunk_id: int
    source: str
    heading: str
    content: str
    score: float  # cosine similarity for vector search


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
