"""Ingest one document, idempotently.

Idempotency layers:
  1. document content hash unchanged  -> skip entirely (no embedding calls)
  2. chunk content hash unchanged     -> reuse the stored embedding, only embed new chunks
  3. write is one transaction (upsert doc, replace chunks) -> a crash never leaves half a doc
Re-running the whole corpus is therefore cheap and safe."""

import logging
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from app.db import vec_literal
from app.ingest.chunker import chunk_markdown, doc_hash
from app.providers.base import Embedder

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """What `ingest_document` did, for worker counters and logging.

    `chunks_embedded` is how many chunks needed a (paid) embedding call; `chunks_reused` is how
    many reused a stored embedding because their hash was unchanged.
    """

    status: str  # "skipped" | "ingested"
    chunks_total: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0


async def ingest_document(
    pool: AsyncConnectionPool,
    embedder: Embedder,
    *,
    source: str,
    title: str,
    content: str,
    batch_size: int = 64,
    max_chars: int = 1800,
) -> IngestResult:
    """Chunk, embed and store one document, skipping all work that is already done.

    Steps: skip if the document hash is unchanged; otherwise chunk it, reuse stored embeddings
    for chunks whose hash matches, embed only the new ones in batches (no DB connection held
    during these network calls), then replace the document's chunks in a single transaction.

    Args:
        pool: Open connection pool.
        embedder: Provider used to embed new chunks.
        source: Unique key of the document (e.g. relative file path).
        title: Document title, prefixed to every chunk's heading.
        content: Full markdown text.
        batch_size: Chunks per embedding API call.
        max_chars: Chunk size limit passed to the chunker.

    Returns:
        An `IngestResult` with status "skipped" or "ingested" and chunk counts.

    Raises:
        RuntimeError: If the embedder returns a different number of vectors than texts.
    """
    new_doc_hash = doc_hash(content)

    async with pool.connection() as conn:
        row = await (
            await conn.execute("SELECT id, content_hash FROM documents WHERE source=%s", (source,))
        ).fetchone()
        if row and row["content_hash"] == new_doc_hash:
            return IngestResult("skipped")
        existing: dict[str, str] = {}
        if row:
            cur = await conn.execute(
                "SELECT content_hash, embedding::text AS emb FROM chunks WHERE document_id=%s",
                (row["id"],),
            )
            existing = {r["content_hash"]: r["emb"] for r in await cur.fetchall()}

    chunks = chunk_markdown(content, title, max_chars)
    embeddings: dict[str, str] = {}  # chunk hash -> pgvector literal
    missing = []
    for c in chunks:
        if c.content_hash in existing:
            embeddings[c.content_hash] = existing[c.content_hash]
        elif c.content_hash not in embeddings:
            embeddings[c.content_hash] = ""  # placeholder, dedupes identical chunks
            missing.append(c)

    # Network calls happen with no DB connection held.
    for i in range(0, len(missing), batch_size):
        batch = missing[i : i + batch_size]
        vectors = await embedder.embed([c.embed_text for c in batch], kind="document")
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)} texts")
        for c, v in zip(batch, vectors, strict=True):
            embeddings[c.content_hash] = vec_literal(v)

    async with pool.connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                """INSERT INTO documents(source, title, content_hash) VALUES (%s, %s, %s)
                   ON CONFLICT (source) DO UPDATE
                     SET title=EXCLUDED.title, content_hash=EXCLUDED.content_hash, updated_at=now()
                   RETURNING id""",
                (source, title, new_doc_hash),
            )
            doc_id = (await cur.fetchone())["id"]
            await conn.execute("DELETE FROM chunks WHERE document_id=%s", (doc_id,))
            async with conn.cursor() as c2:
                await c2.executemany(
                    "INSERT INTO chunks(document_id, chunk_index, heading, content, content_hash, "
                    "embedding) VALUES (%s, %s, %s, %s, %s, %s::vector)",
                    [
                        (
                            doc_id,
                            c.index,
                            c.heading,
                            c.content,
                            c.content_hash,
                            embeddings[c.content_hash],
                        )
                        for c in chunks
                    ],
                )
    return IngestResult(
        "ingested",
        chunks_total=len(chunks),
        chunks_embedded=len(missing),
        chunks_reused=len(chunks) - len(missing),
    )
