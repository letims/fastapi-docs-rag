import logging
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

log = logging.getLogger(__name__)
_pool: AsyncConnectionPool | None = None


async def open_pool(max_size: int = 10) -> AsyncConnectionPool:
    """Open the process-wide connection pool, creating it on first call.

    Later calls return the existing pool, so `max_size` only takes effect the first time.
    Rows come back as dicts (`dict_row`).

    Args:
        max_size: Maximum number of connections the pool may hold (minimum is 1).

    Returns:
        The shared, already-open `AsyncConnectionPool`.
    """
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row},
        )
        await _pool.open()
    return _pool


async def close_pool() -> None:
    """Close the shared pool and reset it so `open_pool` can create a fresh one.

    Safe to call when no pool is open (does nothing).

    Returns:
        None.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def vec_literal(v: list[float]) -> str:
    """Format a vector as pgvector text. Avoids needing a client-side adapter: use with %s::vector.

    Args:
        v: The embedding values.

    Returns:
        A string like "[0.1,0.2,0.3]", with each value at 7 significant digits.
    """
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


async def init_schema(pool: AsyncConnectionPool) -> None:
    """Create the database schema from `sql/schema.sql`.

    The `{dim}` placeholder in the file is replaced with the configured embedding
    dimension (`embed_dim`) before the script runs.

    Args:
        pool: An open pool to run the script on.

    Returns:
        None.
    """
    sql = (Path(__file__).parent.parent / "sql" / "schema.sql").read_text()
    sql = sql.replace("{dim}", str(get_settings().embed_dim))
    async with pool.connection() as conn:
        await conn.execute(sql)


async def build_vector_index(pool: AsyncConnectionPool) -> None:
    """Build the HNSW index after bulk load. Cosine ops: embeddings are unit-normalized.

    Uses `IF NOT EXISTS`, so repeated calls are harmless. Building on a large table
    can take a while.

    Args:
        pool: An open pool to run the statement on.

    Returns:
        None.
    """
    async with pool.connection() as conn:
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )


if __name__ == "__main__":
    import asyncio

    async def _main():
        pool = await open_pool()
        await init_schema(pool)
        log.warning("schema ready")
        print("schema ready")
        await close_pool()

    asyncio.run(_main())
