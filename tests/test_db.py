import app.db as db


def test_vec_literal_format():
    assert db.vec_literal([0.5, 1, -2.25]) == "[0.5,1,-2.25]"
    assert db.vec_literal([]) == "[]"


async def test_schema_and_index_creation_are_idempotent(pool):
    await db.init_schema(pool)
    await db.build_vector_index(pool)
    await db.build_vector_index(pool)
    async with pool.connection() as c:
        row = await (
            await c.execute(
                "SELECT count(*) AS n FROM pg_indexes WHERE indexname='chunks_embedding_hnsw'"
            )
        ).fetchone()
    assert row["n"] == 1


async def test_open_pool_is_a_singleton(pool):
    assert await db.open_pool() is pool
