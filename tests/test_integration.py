"""Ingestion pipeline + queue + retrieval against a real Postgres with pgvector."""

from collections import defaultdict

import pytest

from app.ingest import queue
from app.ingest.pipeline import ingest_document
from app.rag.retrieve import vector_search
from app.worker import worker_loop

DOC = """# Events

## Startup

Use the lifespan parameter to run code before the app starts receiving requests.

## Shutdown

Code after the yield in lifespan runs when the application is shutting down.
"""


async def _ingest(pool, emb, source="events.md", content=DOC, title="Events"):
    return await ingest_document(pool, emb, source=source, title=title, content=content)


async def test_reingest_unchanged_doc_is_a_noop(pool, emb):
    r1 = await _ingest(pool, emb)
    assert r1.status == "ingested" and r1.chunks_embedded == r1.chunks_total > 0
    before = emb.texts_embedded
    r2 = await _ingest(pool, emb)
    assert r2.status == "skipped"
    assert emb.texts_embedded == before  # zero embedding calls


async def test_edit_reuses_embeddings_for_unchanged_chunks(pool, emb):
    await _ingest(pool, emb)
    r = await _ingest(pool, emb, content=DOC.replace("shutting down", "stopping"))
    assert r.status == "ingested"
    assert r.chunks_reused >= 1 and r.chunks_embedded == 1
    stats = await queue.stats(pool)
    assert stats["documents"] == 1 and stats["chunks"] == r.chunks_total  # no duplicates


async def test_identical_chunks_in_one_doc_are_embedded_once(pool, emb):
    twin = "# T\n\n## A\n\nsame text\n\n## A\n\nsame text\n"
    r = await _ingest(pool, emb, "twin.md", twin, "T")
    assert r.chunks_total == 2 and r.chunks_embedded == 1


async def test_embedder_returning_wrong_count_is_an_error(pool):
    class Short:
        dim = 64

        async def embed(self, texts, *, kind="document"):
            return []

    with pytest.raises(RuntimeError, match="returned 0 vectors"):
        await _ingest(pool, Short())
    assert (await queue.stats(pool))["documents"] == 0  # nothing half-written


async def test_enqueue_dedupes_live_jobs(pool):
    assert await queue.enqueue_many(pool, [("a.md", "A", "x")]) == 1
    assert await queue.enqueue_many(pool, [("a.md", "A", "x")]) == 0


async def test_concurrent_workers_process_each_job_exactly_once(pool, emb):
    docs = [(f"d{i}.md", f"Doc {i}", f"# Doc {i}\n\n## S\n\nunique words number{i} here") for i in range(30)]
    assert await queue.enqueue_many(pool, docs) == 30
    counters = defaultdict(int)
    import asyncio

    await asyncio.gather(*(worker_loop(f"w{i}", pool, emb, counters, drain=True) for i in range(6)))
    stats = await queue.stats(pool)
    assert counters["ingested"] == 30
    assert stats["jobs"] == {"done": 30}
    assert stats["documents"] == 30


async def test_failure_backoff_then_dead_letter(pool):
    await queue.enqueue_many(pool, [("a.md", "A", "x")])
    job = await queue.claim(pool, 300)
    assert await queue.fail(pool, job["id"], "boom", attempts=1, max_attempts=2) is True
    assert await queue.claim(pool, 300) is None  # backoff: run_after is in the future
    async with pool.connection() as c:
        await c.execute("UPDATE ingest_jobs SET run_after = now()")
    job = await queue.claim(pool, 300)
    assert job["attempts"] == 2
    assert await queue.fail(pool, job["id"], "boom", attempts=2, max_attempts=2) is False
    assert (await queue.stats(pool))["jobs"] == {"failed": 1}


async def test_stale_lock_from_crashed_worker_is_reclaimed(pool):
    await queue.enqueue_many(pool, [("a.md", "A", "x")])
    job = await queue.claim(pool, 300)
    assert await queue.claim(pool, 300) is None  # still locked by the (live) first worker
    async with pool.connection() as c:
        await c.execute("UPDATE ingest_jobs SET locked_at = now() - interval '1 hour'")
    again = await queue.claim(pool, 300)
    assert again["id"] == job["id"] and again["attempts"] == 2


async def test_complete_marks_done_and_clears_lock(pool):
    await queue.enqueue_many(pool, [("a.md", "A", "x")])
    job = await queue.claim(pool, 300)
    await queue.complete(pool, job["id"])
    assert (await queue.stats(pool))["jobs"] == {"done": 1}
    assert await queue.enqueue_many(pool, [("a.md", "A", "y")]) == 1  # done jobs don't block re-enqueue


async def test_vector_search_returns_the_right_document(pool, emb):
    await _ingest(pool, emb)
    await _ingest(pool, emb, "static.md", "# Static Files\n\n## Mount\n\nServe static files from a directory with StaticFiles.")
    await _ingest(pool, emb, "cors.md", "# CORS\n\n## Origins\n\nAllow cross origin requests from a browser frontend.")
    hits = await vector_search(pool, emb, "how do I serve static files from a directory", k=3)
    assert hits[0].source == "static.md"
    assert hits[0].score >= hits[1].score
    assert len(await vector_search(pool, emb, "anything", k=1)) == 1
