import asyncio
import sys
from collections import defaultdict

import pytest

from app import worker
from app.config import get_settings
from app.ingest import queue
from app.worker import worker_loop

GOOD = "# A\n\n## S\n\nsome text to embed"


class BoomEmbedder:
    dim = 64

    async def embed(self, texts, *, kind="document"):
        raise RuntimeError("boom")


async def test_failing_job_is_retried_then_dead_lettered(pool, monkeypatch):
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()
    monkeypatch.setattr(queue, "backoff_delay", lambda *a, **k: 0.0)
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    counters = defaultdict(int)
    await asyncio.wait_for(worker_loop("w", pool, BoomEmbedder(), counters, drain=True), 30)
    assert counters["retried"] == 1 and counters["failed"] == 1
    async with pool.connection() as c:
        row = await (await c.execute("SELECT status, last_error FROM ingest_jobs")).fetchone()
    assert row["status"] == "failed" and "boom" in row["last_error"]
    assert (await queue.stats(pool))["documents"] == 0


async def test_job_that_keeps_crashing_workers_is_dead_lettered(pool, emb, monkeypatch):
    monkeypatch.setenv("JOB_MAX_ATTEMPTS", "2")
    get_settings.cache_clear()
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    async with pool.connection() as c:
        await c.execute("UPDATE ingest_jobs SET attempts = 10")  # claimed (and lost) many times
    counters = defaultdict(int)
    await asyncio.wait_for(worker_loop("w", pool, emb, counters, drain=True), 30)
    assert counters["failed"] == 1 and counters["ingested"] == 0
    assert (await queue.stats(pool))["jobs"] == {"failed": 1}


async def test_idle_worker_waits_for_backoff_instead_of_exiting(pool, emb):
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    async with pool.connection() as c:
        await c.execute("UPDATE ingest_jobs SET run_after = now() + interval '1 hour'")
    counters = defaultdict(int)
    with pytest.raises(asyncio.TimeoutError):  # drain mode must NOT exit: a job is still live
        await asyncio.wait_for(worker_loop("w", pool, emb, counters, drain=True), 0.5)
    assert counters["ingested"] == 0


async def test_main_drains_queue_and_builds_vector_index(pool, monkeypatch):
    docs = [(f"d{i}.md", f"D{i}", f"# D{i}\n\n## S\n\ncontent number {i}") for i in range(5)]
    await queue.enqueue_many(pool, docs)
    monkeypatch.setattr(sys, "argv", ["worker", "--concurrency", "2", "--drain"])
    await worker.main()  # closes the global pool when done
    import app.db as db

    p = await db.open_pool()
    assert (await queue.stats(p))["jobs"] == {"done": 5}
    async with p.connection() as c:
        row = await (
            await c.execute("SELECT count(*) AS n FROM pg_indexes WHERE indexname='chunks_embedding_hnsw'")
        ).fetchone()
    assert row["n"] == 1
