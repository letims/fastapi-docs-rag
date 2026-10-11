import asyncio
import sys
from collections import defaultdict

import pytest

from app import worker
from app.config import get_settings
from app.ingest import queue
from app.quota import QuotaExhausted
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
            await c.execute(
                "SELECT count(*) AS n FROM pg_indexes WHERE indexname='chunks_embedding_hnsw'"
            )
        ).fetchone()
    assert row["n"] == 1


class QuotaEmbedder:
    dim = 64

    async def embed(self, texts, *, kind="document"):
        raise QuotaExhausted("embed", "daily budget used up")


async def _attempts_and_status(pool):
    async with pool.connection() as c:
        return await (
            await c.execute(
                "SELECT status, attempts, run_after > now() + interval '1 minute' AS delayed FROM ingest_jobs ORDER BY id"
            )
        ).fetchall()


async def test_quota_exhaustion_puts_the_job_back_without_burning_an_attempt(pool):
    await queue.enqueue_many(pool, [(f"d{i}.md", f"D{i}", GOOD) for i in range(3)])
    stop, counters = asyncio.Event(), defaultdict(int)
    await asyncio.wait_for(
        asyncio.gather(
            *(
                worker_loop(f"w{i}", pool, QuotaEmbedder(), counters, drain=True, stop=stop)
                for i in range(3)
            )
        ),
        30,
    )
    assert stop.is_set() and counters["paused"] >= 1 and counters["failed"] == 0
    rows = await _attempts_and_status(pool)
    assert {r["status"] for r in rows} == {"pending"}  # nothing failed, nothing lost
    assert all(r["attempts"] == 0 for r in rows)  # the quota is not the job's fault
    assert any(r["delayed"] for r in rows)  # the released job waits for the quota reset


async def test_workers_stop_claiming_once_the_stop_flag_is_set(pool, emb):
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    stop = asyncio.Event()
    stop.set()
    counters = defaultdict(int)
    await asyncio.wait_for(worker_loop("w", pool, emb, counters, drain=True, stop=stop), 5)
    assert counters["ingested"] == 0 and (await queue.stats(pool))["jobs"] == {"pending": 1}


async def test_main_reports_a_paused_run_and_keeps_the_queue(pool, monkeypatch, caplog):
    await queue.enqueue_many(pool, [("a.md", "A", GOOD), ("b.md", "B", GOOD)])
    monkeypatch.setattr(worker, "get_embedder", lambda: QuotaEmbedder())
    monkeypatch.setattr(sys, "argv", ["worker", "--concurrency", "2", "--drain"])
    with caplog.at_level("WARNING", logger="worker"):
        await worker.main()
    assert "daily API budget is used up" in caplog.text
    import app.db as db

    p = await db.open_pool()
    assert (await queue.stats(p))["jobs"] == {"pending": 2}


async def test_release_never_drives_attempts_negative(pool):
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    job = await queue.claim(pool, 300)
    await queue.release(pool, job["id"], 0)
    await queue.release(pool, job["id"], 0)  # released twice: still >= 0
    assert (await _attempts_and_status(pool))[0]["attempts"] == 0


class SlowEmbedder:
    """Takes longer than the visibility timeout, like a job waiting on a rate limiter."""

    dim = 64

    def __init__(self, delay_s):
        self.delay_s, self.calls = delay_s, 0

    async def embed(self, texts, *, kind="document"):
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        from app.providers.fake import fake_embedding

        return [fake_embedding(t, self.dim) for t in texts]


async def test_slow_job_is_not_stolen_by_another_worker(pool, monkeypatch):
    """A job that outlives the visibility timeout while its worker is alive keeps its lock."""
    monkeypatch.setenv("JOB_VISIBILITY_TIMEOUT_S", "1")
    get_settings.cache_clear()
    emb = SlowEmbedder(delay_s=2.5)
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    counters = defaultdict(int)
    await asyncio.wait_for(
        asyncio.gather(*(worker_loop(f"w{i}", pool, emb, counters, drain=True) for i in range(2))),
        30,
    )
    assert emb.calls == 1 and counters["ingested"] == 1  # without the heartbeat: 2 and 2
    assert (await queue.stats(pool))["jobs"] == {"done": 1}


async def test_heartbeat_ignores_jobs_that_are_not_running(pool):
    await queue.enqueue_many(pool, [("a.md", "A", GOOD)])
    async with pool.connection() as c:
        job_id = (await (await c.execute("SELECT id FROM ingest_jobs")).fetchone())["id"]
    await queue.heartbeat(pool, job_id)  # pending: must not mark it locked
    async with pool.connection() as c:
        row = await (await c.execute("SELECT status, locked_at FROM ingest_jobs")).fetchone()
    assert row["status"] == "pending" and row["locked_at"] is None
