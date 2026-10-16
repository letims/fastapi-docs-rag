"""The log lines are how a person follows a long run, so the important ones are tested:
what was decided and why, not the exact wording."""

import asyncio
import logging
import re
import sys
import time

import app.db as db
from app import worker
from app.config import get_settings
from app.ingest import queue
from app.ingest.pipeline import ingest_document
from app.logging_config import setup_logging
from app.quota import DailyBudget


def msgs(caplog, level=logging.DEBUG):
    return [r.getMessage() for r in caplog.records if r.levelno >= level]


def test_setup_logging_applies_the_level_and_quiets_http_clients(monkeypatch):
    root, httpx_logger = logging.getLogger(), logging.getLogger("httpx")
    before = root.level, httpx_logger.level
    try:
        monkeypatch.setenv("LOG_LEVEL", "debug")
        get_settings.cache_clear()
        setup_logging()
        assert root.level == logging.DEBUG and httpx_logger.level == logging.WARNING
        setup_logging("WARNING")  # an explicit level wins over LOG_LEVEL
        assert root.level == logging.WARNING
    finally:
        root.setLevel(before[0])
        httpx_logger.setLevel(before[1])


async def test_ingest_explains_new_unchanged_and_changed_documents(pool, emb, caplog):
    caplog.set_level(logging.DEBUG, logger="app.ingest.pipeline")
    doc = "# T\n\n## A\n\nfirst section text\n\n## B\n\nsecond section text"

    async def run(content):
        await ingest_document(pool, emb, source="d.md", title="T", content=content)

    await run(doc)
    await run(doc)
    await run(doc.replace("second", "changed"))
    m = msgs(caplog)
    assert any("new document" in x and "to embed" in x for x in m)
    assert any("unchanged" in x and "no embedding calls" in x for x in m)
    assert any(re.search(r"content changed: \d+ chunks, [1-9]\d* reuse", x) for x in m)
    assert any("embedding batch 1 of 1" in x for x in m)


async def test_queue_logs_enqueue_retry_backoff_dead_letter_and_release(pool, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="app.ingest.queue")
    monkeypatch.setattr(queue, "backoff_delay", lambda *a, **k: 0.0)
    await queue.enqueue_many(pool, [("a.md", "A", "x")])
    await queue.enqueue_many(pool, [("a.md", "A", "x")])  # already live: skipped
    job = await queue.claim(pool, 300)
    assert await queue.fail(pool, job["id"], "boom", 1, 3) is True
    job = await queue.claim(pool, 300)  # second attempt
    assert await queue.fail(pool, job["id"], "boom again", 3, 3) is False
    await queue.enqueue_many(pool, [("b.md", "B", "x")])
    job = await queue.claim(pool, 300)
    await queue.release(pool, job["id"], 7200)
    m = msgs(caplog)
    assert any("1 new jobs, 0 already queued" in x for x in m)
    assert any("0 new jobs, 1 already queued" in x for x in m)
    assert any("first attempt" in x for x in m)
    assert any("attempt 1 of 3 failed; retry in 0s" in x for x in m)
    assert any("is on attempt 2" in x for x in m)
    assert any("failed for good after 3 of 3" in x for x in m)
    assert any("released for 2.0 h" in x and "does not count" in x for x in m)


async def test_budget_warns_once_when_it_crosses_80_percent(pool, caplog):
    caplog.set_level(logging.WARNING, logger="app.quota")
    b = DailyBudget("llm", 10, db.open_pool)
    await b.reserve(7)
    assert not msgs(caplog)
    await b.reserve(1)  # 8 of 10
    await b.reserve(1)
    warned = [x for x in msgs(caplog) if "80%" in x]
    assert len(warned) == 1 and "8 of 10" in warned[0]


async def test_progress_reports_counts_rate_and_time_left(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="worker")
    replies = [
        {"jobs": {"done": 5}},
        {"jobs": {"done": 7, "pending": 3, "running": 1}},
        {"jobs": {"done": 5, "pending": 4, "running": 1}},  # nothing finished yet: no rate
    ]

    async def stats(_pool):
        return replies.pop(0) if replies else {"jobs": {}}

    monkeypatch.setattr(worker.queue, "stats", stats)
    task = asyncio.create_task(worker._progress(None, 0.01, time.perf_counter() - 60))
    await asyncio.sleep(0.15)
    task.cancel()
    lines = [x for x in msgs(caplog) if x.startswith("progress:")]
    assert "2 done this run, 3 pending, 1 running, 0 failed" in lines[0]
    assert "docs/min" in lines[0] and "min left" in lines[0]
    assert "time left unknown yet" in lines[1]


async def test_main_logs_the_start_state_and_the_cache_summary(pool, monkeypatch, caplog):
    await queue.enqueue_many(pool, [("d0.md", "D0", "# D0\n\n## S\n\ncontent")])

    class Cached:
        dim = 64
        hits, misses = 3, 1

        async def embed(self, texts, *, kind="document"):
            return [[0.1] * 64 for _ in texts]

    monkeypatch.setattr(worker, "get_embedder", lambda: Cached())
    monkeypatch.setattr(sys, "argv", ["worker", "--concurrency", "1", "--drain"])
    caplog.set_level(logging.INFO)
    await worker.main()
    m = msgs(caplog)
    assert any("starting 1 workers" in x for x in m)
    assert any("queue at start" in x and "pending" in x for x in m)
    assert any("3 texts served from disk" in x and "1 sent to the API" in x for x in m)
    assert any("vector index: building HNSW over 1 chunks" in x for x in m)
    assert any("ingested d0.md: 1 chunks (1 embedded, 0 reused)" in x for x in m)
