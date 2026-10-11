"""Ingestion workers: N asyncio tasks claiming jobs from Postgres.

    python -m app.worker --drain            # process everything queued, then exit
    python -m app.worker --concurrency 8    # keep polling forever
"""

import argparse
import asyncio
import logging
import time

from app.config import get_settings
from app.db import build_vector_index, close_pool, init_schema, open_pool
from app.ingest import queue
from app.ingest.pipeline import ingest_document
from app.providers import get_embedder

log = logging.getLogger("worker")


async def _has_live_jobs(pool) -> bool:
    """True if any job is still pending or running (used by --drain to know when to stop)."""
    async with pool.connection() as conn:
        row = await (
            await conn.execute(
                "SELECT EXISTS(SELECT 1 FROM ingest_jobs WHERE status IN ('pending','running')) AS x"
            )
        ).fetchone()
    return row["x"]


async def worker_loop(name: str, pool, embedder, counters: dict, drain: bool) -> None:
    """One worker: claim a job, ingest it, record the outcome, repeat.

    Failures are caught per job and handed to `queue.fail`, which schedules a retry or marks the
    job dead, so one bad document never stops the worker. A job claimed more than
    `job_max_attempts` times (e.g. repeatedly abandoned by crashed workers) is failed without
    being run.

    Args:
        name: Label used in log lines (e.g. "w0").
        pool: Open connection pool.
        embedder: Provider used to embed chunks.
        counters: Shared dict of totals, updated in place (safe: asyncio is single-threaded).
        drain: If true, return once no pending or running jobs remain; otherwise poll forever.
    """
    s = get_settings()
    while True:
        job = await queue.claim(pool, s.job_visibility_timeout_s)
        if job is None:
            if drain and not await _has_live_jobs(pool):
                return
            await asyncio.sleep(1.0)  # idle, or waiting on a retry's run_after
            continue
        if job["attempts"] > s.job_max_attempts:  # crashed workers burn attempts too
            await queue.fail(pool, job["id"], "max attempts exceeded", job["attempts"], s.job_max_attempts)
            counters["failed"] += 1
            continue
        try:
            res = await ingest_document(
                pool,
                embedder,
                source=job["source"],
                title=job["title"],
                content=job["content"],
                batch_size=s.embed_batch_size,
                max_chars=s.chunk_max_chars,
            )
            await queue.complete(pool, job["id"])
            counters[res.status] += 1
            counters["chunks_embedded"] += res.chunks_embedded
            counters["chunks_reused"] += res.chunks_reused
            log.info("%s %s %s (%d chunks, %d embedded)", name, res.status, job["source"],
                     res.chunks_total, res.chunks_embedded)
        except Exception as e:  # noqa: BLE001
            will_retry = await queue.fail(pool, job["id"], repr(e), job["attempts"], s.job_max_attempts)
            counters["retried" if will_retry else "failed"] += 1
            log.warning("%s error on %s: %r (%s)", name, job["source"], e,
                        "will retry" if will_retry else "giving up")


async def main() -> None:
    """CLI entry point: start `--concurrency` workers on one shared pool.

    With `--drain`, runs until the queue is empty, then builds the HNSW vector index (it is
    faster to build after a bulk load) and logs totals and embedding throughput.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=get_settings().worker_concurrency)
    ap.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    pool = await open_pool(max_size=args.concurrency + 2)
    await init_schema(pool)
    embedder = get_embedder()
    counters = {k: 0 for k in
                ("ingested", "skipped", "retried", "failed", "chunks_embedded", "chunks_reused")}
    t0 = time.perf_counter()
    await asyncio.gather(
        *(worker_loop(f"w{i}", pool, embedder, counters, args.drain) for i in range(args.concurrency))
    )
    elapsed = time.perf_counter() - t0
    if args.drain:
        await build_vector_index(pool)
        log.info("done in %.1fs: %s", elapsed, counters)
        if counters["chunks_embedded"]:
            log.info("embedding throughput: %.1f chunks/s", counters["chunks_embedded"] / elapsed)
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
