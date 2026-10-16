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
from app.logging_config import setup_logging
from app.providers import get_embedder
from app.quota import QuotaExhausted, seconds_until_reset

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


async def _keepalive(pool, job_id: int, every_s: float) -> None:
    """Renew the job's lock every `every_s` seconds until cancelled (see `queue.heartbeat`)."""
    while True:
        await asyncio.sleep(every_s)
        await queue.heartbeat(pool, job_id)


async def _progress(pool, every_s: float, t0: float) -> None:
    """Log queue progress every `every_s` seconds: a long run shows it is alive, how far along it
    is and a rough time left (from the rate so far in this run)."""
    done_at_start = (await queue.stats(pool))["jobs"].get("done", 0)
    while True:
        await asyncio.sleep(every_s)
        jobs = (await queue.stats(pool))["jobs"]
        done = jobs.get("done", 0) - done_at_start
        left = jobs.get("pending", 0) + jobs.get("running", 0)
        rate = done / max(time.perf_counter() - t0, 1e-9) * 60
        eta = f"about {left / rate:.0f} min left" if rate > 0 else "time left unknown yet"
        log.info(
            "progress: %d done this run, %d pending, %d running, %d failed | %.1f docs/min | %s",
            done,
            jobs.get("pending", 0),
            jobs.get("running", 0),
            jobs.get("failed", 0),
            rate,
            eta,
        )


async def worker_loop(
    name: str, pool, embedder, counters: dict, drain: bool, stop: asyncio.Event | None = None
) -> None:
    """One worker: claim a job, ingest it, record the outcome, repeat.

    Failures are caught per job and handed to `queue.fail`, which schedules a retry or marks the
    job dead, so one bad document never stops the worker. A job claimed more than
    `job_max_attempts` times (e.g. repeatedly abandoned by crashed workers) is failed without
    being run.

    If the daily API budget runs out (`QuotaExhausted`), the job is put back WITHOUT burning an
    attempt, scheduled for the quota reset, and `stop` is set so every worker stops cleanly.

    Args:
        name: Label used in log lines (e.g. "w0").
        pool: Open connection pool.
        embedder: Provider used to embed chunks.
        counters: Shared dict of totals, updated in place (safe: asyncio is single-threaded).
        drain: If true, return once no pending or running jobs remain; otherwise poll forever.
        stop: Shared flag. Set by the first worker to hit `QuotaExhausted`; all workers then return.
    """
    s = get_settings()
    while True:
        if stop is not None and stop.is_set():
            return
        job = await queue.claim(pool, s.job_visibility_timeout_s)
        if job is None:
            if drain and not await _has_live_jobs(pool):
                log.debug("%s: queue is empty, exiting", name)
                return
            await asyncio.sleep(1.0)  # idle, or waiting on a retry's run_after
            continue
        if job["attempts"] > s.job_max_attempts:  # crashed workers burn attempts too
            await queue.fail(
                pool, job["id"], "max attempts exceeded", job["attempts"], s.job_max_attempts
            )
            log.warning(
                "%s: %s was claimed %d times (limit %d); giving up on it without running it",
                name,
                job["source"],
                job["attempts"],
                s.job_max_attempts,
            )
            counters["failed"] += 1
            continue
        t_job = time.perf_counter()
        keepalive = asyncio.create_task(
            _keepalive(pool, job["id"], max(s.job_visibility_timeout_s / 3, 0.1))
        )
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
            if res.status == "skipped":
                log.info("%s skipped %s (unchanged)", name, job["source"])
            else:
                log.info(
                    "%s ingested %s: %d chunks (%d embedded, %d reused) in %.1fs",
                    name,
                    job["source"],
                    res.chunks_total,
                    res.chunks_embedded,
                    res.chunks_reused,
                    time.perf_counter() - t_job,
                )
        except QuotaExhausted as e:
            await queue.release(pool, job["id"], delay_s=seconds_until_reset())
            counters["paused"] += 1
            log.warning(
                "%s: %s; put %s back in the queue and stopping all workers", name, e, job["source"]
            )
            if stop is not None:
                stop.set()
            return
        except Exception as e:  # noqa: BLE001
            will_retry = await queue.fail(
                pool, job["id"], repr(e), job["attempts"], s.job_max_attempts
            )
            counters["retried" if will_retry else "failed"] += 1
            log.warning(
                "%s error on %s: %r (%s)",
                name,
                job["source"],
                e,
                "will retry" if will_retry else "giving up",
            )
        finally:
            keepalive.cancel()


async def main() -> None:
    """CLI entry point: start `--concurrency` workers on one shared pool.

    With `--drain`, runs until the queue is empty, then builds the HNSW vector index (it is
    faster to build after a bulk load) and logs totals and embedding throughput.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=get_settings().worker_concurrency)
    ap.add_argument("--drain", action="store_true", help="exit when the queue is empty")
    args = ap.parse_args()
    setup_logging()

    pool = await open_pool(max_size=args.concurrency + 2)
    await init_schema(pool)
    embedder = get_embedder()
    s = get_settings()
    st = await queue.stats(pool)
    log.info(
        "starting %d workers (embedding batch %d, lock timeout %ds, max %d attempts per job)",
        args.concurrency,
        s.embed_batch_size,
        s.job_visibility_timeout_s,
        s.job_max_attempts,
    )
    log.info(
        "queue at start: %s | already stored: %d documents, %d chunks",
        st["jobs"] or "empty",
        st["documents"],
        st["chunks"],
    )
    counters = {
        k: 0
        for k in (
            "ingested",
            "skipped",
            "retried",
            "failed",
            "paused",
            "chunks_embedded",
            "chunks_reused",
        )
    }
    stop = asyncio.Event()
    t0 = time.perf_counter()
    progress = asyncio.create_task(_progress(pool, s.progress_interval_s, t0))
    await asyncio.gather(
        *(
            worker_loop(f"w{i}", pool, embedder, counters, args.drain, stop)
            for i in range(args.concurrency)
        )
    )
    progress.cancel()
    elapsed = time.perf_counter() - t0
    if hasattr(embedder, "hits"):
        log.info(
            "embedding cache: %d texts served from disk (free), %d sent to the API",
            embedder.hits,
            embedder.misses,
        )
    if stop.is_set():
        log.warning(
            "Stopped early: the daily API budget is used up. Unprocessed jobs stay queued and become "
            "eligible again at the quota reset (in %.1f h). Embeddings already computed are cached, "
            "so re-running pays for nothing twice. See `make quota`.",
            seconds_until_reset() / 3600,
        )
    if args.drain:
        await build_vector_index(pool)
        log.info("done in %.1fs: %s", elapsed, counters)
        if counters["chunks_embedded"]:
            log.info("embedding throughput: %.1f chunks/s", counters["chunks_embedded"] / elapsed)
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
