"""Postgres-backed job queue.

Why Postgres and not Redis/SQS here: one fewer moving part for a prototype, jobs are
transactional with the data they produce, and FOR UPDATE SKIP LOCKED gives safe concurrent
claiming. At higher volume or fan-out I'd move to SQS/Cloud Tasks (see README)."""

import logging

from psycopg_pool import AsyncConnectionPool

from app.retry import backoff_delay

log = logging.getLogger(__name__)


async def enqueue_many(pool: AsyncConnectionPool, docs: list[tuple[str, str, str]]) -> int:
    """Insert one pending job per document.

    A source that already has a pending/running job is skipped (partial unique index).

    Args:
        pool: Open connection pool.
        docs: (source, title, content) tuples.

    Returns:
        How many NEW jobs were created.
    """
    created = 0
    async with pool.connection() as conn, conn.cursor() as cur:
        for source, title, content in docs:
            await cur.execute(
                """INSERT INTO ingest_jobs(source, title, content) VALUES (%s, %s, %s)
                   ON CONFLICT (source) WHERE status IN ('pending','running') DO NOTHING""",
                (source, title, content),
            )
            created += cur.rowcount
    log.info(
        "enqueue: %d documents received, %d new jobs, %d already queued (pending or running)",
        len(docs),
        created,
        len(docs) - created,
    )
    return created


async def claim(pool: AsyncConnectionPool, visibility_timeout_s: int) -> dict | None:
    """Atomically claim one job. A 'running' job whose lock is older than the visibility
    timeout is assumed to belong to a crashed worker and is claimed again.

    FOR UPDATE SKIP LOCKED lets many workers call this at once without taking the same job.
    Each claim increments `attempts`, including claims of jobs abandoned by a crashed worker.

    Args:
        pool: Open connection pool.
        visibility_timeout_s: Seconds after which a 'running' job counts as abandoned.

    Returns:
        A dict with id, source, title, content and attempts, or None if nothing is claimable
        (queue empty, or remaining jobs are waiting for their retry time).
    """
    async with pool.connection() as conn:
        cur = await conn.execute(
            """UPDATE ingest_jobs
                  SET status='running', locked_at=now(), attempts=attempts+1, updated_at=now()
                WHERE id = (
                  SELECT id FROM ingest_jobs
                   WHERE (status='pending' AND run_after <= now())
                      OR (status='running' AND locked_at < now() - make_interval(secs => %s))
                   ORDER BY id
                   FOR UPDATE SKIP LOCKED
                   LIMIT 1)
            RETURNING id, source, title, content, attempts""",
            (visibility_timeout_s,),
        )
        job = await cur.fetchone()
    if job and job["attempts"] > 1:
        log.info(
            "claim: job %s (%s) is on attempt %d: an earlier attempt failed or its worker died",
            job["id"],
            job["source"],
            job["attempts"],
        )
    elif job:
        log.debug("claim: job %s (%s), first attempt", job["id"], job["source"])
    return job


async def heartbeat(pool: AsyncConnectionPool, job_id: int) -> None:
    """Renew a running job's lock so a slow-but-alive worker is not mistaken for a crashed one.

    Without this, a job that legitimately waits on a rate limiter for longer than the visibility
    timeout gets claimed by a second worker and embedded twice.
    """
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET locked_at=now() WHERE id=%s AND status='running'", (job_id,)
        )


async def complete(pool: AsyncConnectionPool, job_id: int) -> None:
    """Mark a job done and clear its lock and last error.

    Args:
        pool: Open connection pool.
        job_id: The job's id, from `claim`.
    """
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET status='done', locked_at=NULL, last_error=NULL, "
            "updated_at=now() WHERE id=%s",
            (job_id,),
        )


async def fail(
    pool: AsyncConnectionPool, job_id: int, error: str, attempts: int, max_attempts: int
) -> bool:
    """Record a failure and either schedule a retry or give up.

    Retries go back to 'pending' with `run_after` pushed out by exponential backoff with jitter
    (base 5s, cap 300s). Once `attempts` reaches `max_attempts` the job becomes 'failed'.

    Args:
        pool: Open connection pool.
        job_id: The job's id, from `claim`.
        error: Error text to store (truncated to 2000 characters).
        attempts: How many times the job has been claimed so far.
        max_attempts: Attempts allowed before giving up.

    Returns:
        True if the job will be retried, False if it is dead.
    """
    if attempts >= max_attempts:
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE ingest_jobs SET status='failed', locked_at=NULL, last_error=%s, "
                "updated_at=now() WHERE id=%s",
                (error[:2000], job_id),
            )
        log.warning(
            "job %s failed for good after %d of %d attempts: %s",
            job_id,
            attempts,
            max_attempts,
            error[:200],
        )
        return False
    delay = backoff_delay(attempts, base=5.0, cap=300.0)
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET status='pending', locked_at=NULL, last_error=%s, "
            "run_after=now() + make_interval(secs => %s), updated_at=now() WHERE id=%s",
            (error[:2000], delay, job_id),
        )
    log.info(
        "job %s: attempt %d of %d failed; retry in %.0fs (exponential backoff with jitter)",
        job_id,
        attempts,
        max_attempts,
        delay,
    )
    return True


async def release(pool: AsyncConnectionPool, job_id: int, delay_s: float) -> None:
    """Put a claimed job back WITHOUT burning an attempt.

    Used when the daily API budget runs out: the job did nothing wrong, so it must not count toward
    `max_attempts`.

    Args:
        pool: Open connection pool.
        job_id: The job's id, from `claim`.
        delay_s: Seconds until the job may be claimed again (time until the quota resets).
    """
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE ingest_jobs SET status='pending', attempts=GREATEST(attempts-1, 0), "
            "locked_at=NULL, run_after=now() + make_interval(secs => %s), updated_at=now() "
            "WHERE id=%s",
            (delay_s, job_id),
        )
    log.info(
        "job %s released for %.1f h (daily API budget used up); the attempt does not count",
        job_id,
        delay_s / 3600,
    )


async def stats(pool: AsyncConnectionPool) -> dict:
    """Summarize the queue and the data it has produced.

    Args:
        pool: Open connection pool.

    Returns:
        `{"jobs": {status: count}, "documents": int, "chunks": int}`.
    """
    async with pool.connection() as conn:
        jobs = await (
            await conn.execute("SELECT status, count(*) AS n FROM ingest_jobs GROUP BY status")
        ).fetchall()
        docs = await (await conn.execute("SELECT count(*) AS n FROM documents")).fetchone()
        chunks = await (await conn.execute("SELECT count(*) AS n FROM chunks")).fetchone()
    return {
        "jobs": {r["status"]: r["n"] for r in jobs},
        "documents": docs["n"],
        "chunks": chunks["n"],
    }
