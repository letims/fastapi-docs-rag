"""The scenario that actually happened, replayed against a strict simulated Gemini free tier.

The simulator rejects any request that would exceed 60 RPM / 100K TPM / 100 RPD with the same
body-less 429 a real key returned. Every text counts as one request: the worst case. The real worker,
queue, Postgres budget and governor run against it; only the clock is fake, so it takes milliseconds.
"""

import asyncio
from collections import defaultdict, deque
from datetime import date

import pytest
from google.genai import errors

import app.quota as quota_mod
from app import db
from app.cache import DiskCache
from app.ingest import queue
from app.providers.fake import fake_embedding
from app.providers.gemini import is_rate_limited
from app.providers.governed import CachedEmbedder, GovernedEmbedder
from app.quota import DailyBudget, Governor, Limits
from app.worker import worker_loop


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.t += s


class SimulatedGemini:
    dim = 64

    def __init__(self, clock, rpm=60, tpm=100_000, rpd=100):
        self.clock, self.rpm, self.tpm, self.rpd = clock, rpm, tpm, rpd
        self.window: deque[tuple[float, int, int]] = deque()
        self.today = 0
        self.rejected = 0
        self.texts_embedded = 0
        self.calls = 0

    def new_day(self):
        self.today = 0

    async def embed(self, texts, *, kind="document"):
        now = self.clock()
        while self.window and self.window[0][0] <= now - 60:
            self.window.popleft()
        units, tokens = (
            len(texts),
            sum(len(t) // 4 + 1 for t in texts),
        )  # real tokens < our estimate
        over = (
            sum(w[1] for w in self.window) + units > self.rpm
            or sum(w[2] for w in self.window) + tokens > self.tpm
            or self.today + units > self.rpd
        )
        if over:
            self.rejected += 1
            raise errors.APIError(
                429,
                {
                    "error": {  # exactly what the real key returned: no QuotaFailure
                        "code": 429,
                        "status": "RESOURCE_EXHAUSTED",
                        "message": "You exceeded your current quota, please check your plan and billing details.",
                        "details": [{"@type": "type.googleapis.com/google.rpc.Help", "links": []}],
                    }
                },
            )
        self.window.append((now, units, tokens))
        self.today += units
        self.calls += 1
        self.texts_embedded += units
        return [fake_embedding(t, self.dim) for t in texts]


async def test_the_simulator_is_strict():
    """Guards the guard: an ungoverned burst is rejected, so passing below is meaningful."""
    sim = SimulatedGemini(Clock())
    results = await asyncio.gather(*(sim.embed(["t"]) for _ in range(61)), return_exceptions=True)
    assert sim.rejected == 1 and sum(isinstance(r, errors.APIError) for r in results) == 1
    big = SimulatedGemini(Clock())
    with pytest.raises(errors.APIError):
        await big.embed(["x" * 500_000])  # > 100K tokens in one call


async def test_ingest_under_free_tier_limits_never_gets_a_429_and_resumes_next_day(
    pool, monkeypatch
):
    clock = Clock()
    sim = SimulatedGemini(clock)
    limits = Limits(60, 100_000, 100)
    gov = Governor(
        "embed",
        limits,
        DailyBudget("embed", limits.rpd, db.open_pool),
        is_rate_limited,
        clock=clock,
        sleep=clock.sleep,
    )
    emb = CachedEmbedder(
        GovernedEmbedder(sim, gov, unit="text", limits=limits, max_batch_tokens=15_000),
        DiskCache(":memory:"),
        "sim",
    )

    n_docs, chunks_per_doc = 40, 3  # 120 chunks > 100 requests/day: cannot finish in one day
    docs = [
        (
            f"d{i}.md",
            f"D{i}",
            f"# D{i}\n\n## A\n\nalpha {i}\n\n## B\n\nbeta {i}\n\n## C\n\ngamma {i}",
        )
        for i in range(n_docs)
    ]
    assert await queue.enqueue_many(pool, docs) == n_docs

    history = []
    for n in range(1, 8):
        current = date(2026, 3, n)
        monkeypatch.setattr(quota_mod, "quota_day", lambda current=current: current)
        sim.new_day()
        async with pool.connection() as c:  # time passes: paused jobs become eligible at the reset
            await c.execute("UPDATE ingest_jobs SET run_after = now() WHERE status = 'pending'")
        stop, counters = asyncio.Event(), defaultdict(int)
        await asyncio.wait_for(
            asyncio.gather(
                *(
                    worker_loop(f"w{i}", pool, emb, counters, drain=True, stop=stop)
                    for i in range(4)
                )
            ),
            60,
        )
        history.append((sim.today, stop.is_set(), counters["ingested"]))
        if (await queue.stats(pool))["jobs"] == {"done": n_docs}:
            break

    assert sim.rejected == 0, "the governor must keep us under every limit: zero 429s"
    assert all(used <= 100 for used, _, _ in history), history
    assert len(history) == 2, history  # day 1 hits the daily budget and pauses; day 2 finishes
    assert history[0][1] is True and history[0][0] >= 99  # paused with the day's budget used
    assert history[1][1] is False  # finished without hitting the limit
    stats = await queue.stats(pool)
    assert stats["jobs"] == {"done": n_docs} and stats["documents"] == n_docs
    assert stats["chunks"] == n_docs * chunks_per_doc
    assert sim.texts_embedded == n_docs * chunks_per_doc  # nothing paid for twice across the pause
