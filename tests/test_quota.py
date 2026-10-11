import asyncio
from datetime import UTC, date, datetime

import pytest

import app.db as db
from app.quota import (
    DailyBudget,
    Governor,
    Limits,
    QuotaExhausted,
    SlidingWindowLimiter,
    estimate_tokens,
    quota_day,
    seconds_until_reset,
)


class FakeTime:
    """A clock whose sleep() just advances it: rate-limit tests run instantly."""

    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.slept.append(s)
        self.t += s


def limiter(rpm, tpm, ft):
    return SlidingWindowLimiter(rpm, tpm, clock=ft, sleep=ft.sleep)


# ---------- pure helpers
def test_estimate_tokens_overestimates_and_never_returns_zero():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abc") == 1 and estimate_tokens("abcd") == 2
    assert estimate_tokens("x" * 300) == 100


def test_quota_day_follows_pacific_midnight():
    # 06:59 UTC on Oct 11 is 23:59 PDT on Oct 10; two minutes later it is Oct 11 in Pacific.
    assert quota_day(datetime(2026, 10, 11, 6, 59, tzinfo=UTC)) == date(2026, 10, 10)
    assert quota_day(datetime(2026, 10, 11, 7, 1, tzinfo=UTC)) == date(2026, 10, 11)


def test_seconds_until_reset():
    assert seconds_until_reset(datetime(2026, 10, 11, 6, 59, tzinfo=UTC)) == pytest.approx(60)
    assert 0 < seconds_until_reset() <= 24 * 3600 + 3600


# ---------- sliding window
async def test_requests_over_rpm_wait_for_the_window():
    ft = FakeTime()
    lim = limiter(3, 10**9, ft)
    for _ in range(3):
        await lim.acquire()
    assert ft.slept == []  # three fit instantly
    await lim.acquire()  # the 4th must wait for the first to leave the 60 s window
    assert ft.t == pytest.approx(60.05)


async def test_tokens_over_tpm_wait_for_the_window():
    ft = FakeTime()
    lim = limiter(100, 100, ft)
    await lim.acquire(tokens=60)
    await lim.acquire(tokens=60)  # 120 > 100 within the window: waits
    assert ft.t == pytest.approx(60.05)
    await lim.acquire(tokens=30)  # 60 + 30 fits now: no more waiting
    assert ft.t == pytest.approx(60.05)


async def test_window_slides_instead_of_resetting_in_blocks():
    ft = FakeTime()
    lim = limiter(2, 10**9, ft)
    await lim.acquire()
    ft.t = 30
    await lim.acquire()
    await lim.acquire()  # needs the t=0 event to expire (at t=60), not the t=30 one
    assert ft.t == pytest.approx(60.05)


async def test_a_call_that_can_never_fit_is_rejected_not_deadlocked():
    lim = limiter(5, 100, FakeTime())
    with pytest.raises(ValueError, match="split it"):
        await lim.acquire(tokens=101)
    with pytest.raises(ValueError):
        await lim.acquire(units=6)


async def test_concurrent_callers_never_exceed_the_limit():
    ft = FakeTime()
    lim = limiter(4, 10**9, ft)
    await asyncio.gather(*(lim.acquire() for _ in range(10)))
    stamps = sorted(e[0] for e in lim._events)
    for i, t in enumerate(stamps):  # no 60 s window ever holds more than 4
        assert sum(1 for u in stamps if t <= u < t + 60) <= 4, (i, stamps)


# ---------- daily budget (real Postgres)
async def test_budget_allows_exactly_the_limit_then_raises(pool):
    b = DailyBudget("llm", 3, db.open_pool)
    assert [await b.reserve(1) for _ in range(3)] == [1, 2, 3]
    with pytest.raises(QuotaExhausted, match="used up"):
        await b.reserve(1)
    assert await b.used() == 3


async def test_budget_is_atomic_under_concurrency(pool):
    b = DailyBudget("embed", 10, db.open_pool)
    results = await asyncio.gather(*(b.reserve(1) for _ in range(25)), return_exceptions=True)
    assert sum(isinstance(r, int) for r in results) == 10
    assert sum(isinstance(r, QuotaExhausted) for r in results) == 15
    assert await b.used() == 10  # never overshoots


async def test_budget_units_and_oversize_requests(pool):
    b = DailyBudget("embed", 100, db.open_pool)
    assert await b.reserve(60) == 60
    with pytest.raises(QuotaExhausted):
        await b.reserve(41)  # 60 + 41 > 100: refused whole, nothing partially counted
    assert await b.used() == 60
    with pytest.raises(QuotaExhausted, match="daily limit"):
        await b.reserve(101)


async def test_budget_resets_on_a_new_day_and_apis_are_independent(pool):
    b, other = DailyBudget("llm", 2, db.open_pool), DailyBudget("embed", 2, db.open_pool)
    d1, d2 = date(2026, 1, 1), date(2026, 1, 2)
    await b.reserve(2, day=d1)
    with pytest.raises(QuotaExhausted):
        await b.reserve(1, day=d1)
    assert await b.reserve(1, day=d2) == 1  # new Pacific day, fresh budget
    assert await other.reserve(2, day=d1) == 2  # a different api has its own counter


async def test_mark_exhausted_blocks_everything_today(pool):
    b = DailyBudget("llm", 50, db.open_pool)
    await b.reserve(3)
    await b.mark_exhausted()
    assert await b.used() == 50
    with pytest.raises(QuotaExhausted):
        await b.reserve(1)
    await b.mark_exhausted()  # idempotent, and never lowers a higher count


# ---------- governor
class FakeBudget:
    def __init__(self):
        self.reserved, self.exhausted = [], False

    async def reserve(self, units):
        self.reserved.append(units)

    async def mark_exhausted(self):
        self.exhausted = True


class RateLimited(Exception):
    pass


DEFAULT_LIMITS = Limits(60, 100_000, 100)


def governor(ft, budget=None, cooldown=65.0, limits=DEFAULT_LIMITS):
    return Governor(
        "t",
        limits,
        budget or FakeBudget(),
        lambda e: isinstance(e, RateLimited),
        cooldown_s=cooldown,
        clock=ft,
        sleep=ft.sleep,
    )


def scripted(*outcomes):
    calls = []

    async def fn():
        calls.append(1)
        out = outcomes[len(calls) - 1]
        if isinstance(out, Exception):
            raise out
        return out

    return fn, calls


async def test_governor_reserves_budget_then_calls():
    ft, budget = FakeTime(), FakeBudget()
    fn, calls = scripted("ok")
    assert await governor(ft, budget).call(fn, units=3, tokens=10) == "ok"
    assert budget.reserved == [3] and len(calls) == 1 and ft.slept == []


async def test_exhausted_budget_stops_before_any_network_call():
    class Empty(FakeBudget):
        async def reserve(self, units):
            raise QuotaExhausted("t", "gone")

    fn, calls = scripted("never")
    with pytest.raises(QuotaExhausted):
        await governor(FakeTime(), Empty()).call(fn)
    assert calls == []


async def test_one_429_cools_down_then_succeeds_without_marking_the_day_exhausted():
    ft, budget = FakeTime(), FakeBudget()
    fn, calls = scripted(RateLimited(), "ok")
    assert await governor(ft, budget).call(fn) == "ok"
    assert len(calls) == 2 and ft.slept[0] == 65.0 and not budget.exhausted
    assert budget.reserved == [1]  # the retry does not reserve (and count) twice


async def test_two_429s_mean_the_quota_is_gone():
    ft, budget = FakeTime(), FakeBudget()
    fn, calls = scripted(RateLimited(), RateLimited())
    with pytest.raises(QuotaExhausted, match="still 429"):
        await governor(ft, budget).call(fn)
    assert len(calls) == 2 and budget.exhausted  # exactly two calls: no hammering


async def test_other_errors_propagate_immediately():
    ft = FakeTime()
    fn, calls = scripted(ValueError("bug"))
    with pytest.raises(ValueError):
        await governor(ft).call(fn)
    assert len(calls) == 1 and ft.slept == []


async def test_error_after_cooldown_propagates_without_marking_exhausted():
    ft, budget = FakeTime(), FakeBudget()
    fn, _ = scripted(RateLimited(), ValueError("boom"))
    with pytest.raises(ValueError):
        await governor(ft, budget).call(fn)
    assert not budget.exhausted


async def test_governor_applies_the_per_minute_limiter():
    ft = FakeTime()
    gov = governor(ft, limits=Limits(2, 100_000, 100))
    for _ in range(3):
        fn, _ = scripted("ok")
        await gov.call(fn)
    assert ft.t == pytest.approx(60.05)
