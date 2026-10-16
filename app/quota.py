"""Client-side quota governance for rate-limited APIs (built for Gemini's free tier).

Why not just retry on 429? Google's 429 body for quota errors carries no machine-readable hint of
WHICH quota was hit (RPM, TPM or RPD), so reacting to errors cannot tell "wait a minute" from
"come back tomorrow". So we count ourselves:

  * SlidingWindowLimiter: never exceed RPM / TPM in any rolling 60 s window (waits, never fails)
  * DailyBudget: RPD counter in Postgres, shared by every process, resets at midnight Pacific
  * Governor: reserve budget -> wait for the window -> call. A 429 anyway means our accounting is
    off or the day's quota is gone: cool down once and retry; a second 429 marks the day exhausted
    and raises QuotaExhausted, which callers treat as "pause", not "fail".
"""

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Protocol, TypeVar
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
PACIFIC = ZoneInfo("America/Los_Angeles")  # Gemini's daily quota resets at midnight Pacific
T = TypeVar("T")


class QuotaExhausted(Exception):
    """The day's budget for `api` is used up. Do not retry; resume after the reset."""

    def __init__(self, api: str, detail: str = ""):
        super().__init__(f"{api}: {detail}" if detail else api)
        self.api = api


@dataclass(frozen=True)
class Limits:
    rpm: int
    tpm: int
    rpd: int


def estimate_tokens(text: str) -> int:
    """Deliberate over-estimate (~3 chars/token): prose is nearer 4, code nearer 3."""
    return max(1, math.ceil(len(text) / 3))


def quota_day(now: datetime | None = None) -> date:
    return (now or datetime.now(PACIFIC)).astimezone(PACIFIC).date()


def seconds_until_reset(now: datetime | None = None) -> float:
    now = (now or datetime.now(PACIFIC)).astimezone(PACIFIC)
    midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=PACIFIC)
    return (midnight.astimezone(UTC) - now.astimezone(UTC)).total_seconds()  # UTC: DST-safe


class SlidingWindowLimiter:
    """At most `rpm` request-units and `tpm` tokens in any rolling 60 s window.
    acquire() WAITS until the call fits; waiters are served in order (the lock is held while
    sleeping, which is what we want: nobody may jump the queue)."""

    WINDOW_S = 60.0

    def __init__(
        self,
        rpm: int,
        tpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.rpm, self.tpm = rpm, tpm
        self._clock, self._sleep = clock, sleep
        self._events: deque[tuple[float, int, int]] = deque()  # (time, units, tokens)
        self._lock = asyncio.Lock()

    async def acquire(self, units: int = 1, tokens: int = 0) -> None:
        """Wait until `units` requests and `tokens` tokens fit in the rolling window, then count them.

        Raises:
            ValueError: One call alone is bigger than a whole minute allows (it must be split).
        """
        if units > self.rpm or tokens > self.tpm:
            raise ValueError(
                f"a single call ({units} units, {tokens} tokens) exceeds the per-minute limits "
                f"({self.rpm} RPM, {self.tpm} TPM); split it into smaller calls"
            )
        async with self._lock:
            while True:
                now = self._clock()
                while self._events and self._events[0][0] <= now - self.WINDOW_S:
                    self._events.popleft()
                used_units = sum(e[1] for e in self._events)
                used_tokens = sum(e[2] for e in self._events)
                if used_units + units <= self.rpm and used_tokens + tokens <= self.tpm:
                    self._events.append((now, units, tokens))
                    return
                wait = self._events[0][0] + self.WINDOW_S - now + 0.05  # until the oldest expires
                log.info(
                    "rate limit: waiting %.1fs (%d/%d RPM, %d/%d TPM in window)",
                    wait,
                    used_units,
                    self.rpm,
                    used_tokens,
                    self.tpm,
                )
                await self._sleep(max(wait, 0.05))


class Budget(Protocol):
    """A daily request counter. `reserve` raises `QuotaExhausted` when the day is used up;
    `mark_exhausted` records that the provider itself said so."""

    async def reserve(self, units: int) -> None: ...
    async def mark_exhausted(self) -> None: ...


_RESERVE_SQL = """
INSERT INTO api_usage(day, api, requests) VALUES (%s, %s, %s)
ON CONFLICT (day, api) DO UPDATE SET requests = api_usage.requests + EXCLUDED.requests
  WHERE api_usage.requests + EXCLUDED.requests <= %s
RETURNING requests"""


class DailyBudget:
    """Requests-per-day counter in Postgres. reserve() is one atomic statement, so concurrent
    workers (even in different processes) can never overshoot the limit."""

    def __init__(self, api: str, rpd: int, get_pool: Callable[[], Awaitable]):
        self.api, self.rpd, self._get_pool = api, rpd, get_pool

    async def reserve(self, units: int = 1, *, day: date | None = None) -> int:
        """Atomically add `units` to today's counter and return the new total.

        Raises:
            QuotaExhausted: The addition would pass the daily limit (nothing is added).
        """
        if units > self.rpd:
            raise QuotaExhausted(
                self.api, f"a call needs {units} units but the daily limit is {self.rpd}"
            )
        pool = await self._get_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(_RESERVE_SQL, (day or quota_day(), self.api, units, self.rpd))
            ).fetchone()
        if row is None:
            raise QuotaExhausted(
                self.api,
                f"daily budget of {self.rpd} used up; resets in {seconds_until_reset() / 3600:.1f} h",
            )
        used = row["requests"]
        if used >= 0.8 * self.rpd and used - units < 0.8 * self.rpd:
            log.warning(
                "daily budget %s: %d of %d requests used (80%%); it resets in %.1f h",
                self.api,
                used,
                self.rpd,
                seconds_until_reset() / 3600,
            )
        return used

    async def used(self, *, day: date | None = None) -> int:
        """Requests counted so far today (0 if none)."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT requests FROM api_usage WHERE day=%s AND api=%s",
                    (day or quota_day(), self.api),
                )
            ).fetchone()
        return row["requests"] if row else 0

    async def mark_exhausted(self, *, day: date | None = None) -> None:
        """Set today's counter to the limit, after the provider refused a request we thought was ok."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """INSERT INTO api_usage(day, api, requests) VALUES (%s, %s, %s)
                   ON CONFLICT (day, api) DO UPDATE
                     SET requests = GREATEST(api_usage.requests, EXCLUDED.requests)""",
                (day or quota_day(), self.api, self.rpd),
            )


class Governor:
    """The single door for calls to one external API: daily budget, then per-minute window, then the call.

    A 429 despite our own limits gets one cool-down and one retry; a second 429 marks the day used up
    and raises `QuotaExhausted`, which callers treat as "pause", not "fail"."""

    def __init__(
        self,
        api: str,
        limits: Limits,
        budget: Budget,
        is_rate_limited: Callable[[BaseException], bool],
        *,
        cooldown_s: float = 65.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.api, self.budget, self.is_rate_limited = api, budget, is_rate_limited
        self.cooldown_s, self._sleep = cooldown_s, sleep
        self.limiter = SlidingWindowLimiter(limits.rpm, limits.tpm, clock=clock, sleep=sleep)

    async def call(self, fn: Callable[[], Awaitable[T]], *, units: int = 1, tokens: int = 0) -> T:
        """Run `fn` (one provider call) under the budget and rate limits.

        Args:
            fn: Zero-argument async function that performs the call; may be run twice (after a 429).
            units: Request units this call counts for (texts, or 1 per call).
            tokens: Estimated tokens, for the per-minute token limit.

        Raises:
            QuotaExhausted: Daily budget used up, or the provider kept answering 429.
        """
        await self.budget.reserve(units)  # raises QuotaExhausted BEFORE touching the network
        await self.limiter.acquire(units, tokens)
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            if not self.is_rate_limited(e):
                raise
            log.warning(
                "%s: got 429 despite local limits; cooling down %.0fs and retrying once. "
                "Provider said: %s",
                self.api,
                self.cooldown_s,
                str(e)[:400],
            )
        await self._sleep(self.cooldown_s)
        await self.limiter.acquire(units, tokens)
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            if not self.is_rate_limited(e):
                raise
            await self.budget.mark_exhausted()
            raise QuotaExhausted(
                self.api,
                f"still 429 after a cool-down: treating today's quota as used up. "
                f"Provider said: {str(e)[:400]}",
            ) from e
