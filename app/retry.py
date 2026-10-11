"""Retry with exponential backoff and full jitter.

Full jitter (sleep = U(0, min(cap, base * 2**n))) avoids synchronized retry storms when many
workers hit the same rate limit at once.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    """Delay before retry number `attempt` (0-based), full jitter.

    Args:
        attempt: 0 for the first retry, 1 for the second, and so on.
        base: Delay ceiling for attempt 0, in seconds; the ceiling doubles each attempt.
        cap: Upper limit on the ceiling, in seconds.

    Returns:
        A random number of seconds between 0 and min(cap, base * 2**attempt).
    """
    return random.uniform(0, min(cap, base * (2**attempt)))


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 6,
    base: float = 1.0,
    cap: float = 60.0,
    retry_on: Callable[[BaseException], bool] = lambda e: True,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call `fn` until it succeeds, sleeping with backoff between failures.

    Args:
        fn: Zero-argument async callable to run. Called again from scratch on each retry.
        attempts: Total tries, including the first.
        base: Backoff base in seconds (see `backoff_delay`).
        cap: Backoff cap in seconds.
        retry_on: Called with the exception; return False to re-raise immediately. Use it to
            retry only transient errors (429/5xx) and fail fast on bugs.
        sleep: Awaitable sleep function; injectable so tests do not wait.

    Returns:
        Whatever `fn` returned on its first successful call.

    Raises:
        Exception: The last error, once attempts are exhausted or `retry_on` rejects it.
    """
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001
            if attempt == attempts - 1 or not retry_on(e):
                raise
            delay = backoff_delay(attempt, base, cap)
            log.warning("retrying after %s (attempt %d/%d, sleeping %.1fs)", e, attempt + 1, attempts, delay)
            await sleep(delay)
    raise AssertionError("unreachable")
