import pytest

from app.retry import backoff_delay, with_retries


async def _nosleep(_):
    return None


async def test_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429")
        return "ok"

    assert await with_retries(flaky, attempts=5, sleep=_nosleep) == "ok"
    assert calls["n"] == 3


async def test_gives_up_after_attempts():
    calls = {"n": 0}

    async def always():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await with_retries(always, attempts=3, sleep=_nosleep)
    assert calls["n"] == 3


async def test_non_retryable_fails_fast():
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        await with_retries(
            bad, attempts=5, retry_on=lambda e: isinstance(e, RuntimeError), sleep=_nosleep
        )
    assert calls["n"] == 1


def test_backoff_is_bounded_and_jittered():
    for attempt in range(12):
        d = backoff_delay(attempt, base=1.0, cap=30.0)
        assert 0 <= d <= 30.0
    assert len({backoff_delay(5) for _ in range(20)}) > 1  # not deterministic
