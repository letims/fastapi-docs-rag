import pytest

from app.cache import DiskCache
from app.providers.base import LLMResult
from app.providers.governed import (
    CachedEmbedder,
    CachedLLM,
    GovernedEmbedder,
    GovernedLLM,
    split_batches,
)
from app.quota import Governor, Limits


class RecordingEmbedder:
    dim = 3

    def __init__(self):
        self.batches: list[list[str]] = []

    async def embed(self, texts, *, kind="document"):
        self.batches.append(list(texts))
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class RecordingGovernor:
    """Stands in for Governor: records (units, tokens) per call, runs the call straight away."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    async def call(self, fn, *, units=1, tokens=0):
        self.calls.append((units, tokens))
        return await fn()


# ---------- batching
def test_split_batches_by_items_and_tokens():
    texts = ["x" * 30] * 7  # 10 estimated tokens each
    assert [len(b) for b in split_batches(texts, max_items=3, max_tokens=10**6)] == [3, 3, 1]
    assert [len(b) for b in split_batches(texts, max_items=100, max_tokens=25)] == [2, 2, 2, 1]
    assert split_batches([], 3, 10) == []
    assert split_batches(["x" * 300], 3, 10) == [["x" * 300]]  # an oversize text still goes alone


# ---------- governed embedder
def make_governed(unit, rpm=60, tpm=100_000, max_tokens=15_000):
    inner, gov = RecordingEmbedder(), RecordingGovernor()
    return (
        GovernedEmbedder(
            inner, gov, unit=unit, limits=Limits(rpm, tpm, 100), max_batch_tokens=max_tokens
        ),
        inner,
        gov,
    )


async def test_text_unit_counts_every_text_and_caps_batches_at_rpm():
    emb, inner, gov = make_governed("text", rpm=5)
    out = await emb.embed(["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "g"])
    assert [len(b) for b in inner.batches] == [5, 2]  # never more texts per call than RPM allows
    assert [u for u, _ in gov.calls] == [5, 2]  # units == texts
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]  # order preserved


async def test_call_unit_counts_one_request_per_batch():
    emb, inner, gov = make_governed("call")
    await emb.embed(["a"] * 150)
    assert [len(b) for b in inner.batches] == [100, 50]
    assert [u for u, _ in gov.calls] == [1, 1]


async def test_batches_respect_the_token_cap():
    emb, inner, _ = make_governed("call", max_tokens=50)
    await emb.embed(["x" * 60] * 5)  # 20 tokens each -> 2 per batch
    assert [len(b) for b in inner.batches] == [2, 2, 1]


async def test_tokens_are_estimated_and_empty_input_makes_no_call():
    emb, _, gov = make_governed("text")
    await emb.embed(["x" * 30])
    assert gov.calls == [(1, 10)]
    gov.calls.clear()
    assert await emb.embed([]) == [] and gov.calls == []


def test_unknown_unit_is_rejected():
    with pytest.raises(ValueError, match="'text' or 'call'"):
        make_governed("batch")


async def test_governed_llm_budgets_prompt_plus_output_reserve():
    class LLM:
        async def generate(self, prompt, *, system="", json=False, temperature=0.0):
            return LLMResult("hi", 1, 1)

    gov = RecordingGovernor()
    res = await GovernedLLM(LLM(), gov, output_reserve=1000).generate("x" * 30, system="y" * 6)
    assert res.text == "hi" and gov.calls == [(1, 10 + 2 + 1000)]


async def test_real_governor_stack_never_exceeds_limits():
    """End to end with the real Governor: 3 text-units/min means 7 texts need 3 minutes."""

    class FakeTime:
        t = 0.0

        def __call__(self):
            return self.t

        async def sleep(self, s):
            self.t += s

    class FakeBudget:
        def __init__(self):
            self.reserved = []

        async def reserve(self, units):
            self.reserved.append(units)

        async def mark_exhausted(self):
            pass

    ft, budget = FakeTime(), FakeBudget()
    gov = Governor(
        "embed", Limits(3, 100_000, 100), budget, lambda e: False, clock=ft, sleep=ft.sleep
    )
    emb = GovernedEmbedder(
        RecordingEmbedder(),
        gov,
        unit="text",
        limits=Limits(3, 100_000, 100),
        max_batch_tokens=15_000,
    )
    await emb.embed(["t"] * 7)  # batches of 3, 3, 1
    assert budget.reserved == [3, 3, 1] and ft.t >= 120  # two full waits of the 60 s window


# ---------- cache
async def test_cached_embedder_pays_once_per_text():
    inner = RecordingEmbedder()
    emb = CachedEmbedder(inner, DiskCache(":memory:"), "m")
    first = await emb.embed(["alpha", "beta"])
    again = await emb.embed(["beta", "gamma", "alpha"])  # only "gamma" is new
    assert inner.batches == [["alpha", "beta"], ["gamma"]]
    assert (emb.hits, emb.misses) == (2, 3)
    assert again[0] == pytest.approx(first[1]) and again[2] == pytest.approx(first[0])


async def test_cache_key_separates_model_kind_and_dim():
    cache, inner = DiskCache(":memory:"), RecordingEmbedder()
    await CachedEmbedder(inner, cache, "m1").embed(["t"])
    await CachedEmbedder(inner, cache, "m2").embed(["t"])  # different model: miss
    await CachedEmbedder(inner, cache, "m1").embed(["t"], kind="query")  # different kind: miss
    assert len(inner.batches) == 3


async def test_cached_embedder_fully_cached_makes_no_call_and_exposes_batch_limits():
    inner = RecordingEmbedder()
    emb = CachedEmbedder(inner, DiskCache(":memory:"), "m")
    await emb.embed(["a"])
    inner.batches.clear()
    await emb.embed(["a"])
    assert inner.batches == []
    assert (
        emb.max_batch_items == 100 and emb.max_batch_tokens == 15_000
    )  # defaults for plain inners


async def test_cached_embedder_rejects_a_short_reply():
    class Short(RecordingEmbedder):
        async def embed(self, texts, *, kind="document"):
            return []

    with pytest.raises(RuntimeError, match="returned 0 vectors"):
        await CachedEmbedder(Short(), DiskCache(":memory:"), "m").embed(["a"])


class CountingLLM:
    def __init__(self, text="answer [1]"):
        self.text, self.calls = text, 0

    async def generate(self, prompt, *, system="", json=False, temperature=0.0):
        self.calls += 1
        return LLMResult(self.text, 11, 7)


async def test_cached_llm_replays_identical_prompts_for_free():
    inner = CountingLLM()
    llm = CachedLLM(inner, DiskCache(":memory:"), "m")
    first = await llm.generate("q", system="s")
    second = await llm.generate("q", system="s")
    assert (first.cached, second.cached) == (False, True)
    assert second.text == "answer [1]" and (second.input_tokens, second.output_tokens) == (11, 7)
    assert inner.calls == 1 and (llm.hits, llm.misses) == (1, 1)
    await llm.generate("q", system="other system")  # any input change is a new request
    await llm.generate("q", system="s", temperature=0.5)
    assert inner.calls == 3


async def test_cached_llm_never_pins_an_empty_answer():
    inner = CountingLLM(text="   ")
    llm = CachedLLM(inner, DiskCache(":memory:"), "m")
    await llm.generate("q")
    await llm.generate("q")
    assert inner.calls == 2
