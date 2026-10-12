"""Wrappers that make any provider quota-safe and (optionally) free on repeat.

    CachedEmbedder -> GovernedEmbedder -> GeminiEmbedder      (same for the LLM)

Cache is outermost so a hit costs nothing: no budget reserved, no waiting, no network."""

import json as _json
import logging

from app.cache import DiskCache, make_key, pack_vector, unpack_vector
from app.providers.base import LLM, Embedder, LLMResult, Reranker
from app.quota import Governor, Limits, estimate_tokens

log = logging.getLogger(__name__)

MAX_TEXTS_PER_CALL = 100  # assumed Gemini batchEmbedContents cap; also the "call" unit upper bound


def split_batches(texts: list[str], max_items: int, max_tokens: int) -> list[list[str]]:
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for t in texts:
        n = estimate_tokens(t)
        if cur and (len(cur) >= max_items or cur_tokens + n > max_tokens):
            batches.append(cur)
            cur, cur_tokens = [], 0
        cur.append(t)
        cur_tokens += n
    if cur:
        batches.append(cur)
    return batches


class GovernedEmbedder:
    """unit='text': every text counts as one request against RPM/RPD (conservative, cannot overshoot
    whichever way Google counts). unit='call': one batch call = one request (use only after
    confirming on the AI Studio usage page that batches count once)."""

    def __init__(
        self,
        inner: Embedder,
        governor: Governor,
        *,
        unit: str,
        limits: Limits,
        max_batch_tokens: int,
    ):
        if unit not in ("text", "call"):
            raise ValueError("embed quota unit must be 'text' or 'call'")
        self.inner, self.governor, self.unit, self.dim = inner, governor, unit, inner.dim
        self.max_batch_items = (
            min(MAX_TEXTS_PER_CALL, limits.rpm) if unit == "text" else MAX_TEXTS_PER_CALL
        )
        self.max_batch_tokens = min(max_batch_tokens, limits.tpm)

    async def embed(self, texts: list[str], *, kind="document") -> list[list[float]]:
        out: list[list[float]] = []
        for batch in split_batches(texts, self.max_batch_items, self.max_batch_tokens):
            units = len(batch) if self.unit == "text" else 1
            tokens = sum(estimate_tokens(t) for t in batch)
            out.extend(
                await self.governor.call(
                    lambda b=batch: self.inner.embed(b, kind=kind), units=units, tokens=tokens
                )
            )
        return out


class GovernedLLM:
    def __init__(self, inner: LLM, governor: Governor, *, output_reserve: int):
        self.inner, self.governor, self.output_reserve = inner, governor, output_reserve

    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult:
        tokens = estimate_tokens(prompt) + estimate_tokens(system) + self.output_reserve
        return await self.governor.call(
            lambda: self.inner.generate(prompt, system=system, json=json, temperature=temperature),
            units=1,
            tokens=tokens,
        )


class CachedEmbedder:
    def __init__(self, inner: Embedder, cache: DiskCache, model: str):
        self.inner, self.cache, self.model, self.dim = inner, cache, model, inner.dim
        self.max_batch_items = getattr(inner, "max_batch_items", MAX_TEXTS_PER_CALL)
        self.max_batch_tokens = getattr(inner, "max_batch_tokens", 15_000)
        self.hits = self.misses = 0

    async def embed(self, texts: list[str], *, kind="document") -> list[list[float]]:
        keys = [make_key("emb", self.model, str(self.dim), kind, t) for t in texts]
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, k in enumerate(keys):
            blob = self.cache.get(k)
            if blob is None:
                missing.append(i)
            else:
                out[i] = unpack_vector(blob)
        self.hits += len(texts) - len(missing)
        if missing:
            vectors = await self.inner.embed([texts[i] for i in missing], kind=kind)
            if len(vectors) != len(missing):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(missing)} texts"
                )
            self.misses += len(missing)
            for i, v in zip(missing, vectors, strict=True):
                out[i] = v
                self.cache.set(keys[i], pack_vector(v))
        return out  # type: ignore[return-value]


class CachedLLM:
    def __init__(self, inner: LLM, cache: DiskCache, model: str):
        self.inner, self.cache, self.model = inner, cache, model
        self.hits = self.misses = 0

    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult:
        key = make_key("llm", self.model, str(temperature), str(json), system, prompt)
        blob = self.cache.get(key)
        if blob is not None:
            d = _json.loads(blob)
            self.hits += 1
            return LLMResult(d["text"], d["in"], d["out"], cached=True)
        res = await self.inner.generate(prompt, system=system, json=json, temperature=temperature)
        self.misses += 1
        if res.text.strip():  # never pin an empty/failed answer forever
            self.cache.set(
                key,
                _json.dumps(
                    {"text": res.text, "in": res.input_tokens, "out": res.output_tokens}
                ).encode(),
            )
        return res


class GovernedReranker:
    """One rerank call is one request; its tokens are the query plus every document.

    If the candidates would not fit in a single minute of token budget, the tail is dropped (the
    candidates arrive best-first from the first-stage search, so the tail matters least). Indexes
    in the result still refer to the caller's original list."""

    def __init__(self, inner: Reranker, governor: Governor, *, limits: Limits):
        self.inner, self.governor, self.limits = inner, governor, limits

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        budget = self.limits.tpm - estimate_tokens(query)
        keep, used = 0, 0
        for d in documents:
            n = estimate_tokens(d)
            if keep and used + n > budget:
                break
            keep, used = keep + 1, used + n
        if keep < len(documents):
            log.warning("rerank: %d of %d candidates fit the token budget", keep, len(documents))
        docs = documents[:keep]
        return await self.governor.call(
            lambda: self.inner.rerank(query, docs, top_k),
            units=1,
            tokens=estimate_tokens(query) + used,
        )


class CachedReranker:
    """Persistent cache for rerank results, keyed by (model, query, top_k, every document)."""

    def __init__(self, inner: Reranker, cache: DiskCache, model: str):
        self.inner, self.cache, self.model = inner, cache, model
        self.hits = self.misses = 0

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        key = make_key("rerank", self.model, str(top_k), query, *documents)
        blob = self.cache.get(key)
        if blob is not None:
            self.hits += 1
            return [(int(i), float(s)) for i, s in _json.loads(blob)]
        res = await self.inner.rerank(query, documents, top_k)
        self.misses += 1
        self.cache.set(key, _json.dumps(res).encode())
        return res
