"""Offline, deterministic providers for tests and local dev without an API key.

The embedder hashes words into a fixed-size bag-of-words vector, so texts that share words are
close in cosine space: good enough to exercise retrieval plumbing, NOT a quality baseline."""

import hashlib
import math
import re
from typing import Literal

from .base import LLMResult

_WORD = re.compile(r"[a-z0-9_]+")


def fake_embedding(text: str, dim: int) -> list[float]:
    """Hash each lowercase word into one of `dim` buckets, count, and normalize to length 1.

    Args:
        text: Text to embed.
        dim: Vector length.

    Returns:
        A deterministic unit vector; texts sharing words point in similar directions.
    """
    vec = [0.0] * dim
    for w in _WORD.findall(text.lower()):
        h = int.from_bytes(hashlib.md5(w.encode()).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeEmbedder:
    """Offline `Embedder` that counts calls and texts embedded, so tests can assert on them."""

    def __init__(self, dim: int):
        self.dim = dim
        self.calls = 0
        self.texts_embedded = 0

    async def embed(
        self, texts: list[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        """Embed texts with `fake_embedding`; `kind` is ignored (interface parity only)."""
        self.calls += 1
        self.texts_embedded += len(texts)
        return [fake_embedding(t, self.dim) for t in texts]


class FakeLLM:
    """Offline `LLM` that returns a fixed placeholder string; never produces real answers."""

    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult:
        """Return a canned reply, estimating input tokens as len(prompt) // 4."""
        return LLMResult(text="[fake llm] no model configured", input_tokens=len(prompt) // 4)


class FakeReranker:
    """Offline `Reranker`: scores by shared words with the query. Deterministic, NOT a quality
    baseline (a real cross-encoder reads query and passage together; this only counts overlap)."""

    def __init__(self):
        self.calls = 0

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Return `(index, overlap score)` pairs, best first; ties keep the input order."""
        self.calls += 1
        q = set(_WORD.findall(query.lower()))
        scored = [
            (i, len(q & set(_WORD.findall(d.lower()))) / (len(q) or 1))
            for i, d in enumerate(documents)
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k] if top_k is not None else scored
