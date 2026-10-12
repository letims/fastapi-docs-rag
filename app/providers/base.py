"""Provider interfaces. The rest of the app only knows these, so swapping Gemini for Claude
(or anything else) is a config change plus one adapter file."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # served from the disk cache: cost no API request


class LLM(Protocol):
    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult: ...


class Embedder(Protocol):
    dim: int

    async def embed(
        self, texts: list[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Score documents against the query; return (index into documents, score), best first."""
        ...
