"""Provider interfaces. The rest of the app only knows these, so swapping one vendor for another
(Voyage, Groq, Gemini, ...) is a config change plus one adapter file."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class LLMResult:
    """What one LLM call returns: the text plus token counts for cost and budget tracking."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # served from the disk cache: cost no API request


class LLM(Protocol):
    """A chat model. `generate` sends one prompt (with an optional system prompt) and returns the
    reply; `json=True` asks the model for a JSON object; temperature 0 keeps answers repeatable."""

    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult: ...


class Embedder(Protocol):
    """Turns texts into vectors of `dim` numbers. `kind` tells vendors that distinguish documents
    from queries which one this is; the query side must use the same model as the stored chunks."""

    dim: int

    async def embed(
        self, texts: list[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]: ...


class Reranker(Protocol):
    """A cross-encoder: reads the query and each candidate together and scores how well the
    candidate answers it (slower than vector search, but more precise)."""

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Score documents against the query; return (index into documents, score), best first."""
        ...
