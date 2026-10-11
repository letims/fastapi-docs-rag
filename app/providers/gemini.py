import math
from typing import Literal

from google import genai
from google.genai import errors, types

from app.retry import with_retries

from .base import LLMResult


def _retryable(e: BaseException) -> bool:
    """True for Gemini errors worth retrying: rate limits (429) and server errors (5xx)."""
    # 429 = rate limited, 5xx = transient server trouble. Everything else is our bug: fail fast.
    return isinstance(e, errors.APIError) and (e.code == 429 or (e.code or 0) >= 500)


class GeminiLLM:
    """`LLM` implementation backed by the Gemini API."""

    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def generate(
        self, prompt: str, *, system: str = "", json: bool = False, temperature: float = 0.0
    ) -> LLMResult:
        """Generate text from a prompt, retrying 429/5xx errors with backoff.

        Args:
            prompt: The user prompt.
            system: System instruction; omitted if empty.
            json: If true, ask the model to reply with JSON.
            temperature: Sampling temperature; 0.0 keeps answers as repeatable as possible.

        Returns:
            An `LLMResult` with the text and input/output token counts (0 if not reported).
        """
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=temperature,
            response_mime_type="application/json" if json else None,
        )

        async def call():
            return await self.client.aio.models.generate_content(
                model=self.model, contents=prompt, config=config
            )

        resp = await with_retries(call, retry_on=_retryable)
        usage = resp.usage_metadata
        return LLMResult(
            text=resp.text or "",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        )


class GeminiEmbedder:
    """`Embedder` implementation backed by the Gemini embeddings API."""

    def __init__(self, api_key: str, model: str, dim: int):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dim = dim

    async def embed(
        self, texts: list[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        """Embed texts into unit-length vectors of `self.dim` dimensions.

        Args:
            texts: Texts to embed in one API call (callers batch).
            kind: "document" or "query". Currently unused (see TODO below).

        Returns:
            One normalized vector per input text, in the same order.
        """
        # TODO(verify in the Gemini embeddings docs): whether gemini-embedding-2 expects a
        # task prefix or task_type to distinguish queries from documents. `kind` is plumbed
        # through so this is a one-line change once confirmed.
        config = types.EmbedContentConfig(output_dimensionality=self.dim)

        async def call():
            return await self.client.aio.models.embed_content(
                model=self.model, contents=texts, config=config
            )

        resp = await with_retries(call, retry_on=_retryable)
        # Truncated (MRL) embeddings are not unit-norm: normalize so cosine == dot product.
        return [_normalize(e.values) for e in resp.embeddings]


def _normalize(v: list[float]) -> list[float]:
    """Scale `v` to length 1 (an all-zero vector is returned unchanged)."""
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]
