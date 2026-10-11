"""Voyage AI: embeddings and reranking over plain HTTP (no SDK)."""

from typing import Literal

import httpx

from .http import post_json

BASE_URL = "https://api.voyageai.com/v1"
# Voyage 4 / 3.5 models are Matryoshka: they can return these sizes only (not 768).
VOYAGE_DIMS = (256, 512, 1024, 2048)


def _client(api_key: str, client: httpx.AsyncClient | None) -> httpx.AsyncClient:
    return client or httpx.AsyncClient(headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0)


class VoyageEmbedder:
    """`Embedder` backed by the Voyage embeddings API.

    Voyage returns unit-length vectors and has a native query/document distinction
    (`input_type`), so unlike the Gemini adapter there is no prefix workaround.
    """

    def __init__(
        self, api_key: str, model: str, dim: int, *, client: httpx.AsyncClient | None = None
    ):
        if dim not in VOYAGE_DIMS:
            raise ValueError(
                f"Voyage supports EMBED_DIM in {VOYAGE_DIMS}, got {dim}. "
                "Set EMBED_DIM=1024 in .env (and use a fresh database)."
            )
        self.client, self.model, self.dim = _client(api_key, client), model, dim

    async def embed(
        self, texts: list[str], *, kind: Literal["document", "query"] = "document"
    ) -> list[list[float]]:
        body = await post_json(
            self.client,
            f"{BASE_URL}/embeddings",
            {"input": texts, "model": self.model, "input_type": kind, "output_dimension": self.dim},
        )
        rows = sorted(body["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]


class VoyageReranker:
    """Cross-encoder reranking (Phase 2): scores each (query, document) pair jointly."""

    def __init__(self, api_key: str, model: str, *, client: httpx.AsyncClient | None = None):
        self.client, self.model = _client(api_key, client), model

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Return `(index into documents, relevance score)` pairs, best first."""
        payload: dict = {"query": query, "documents": documents, "model": self.model}
        if top_k is not None:
            payload["top_k"] = top_k
        body = await post_json(self.client, f"{BASE_URL}/rerank", payload)
        return [(r["index"], r["relevance_score"]) for r in body["data"]]
