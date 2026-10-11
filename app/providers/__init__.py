from app.config import get_settings

from .base import LLM, Embedder

_cache: dict[str, object] = {}


def get_embedder() -> Embedder:
    """Return the process-wide embedder chosen by the `provider` setting (fake or Gemini).

    Built on first use and cached, so every caller shares one client.
    """
    if "embedder" not in _cache:
        s = get_settings()
        if s.provider == "fake":
            from .fake import FakeEmbedder

            _cache["embedder"] = FakeEmbedder(s.embed_dim)
        else:
            from .gemini import GeminiEmbedder

            _cache["embedder"] = GeminiEmbedder(s.gemini_api_key, s.gemini_embed_model, s.embed_dim)
    return _cache["embedder"]  # type: ignore[return-value]


def get_llm() -> LLM:
    """Return the process-wide LLM chosen by the `provider` setting (fake or Gemini).

    Built on first use and cached, so every caller shares one client.
    """
    if "llm" not in _cache:
        s = get_settings()
        if s.provider == "fake":
            from .fake import FakeLLM

            _cache["llm"] = FakeLLM()
        else:
            from .gemini import GeminiLLM

            _cache["llm"] = GeminiLLM(s.gemini_api_key, s.gemini_llm_model)
    return _cache["llm"]  # type: ignore[return-value]
