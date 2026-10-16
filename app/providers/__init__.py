import logging

from app.cache import DiskCache
from app.config import Settings, get_settings
from app.db import open_pool

from .base import LLM, Embedder, Reranker

log = logging.getLogger(__name__)
_cache: dict[str, object] = {}


def get_disk_cache() -> DiskCache:
    """Return the process-wide on-disk API cache (created on first use at `API_CACHE_PATH`)."""
    if "disk" not in _cache:
        _cache["disk"] = DiskCache(get_settings().api_cache_path)
    return _cache["disk"]  # type: ignore[return-value]


def _governor(s: Settings, api: str, is_rate_limited):
    """Build the (Limits, Governor) pair that enforces RPM/TPM locally and RPD via Postgres.

    Limits come from the provider's preset unless overridden in `.env`; the daily counter is keyed
    per provider, so switching vendors never inherits another vendor's usage.
    """
    from app.quota import DailyBudget, Governor, Limits

    rpm, tpm, rpd = s.quota(api)
    limits = Limits(rpm, tpm, rpd)
    log.info(
        "quota %s: %d requests/min, %d tokens/min, %d requests/day", s.usage_key(api), rpm, tpm, rpd
    )
    return limits, Governor(
        api,
        limits,
        DailyBudget(s.usage_key(api), rpd, open_pool),
        is_rate_limited,
        cooldown_s=s.quota_cooldown_s,
    )


def _require_key(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is empty: put it in .env (never in chat or in git)")
    return value


def _embed_inner(s: Settings) -> tuple[Embedder, str, object]:
    """Return (raw embedder, model name for the cache key, rate-limit classifier)."""
    backend = s.embed_backend
    if backend == "gemini":
        from .gemini import GeminiEmbedder, is_rate_limited

        return (
            GeminiEmbedder(s.gemini_api_key, s.gemini_embed_model, s.embed_dim),
            s.gemini_embed_model,
            is_rate_limited,
        )
    if backend == "voyage":
        from .http import is_rate_limited
        from .voyage import VoyageEmbedder

        key = _require_key("VOYAGE_API_KEY", s.voyage_api_key)
        return (
            VoyageEmbedder(key, s.voyage_embed_model, s.embed_dim),
            s.voyage_embed_model,
            is_rate_limited,
        )
    raise ValueError(f"unknown embedding provider {backend!r} (gemini | voyage | fake)")


def _llm_inner(s: Settings) -> tuple[LLM, str, object]:
    """Return (raw chat model, model name for the cache key, rate-limit classifier) for `LLM_PROVIDER`."""
    backend = s.llm_backend
    if backend == "gemini":
        from .gemini import GeminiLLM, is_rate_limited

        return GeminiLLM(s.gemini_api_key, s.gemini_llm_model), s.gemini_llm_model, is_rate_limited
    from .http import is_rate_limited
    from .openai_compat import GROQ_BASE_URL, OpenAICompatLLM

    if backend == "groq":
        key = _require_key("GROQ_API_KEY", s.groq_api_key)
        return OpenAICompatLLM(key, s.groq_model, GROQ_BASE_URL), s.groq_model, is_rate_limited
    if backend == "openai_compat":
        key = _require_key("OPENAI_COMPAT_API_KEY", s.openai_compat_api_key)
        if not (s.openai_compat_base_url and s.openai_compat_model):
            raise RuntimeError("OPENAI_COMPAT_BASE_URL and OPENAI_COMPAT_MODEL are required")
        return (
            OpenAICompatLLM(key, s.openai_compat_model, s.openai_compat_base_url),
            s.openai_compat_model,
            is_rate_limited,
        )
    raise ValueError(f"unknown LLM provider {backend!r} (gemini | groq | openai_compat | fake)")


def get_embedder() -> Embedder:
    """Return the shared embedder.

    `fake` gives the offline embedder. Otherwise the chosen vendor, wrapped in the quota Governor
    and (if `API_CACHE`) the persistent cache, so a cache hit costs no quota at all.
    """
    if "embedder" not in _cache:
        s = get_settings()
        if s.embed_backend == "fake":
            from .fake import FakeEmbedder

            emb: Embedder = FakeEmbedder(s.embed_dim)
        else:
            from .governed import CachedEmbedder, GovernedEmbedder

            inner, model, classify = _embed_inner(s)
            limits, gov = _governor(s, "embed", classify)
            emb = GovernedEmbedder(
                inner,
                gov,
                unit=s.embed_unit,
                limits=limits,
                max_batch_tokens=s.embed_batch_max_tokens,
            )
            if s.api_cache:
                emb = CachedEmbedder(emb, get_disk_cache(), model)
        log.info(
            "embedder ready: %s, dim %d, disk cache %s",
            s.embed_backend,
            s.embed_dim,
            "on" if s.api_cache and s.embed_backend != "fake" else "off",
        )
        _cache["embedder"] = emb
    return _cache["embedder"]  # type: ignore[return-value]


def get_llm() -> LLM:
    """Return the shared LLM, wrapped like `get_embedder` (quota governor, then persistent cache)."""
    if "llm" not in _cache:
        s = get_settings()
        if s.llm_backend == "fake":
            from .fake import FakeLLM

            llm: LLM = FakeLLM()
        else:
            from .governed import CachedLLM, GovernedLLM

            inner, model, classify = _llm_inner(s)
            _, gov = _governor(s, "llm", classify)
            llm = GovernedLLM(inner, gov, output_reserve=s.llm_output_token_reserve)
            if s.api_cache:
                llm = CachedLLM(llm, get_disk_cache(), model)
        log.info(
            "llm ready: %s, disk cache %s",
            s.llm_backend,
            "on" if s.api_cache and s.llm_backend != "fake" else "off",
        )
        _cache["llm"] = llm
    return _cache["llm"]  # type: ignore[return-value]


def get_reranker() -> Reranker:
    """Return the shared reranker, wrapped like the others (cache outermost, then the governor)."""
    if "reranker" not in _cache:
        s = get_settings()
        backend = s.rerank_backend
        if backend == "fake":
            from .fake import FakeReranker

            rr: Reranker = FakeReranker()
        elif backend == "voyage":
            from .governed import CachedReranker, GovernedReranker
            from .http import is_rate_limited
            from .voyage import VoyageReranker

            key = _require_key("VOYAGE_API_KEY", s.voyage_api_key)
            limits, gov = _governor(s, "rerank", is_rate_limited)
            rr = GovernedReranker(VoyageReranker(key, s.voyage_rerank_model), gov, limits=limits)
            if s.api_cache:
                rr = CachedReranker(rr, get_disk_cache(), s.voyage_rerank_model)
        else:
            raise ValueError(f"unknown rerank provider {backend!r} (voyage | fake)")
        log.info("reranker ready: %s", backend)
        _cache["reranker"] = rr
    return _cache["reranker"]  # type: ignore[return-value]
