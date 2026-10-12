from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or `.env` (names are case-insensitive).

    Model IDs live here so providers can be swapped without touching the rest of the code.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://docsrag:docsrag@localhost:5433/docsrag"

    # One value for BOTH roles (mainly `fake` for offline runs, or `gemini`). Empty = each role uses
    # its own default: Voyage for embeddings, Groq for chat. EMBED_PROVIDER / LLM_PROVIDER win.
    provider: str = ""
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3.1-flash-lite"
    gemini_embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768

    embed_batch_size: int = 64
    worker_concurrency: int = 4
    job_max_attempts: int = 5
    job_visibility_timeout_s: int = 300

    # Which vendor serves each role. Empty = PROVIDER, else the role default. Mix freely, e.g.
    # EMBED_PROVIDER=voyage + LLM_PROVIDER=groq. Values: gemini | voyage | groq | openai_compat | fake
    embed_provider: str = ""
    llm_provider: str = ""
    # Reranker: voyage | fake. Empty = voyage, or fake when embeddings are fake (offline runs).
    rerank_provider: str = ""
    # How many first-stage candidates the reranker sees, and which retriever the API uses.
    rerank_candidates: int = 20
    retrieval_mode: str = "vector"  # vector | hybrid | rerank
    # Phase 3 agent loop (retrieve -> grade -> plan -> retrieve ... -> answer).
    agent_default: bool = False  # /ask uses the agent unless the request says otherwise
    agent_max_rounds: int = 3  # retrieval rounds, hard cap
    agent_token_budget: int = 12_000  # tokens the grading steps may spend per question
    agent_per_query: int = 3  # chunks kept per follow-up query
    agent_context_extra: int = 4  # extra chunks in the final context beyond k

    voyage_api_key: str = ""
    voyage_embed_model: str = "voyage-4-lite"
    voyage_rerank_model: str = "rerank-2.5-lite"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # Any other OpenAI-compatible endpoint (GitHub Models, OpenRouter, ...)
    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""

    # API quotas. Leave unset to get the preset for the chosen provider (QUOTA_PRESETS below);
    # set them to match what YOUR account's console shows.
    embed_rpm: int | None = None
    embed_tpm: int | None = None
    embed_rpd: int | None = None
    llm_rpm: int | None = None
    llm_tpm: int | None = None
    llm_rpd: int | None = None
    rerank_rpm: int | None = None
    rerank_tpm: int | None = None
    rerank_rpd: int | None = None
    # "text": each text in a batch counts as one request (cannot overshoot, whichever way the API
    # counts). "call": one batch = one request. Unset = provider preset.
    embed_quota_unit: str | None = None
    embed_batch_max_tokens: int = 15_000  # keeps batches small enough for the API and under TPM
    llm_output_token_reserve: int = 1_000  # assumed output size when budgeting TPM
    quota_cooldown_s: float = 65.0  # after an unexpected 429, wait out the minute window

    # Persistent cache so no input is ever paid for twice (survives DB resets)
    api_cache: bool = True
    api_cache_path: str = ".cache/api_cache.sqlite"

    # Chunking
    chunk_max_chars: int = 1800

    # Files that add noise to retrieval (changelogs, people lists, translation test pages)
    corpus_exclude: tuple[str, ...] = (
        "_llm-test.md",
        "release-notes.md",
        "fastapi-people.md",
        "management.md",
        "newsletter.md",
        "translation-banner.md",
        "external-links.md",
        "contributing.md",
        "help-fastapi.md",
    )

    @property
    def embed_backend(self) -> str:
        """Provider serving embeddings: `EMBED_PROVIDER`, else `PROVIDER`, else voyage."""
        return (self.embed_provider or self.provider or "voyage").lower()

    @property
    def llm_backend(self) -> str:
        """Provider serving chat completions: `LLM_PROVIDER`, else `PROVIDER`, else groq."""
        return (self.llm_provider or self.provider or "groq").lower()

    @property
    def rerank_backend(self) -> str:
        """Provider serving reranking (`RERANK_PROVIDER`; else voyage, or fake if embeddings are)."""
        if self.rerank_provider:
            return self.rerank_provider.lower()
        return "fake" if self.embed_backend == "fake" else "voyage"

    def backend(self, api: str) -> str:
        """Provider name for "embed", "llm" or "rerank"."""
        return {
            "embed": self.embed_backend,
            "llm": self.llm_backend,
            "rerank": self.rerank_backend,
        }[api]

    def quota(self, api: str) -> tuple[int, int, int]:
        """Effective `(rpm, tpm, rpd)` for "embed", "llm" or "rerank": explicit setting, else preset."""
        backend = self.backend(api)
        rpm, tpm, rpd, _ = QUOTA_PRESETS.get((api, backend), DEFAULT_PRESET)
        mine = (
            getattr(self, f"{api}_rpm"),
            getattr(self, f"{api}_tpm"),
            getattr(self, f"{api}_rpd"),
        )
        return tuple(m if m is not None else d for m, d in zip(mine, (rpm, tpm, rpd), strict=True))  # type: ignore[return-value]

    def usage_key(self, api: str) -> str:
        """Key of the daily-budget row, e.g. "embed:voyage": a provider switch never shares a counter."""
        return f"{api}:{self.backend(api)}"

    @property
    def embed_unit(self) -> str:
        """How embedding batches count against the request limits: "text" or "call"."""
        return (
            self.embed_quota_unit
            or QUOTA_PRESETS.get(("embed", self.embed_backend), DEFAULT_PRESET)[3]
        )


# (rpm, tpm, rpd, embed unit) per (api, provider). Conservative: when a number is unknown we assume
# the tighter one. Re-check against your own console; these are free-tier figures at time of writing.
#  - gemini: AI Studio free tier as shown on the user's account (60 / 100K / 100).
#  - voyage: no-card accounts have been 3 RPM / 10K TPM; a batch is one request, so unit "call".
#    There is no daily request cap (the free allowance is 200M tokens), hence the huge rpd.
#  - groq: gpt-oss-20b/120b free tier 30 RPM / 8K TPM / 1K RPD. A 200K tokens/day cap also exists
#    and is NOT modeled: ~25 full-size questions a day at 8K tokens each.
DEFAULT_PRESET = (60, 100_000, 100, "text")
QUOTA_PRESETS: dict[tuple[str, str], tuple[int, int, int, str]] = {
    ("embed", "gemini"): (60, 100_000, 100, "text"),
    ("llm", "gemini"): (60, 100_000, 100, "text"),
    ("embed", "voyage"): (3, 10_000, 1_000_000, "call"),
    # Rerank: assumed to follow the same no-card Voyage limits as embeddings; NOT yet confirmed
    # against the console (rerank is a separate model family there).
    ("rerank", "voyage"): (3, 10_000, 1_000_000, "call"),
    ("llm", "groq"): (30, 8_000, 1_000, "text"),
    ("llm", "openai_compat"): (15, 8_000, 150, "text"),
}


@lru_cache
def get_settings() -> Settings:
    """Return the cached `Settings`, so `.env` is parsed once per process."""
    return Settings()
