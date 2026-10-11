from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or `.env` (names are case-insensitive).

    Model IDs live here so providers can be swapped without touching the rest of the code.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://ragnarok:ragnarok@localhost:5433/ragnarok"

    provider: str = (
        "gemini"  # default vendor for both roles: gemini | fake (see EMBED_/LLM_PROVIDER)
    )
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3.1-flash-lite"
    gemini_embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768

    embed_batch_size: int = 64
    worker_concurrency: int = 4
    job_max_attempts: int = 5
    job_visibility_timeout_s: int = 300

    # Which vendor serves each role. Empty = follow PROVIDER. Mix freely, e.g.
    # EMBED_PROVIDER=voyage + LLM_PROVIDER=groq. Values: gemini | voyage | groq | openai_compat | fake
    embed_provider: str = ""
    llm_provider: str = ""

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
        """Provider serving embeddings (`EMBED_PROVIDER`, else `PROVIDER`)."""
        return (self.embed_provider or self.provider).lower()

    @property
    def llm_backend(self) -> str:
        """Provider serving chat completions (`LLM_PROVIDER`, else `PROVIDER`)."""
        return (self.llm_provider or self.provider).lower()

    def quota(self, api: str) -> tuple[int, int, int]:
        """Effective `(rpm, tpm, rpd)` for "embed" or "llm": explicit setting, else the preset."""
        backend = self.embed_backend if api == "embed" else self.llm_backend
        rpm, tpm, rpd, _ = QUOTA_PRESETS.get((api, backend), DEFAULT_PRESET)
        mine = (
            getattr(self, f"{api}_rpm"),
            getattr(self, f"{api}_tpm"),
            getattr(self, f"{api}_rpd"),
        )
        return tuple(m if m is not None else d for m, d in zip(mine, (rpm, tpm, rpd), strict=True))  # type: ignore[return-value]

    def usage_key(self, api: str) -> str:
        """Key of the daily-budget row, e.g. "embed:voyage": a provider switch never shares a counter."""
        return f"{api}:{self.embed_backend if api == 'embed' else self.llm_backend}"

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
    ("llm", "groq"): (30, 8_000, 1_000, "text"),
    ("llm", "openai_compat"): (15, 8_000, 150, "text"),
}


@lru_cache
def get_settings() -> Settings:
    """Return the cached `Settings`, so `.env` is parsed once per process."""
    return Settings()
