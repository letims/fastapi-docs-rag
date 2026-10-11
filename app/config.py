from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or `.env` (names are case-insensitive).

    Model IDs live here so providers can be swapped without touching the rest of the code.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://ragnarok:ragnarok@localhost:5433/ragnarok"

    provider: str = "gemini"  # "gemini" | "fake"
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-3.1-flash-lite"
    gemini_embed_model: str = "gemini-embedding-2"
    embed_dim: int = 768

    embed_batch_size: int = 64
    worker_concurrency: int = 4
    job_max_attempts: int = 5
    job_visibility_timeout_s: int = 300

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


@lru_cache
def get_settings() -> Settings:
    """Return the cached `Settings`, so `.env` is parsed once per process."""
    return Settings()
