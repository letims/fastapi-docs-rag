"""Shared fixtures.

DB-backed tests need Postgres + pgvector:
  1. TEST_DATABASE_URL=<conninfo>  (e.g. the compose DB: postgresql://docsrag:docsrag@localhost:5433/docsrag).
     WARNING: the app tables in that database are DROPPED on every test.
  2. Otherwise a throwaway Postgres is started with the `pgserver` dev dependency (no Docker).
If neither works the DB tests ERROR on purpose: silently skipping them would let the coverage gate
pass or fail for the wrong reasons."""

import os

# Before any `app` import: tests must never report to the real Sentry project.
os.environ["SENTRY_DSN"] = ""

import pytest  # noqa: E402

import app.db as db
import app.providers as providers
from app.config import Settings, get_settings
from app.providers.fake import FakeEmbedder

DIM = 64


def pytest_collection_modifyitems(items):
    """Mark every test that needs Postgres as `db`, so `-m "not db"` is a ~5 s unit run (Stop hook)."""
    for item in items:
        if "db_url" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.db)


@pytest.fixture(autouse=True)
def _ignore_dotenv(monkeypatch):
    """Tests must not depend on (or leak) the developer's real `.env`: keys, providers, limits."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def db_url(tmp_path_factory):
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        yield url
        return
    try:
        import pgserver
    except ImportError:
        pytest.fail("No test database: set TEST_DATABASE_URL or run `uv sync` (installs pgserver).")
    srv = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="delete")
    yield srv.get_uri()
    srv.cleanup()


@pytest.fixture
async def pool(db_url, monkeypatch):
    """Fresh schema on every test, with the offline fake provider (no API key, deterministic)."""
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PROVIDER", "fake")
    monkeypatch.setenv("EMBED_DIM", str(DIM))
    get_settings.cache_clear()
    providers._cache.clear()
    db._pool = None
    p = await db.open_pool()
    async with p.connection() as c:
        await c.execute("DROP TABLE IF EXISTS chunks, documents, ingest_jobs, api_usage CASCADE")
    await db.init_schema(p)
    yield p
    await db.close_pool()
    get_settings.cache_clear()
    providers._cache.clear()


@pytest.fixture
def emb():
    return FakeEmbedder(DIM)
