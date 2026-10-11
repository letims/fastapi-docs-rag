import pytest

import app.providers as providers
from app.config import Settings, get_settings
from app.providers.fake import FakeEmbedder, FakeLLM


@pytest.fixture(autouse=True)
def _clean():
    get_settings.cache_clear()
    providers._cache.clear()
    yield
    get_settings.cache_clear()
    providers._cache.clear()


def test_fake_provider_is_selected_and_cached(monkeypatch):
    monkeypatch.setenv("PROVIDER", "fake")
    monkeypatch.setenv("EMBED_DIM", "16")
    e = providers.get_embedder()
    assert isinstance(e, FakeEmbedder) and e.dim == 16
    assert providers.get_embedder() is e
    llm = providers.get_llm()
    assert isinstance(llm, FakeLLM) and providers.get_llm() is llm


def test_gemini_provider_is_selected(monkeypatch):
    import app.providers.gemini as g

    monkeypatch.setattr(g.genai, "Client", lambda api_key=None: object())
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("EMBED_DIM", "32")
    e = providers.get_embedder()
    assert isinstance(e, g.GeminiEmbedder) and e.dim == 32
    assert isinstance(providers.get_llm(), g.GeminiLLM)


async def test_fake_llm_and_embedder_behave():
    e = FakeEmbedder(8)
    assert [len(v) for v in await e.embed(["a b", "c"])] == [8, 8]
    assert (e.calls, e.texts_embedded) == (1, 2)
    res = await FakeLLM().generate("x" * 40)
    assert res.text.startswith("[fake llm]") and res.input_tokens == 10


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.embed_dim == 768 and s.chunk_max_chars == 1800 and s.job_max_attempts == 5
    assert "release-notes.md" in s.corpus_exclude
    assert get_settings() is get_settings()  # cached
