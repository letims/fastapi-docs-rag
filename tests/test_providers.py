import pytest

import app.providers as providers
from app.config import Settings, get_settings
from app.providers.fake import FakeEmbedder, FakeLLM
from app.providers.governed import CachedEmbedder, CachedLLM, GovernedEmbedder, GovernedLLM


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
    monkeypatch.setenv("API_CACHE", "false")
    get_settings.cache_clear()
    e = providers.get_embedder()
    assert isinstance(e, GovernedEmbedder) and isinstance(e.inner, g.GeminiEmbedder) and e.dim == 32
    assert e.unit == "text"  # conservative default: every text counts as a request
    llm = providers.get_llm()
    assert isinstance(llm, GovernedLLM) and isinstance(llm.inner, g.GeminiLLM)


def test_gemini_provider_is_cached_when_enabled(monkeypatch, tmp_path):
    import app.providers.gemini as g

    monkeypatch.setattr(g.genai, "Client", lambda api_key=None: object())
    monkeypatch.setenv("PROVIDER", "gemini")
    monkeypatch.setenv("API_CACHE", "true")
    monkeypatch.setenv("API_CACHE_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.setenv("EMBED_QUOTA_UNIT", "call")
    get_settings.cache_clear()
    e, llm = providers.get_embedder(), providers.get_llm()
    assert isinstance(e, CachedEmbedder) and isinstance(e.inner, GovernedEmbedder)
    assert e.inner.unit == "call" and isinstance(llm, CachedLLM)
    assert e.cache is llm.cache is providers.get_disk_cache()  # one shared cache file


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


def _env(monkeypatch, tmp_path, **kv):
    monkeypatch.setenv("API_CACHE", "false")
    monkeypatch.setenv("EMBED_DIM", "1024")
    for k, v in kv.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()


def test_voyage_embedder_and_groq_llm_are_selected_with_their_own_presets(monkeypatch, tmp_path):
    from app.providers.openai_compat import OpenAICompatLLM
    from app.providers.voyage import VoyageEmbedder

    _env(
        monkeypatch,
        tmp_path,
        EMBED_PROVIDER="voyage",
        LLM_PROVIDER="groq",
        VOYAGE_API_KEY="v",
        GROQ_API_KEY="g",
    )
    e, llm = providers.get_embedder(), providers.get_llm()
    assert isinstance(e, GovernedEmbedder) and isinstance(e.inner, VoyageEmbedder)
    assert e.unit == "call" and e.dim == 1024  # preset: a Voyage batch is one request
    assert e.governor.limiter.rpm == 3 and e.governor.budget.api == "embed:voyage"
    assert isinstance(llm, GovernedLLM) and isinstance(llm.inner, OpenAICompatLLM)
    assert llm.governor.limiter.tpm == 8_000 and llm.governor.budget.api == "llm:groq"


def test_explicit_limits_override_the_preset(monkeypatch, tmp_path):
    _env(
        monkeypatch,
        tmp_path,
        EMBED_PROVIDER="voyage",
        VOYAGE_API_KEY="v",
        EMBED_RPM="300",
        EMBED_QUOTA_UNIT="text",
    )
    e = providers.get_embedder()
    assert e.governor.limiter.rpm == 300 and e.governor.limiter.tpm == 10_000 and e.unit == "text"


def test_openai_compat_needs_url_and_model(monkeypatch, tmp_path):
    from app.providers.openai_compat import OpenAICompatLLM

    _env(monkeypatch, tmp_path, LLM_PROVIDER="openai_compat", OPENAI_COMPAT_API_KEY="k")
    with pytest.raises(RuntimeError, match="OPENAI_COMPAT_BASE_URL"):
        providers.get_llm()
    _env(monkeypatch, tmp_path, OPENAI_COMPAT_BASE_URL="https://x.test/v1", OPENAI_COMPAT_MODEL="m")
    llm = providers.get_llm()
    assert isinstance(llm.inner, OpenAICompatLLM) and llm.inner.model == "m"


@pytest.mark.parametrize(
    "env,match",
    [
        ({"EMBED_PROVIDER": "voyage"}, "VOYAGE_API_KEY"),
        ({"LLM_PROVIDER": "groq"}, "GROQ_API_KEY"),
        ({"LLM_PROVIDER": "openai_compat"}, "OPENAI_COMPAT_API_KEY"),
    ],
)
def test_missing_key_fails_loudly(monkeypatch, tmp_path, env, match):
    _env(monkeypatch, tmp_path, **env)
    with pytest.raises(RuntimeError, match=match):
        (providers.get_embedder if "EMBED_PROVIDER" in env else providers.get_llm)()


def test_unknown_providers_are_rejected(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, EMBED_PROVIDER="nope", LLM_PROVIDER="nope")
    with pytest.raises(ValueError, match="embedding provider"):
        providers.get_embedder()
    with pytest.raises(ValueError, match="LLM provider"):
        providers.get_llm()


def test_roles_can_be_split_between_fake_and_real(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path, PROVIDER="fake", LLM_PROVIDER="groq", GROQ_API_KEY="g")
    assert isinstance(providers.get_embedder(), FakeEmbedder)
    assert isinstance(providers.get_llm(), GovernedLLM)
