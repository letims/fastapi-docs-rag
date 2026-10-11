"""The Gemini adapter against a fake client: no network, no key."""

from types import SimpleNamespace

import pytest
from google.genai import errors

import app.providers.gemini as g
import app.retry as retry_mod


class FakeModels:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def _next(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    async def generate_content(self, **kw):
        return await self._next(**kw)

    async def embed_content(self, **kw):
        return await self._next(**kw)


@pytest.fixture
def install(monkeypatch):
    def _install(*outcomes):
        models = FakeModels(*outcomes)
        client = SimpleNamespace(aio=SimpleNamespace(models=models))
        monkeypatch.setattr(g.genai, "Client", lambda api_key=None: client)
        monkeypatch.setattr(retry_mod, "backoff_delay", lambda *a, **k: 0.0)
        return models

    return _install


def api_error(code):
    return errors.APIError(code, {"error": {"message": "nope", "status": "X"}})


def reply(text="hi", usage=True):
    u = SimpleNamespace(prompt_token_count=5, candidates_token_count=3) if usage else None
    return SimpleNamespace(text=text, usage_metadata=u)


async def test_generate_returns_text_and_usage(install):
    models = install(reply("hello"))
    res = await g.GeminiLLM("k", "m").generate("p", system="s", json=True, temperature=0.2)
    assert (res.text, res.input_tokens, res.output_tokens) == ("hello", 5, 3)
    cfg = models.calls[0]["config"]
    assert cfg.response_mime_type == "application/json" and cfg.system_instruction == "s"
    assert models.calls[0]["model"] == "m" and models.calls[0]["contents"] == "p"


async def test_generate_tolerates_missing_text_and_usage(install):
    install(reply(text=None, usage=False))
    res = await g.GeminiLLM("k", "m").generate("p")
    assert (res.text, res.input_tokens, res.output_tokens) == ("", 0, 0)


@pytest.mark.parametrize("code", [429, 500, 503])
async def test_transient_errors_are_retried(install, code):
    models = install(api_error(code), api_error(code), reply("ok"))
    res = await g.GeminiLLM("k", "m").generate("p")
    assert res.text == "ok" and len(models.calls) == 3


async def test_client_errors_fail_fast(install):
    models = install(api_error(400), reply("never"))
    with pytest.raises(errors.APIError):
        await g.GeminiLLM("k", "m").generate("p")
    assert len(models.calls) == 1


async def test_retries_eventually_give_up(install):
    models = install(*[api_error(429)] * 6)
    with pytest.raises(errors.APIError):
        await g.GeminiLLM("k", "m").generate("p")
    assert len(models.calls) == 6


def test_retryable_ignores_non_api_errors():
    assert g._retryable(ValueError("bug")) is False


async def test_embeddings_are_normalized_and_dimension_requested(install):
    models = install(SimpleNamespace(embeddings=[SimpleNamespace(values=[3.0, 4.0]), SimpleNamespace(values=[0.0, 0.0])]))
    vecs = await g.GeminiEmbedder("k", "em", dim=2).embed(["a", "b"], kind="query")
    assert vecs[0] == pytest.approx([0.6, 0.8])
    assert vecs[1] == [0.0, 0.0]  # zero vector must not divide by zero
    assert models.calls[0]["config"].output_dimensionality == 2
    assert models.calls[0]["contents"] == ["a", "b"] and models.calls[0]["model"] == "em"
