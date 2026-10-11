"""Voyage and OpenAI-compatible (Groq) adapters against httpx.MockTransport: no network, no keys."""

import json

import httpx
import pytest

import app.retry as retry_mod
from app.providers.http import is_rate_limited, is_transient, post_json
from app.providers.openai_compat import OpenAICompatLLM
from app.providers.voyage import VoyageEmbedder, VoyageReranker


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(retry_mod, "backoff_delay", lambda *a, **k: 0.0)


def client_for(*responses):
    """A client that returns the given responses in order and records the requests."""
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        out = queue.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    c = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    c.seen = seen  # type: ignore[attr-defined]
    return c


def body_of(req: httpx.Request) -> dict:
    return json.loads(req.content)


async def test_voyage_embeds_in_input_order_with_dim_and_input_type():
    c = client_for(
        httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )
    )
    e = VoyageEmbedder("k", "voyage-4-lite", 256, client=c)
    assert await e.embed(["a", "b"], kind="query") == [[1.0, 0.0], [0.0, 1.0]]
    sent = body_of(c.seen[0])
    assert sent == {
        "input": ["a", "b"],
        "model": "voyage-4-lite",
        "input_type": "query",
        "output_dimension": 256,
    }


def test_voyage_rejects_unsupported_dimension():
    with pytest.raises(ValueError, match="EMBED_DIM=1024"):
        VoyageEmbedder("k", "m", 768)


def test_voyage_builds_its_own_client_with_auth_header():
    e = VoyageEmbedder("secret", "m", 1024)
    assert e.client.headers["authorization"] == "Bearer secret"


async def test_voyage_rerank_returns_index_and_score_pairs():
    c = client_for(
        httpx.Response(
            200,
            json={
                "data": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]
            },
        )
    )
    r = VoyageReranker("k", "rerank-2.5-lite", client=c)
    assert await r.rerank("q", ["a", "b", "c"], top_k=2) == [(2, 0.9), (0, 0.1)]
    assert body_of(c.seen[0])["top_k"] == 2


async def test_voyage_rerank_omits_top_k_by_default():
    c = client_for(httpx.Response(200, json={"data": []}))
    await VoyageReranker("k", "m", client=c).rerank("q", ["a"])
    assert "top_k" not in body_of(c.seen[0])


async def test_chat_returns_text_and_usage_and_sends_system_and_json_mode():
    c = client_for(
        httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            },
        )
    )
    llm = OpenAICompatLLM("k", "m", "https://x.test/v1/", client=c)
    res = await llm.generate("p", system="s", json=True, temperature=0.3)
    assert (res.text, res.input_tokens, res.output_tokens) == ("hello", 7, 2)
    assert str(c.seen[0].url) == "https://x.test/v1/chat/completions"
    sent = body_of(c.seen[0])
    assert sent["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "p"},
    ]
    assert sent["response_format"] == {"type": "json_object"} and sent["temperature"] == 0.3


async def test_chat_without_system_json_or_usage():
    c = client_for(httpx.Response(200, json={"choices": [{"message": {"content": None}}]}))
    res = await OpenAICompatLLM("k", "m", client=c).generate("p")
    assert (res.text, res.input_tokens, res.output_tokens) == ("", 0, 0)
    sent = body_of(c.seen[0])
    assert sent["messages"] == [{"role": "user", "content": "p"}] and "response_format" not in sent


def test_chat_builds_its_own_client_with_auth_header():
    llm = OpenAICompatLLM("secret", "m")
    assert llm.client.headers["authorization"] == "Bearer secret"
    assert llm.base_url == "https://api.groq.com/openai/v1"


async def test_5xx_and_network_errors_are_retried_then_succeed():
    ok = httpx.Response(200, json={"ok": True})
    c = client_for(httpx.Response(503, text="busy"), httpx.ConnectError("boom"), ok)
    assert await post_json(c, "https://x.test/a", {}) == {"ok": True}
    assert len(c.seen) == 3


async def test_429_is_not_retried_and_is_classified_as_rate_limited():
    c = client_for(httpx.Response(429, text="slow down"))
    with pytest.raises(httpx.HTTPStatusError, match="slow down") as ei:
        await post_json(c, "https://x.test/a", {})
    assert len(c.seen) == 1 and is_rate_limited(ei.value)


async def test_other_4xx_fails_fast_with_the_body_in_the_message():
    c = client_for(httpx.Response(400, text="bad dimension"))
    with pytest.raises(httpx.HTTPStatusError, match="bad dimension") as ei:
        await post_json(c, "https://x.test/a", {})
    assert len(c.seen) == 1 and not is_rate_limited(ei.value) and not is_transient(ei.value)


def test_classifiers_ignore_unrelated_errors():
    assert not is_rate_limited(ValueError("x")) and not is_transient(ValueError("x"))
    assert is_transient(httpx.ReadTimeout("t"))
