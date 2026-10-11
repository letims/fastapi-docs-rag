import pytest
from httpx import ASGITransport, AsyncClient

from app.ingest.pipeline import ingest_document
from app.main import app
from app.providers import get_embedder


@pytest.fixture
async def client(pool):
    async with app.router.lifespan_context(app):  # ASGITransport doesn't run lifespan itself
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json() == {"ok": True}


async def test_documents_are_enqueued_once(client):
    docs = [
        {"source": "a.md", "title": "A", "content": "# A\n\ntext"},
        {"source": "b.md", "title": "B", "content": "# B\n\ntext"},
    ]
    r = await client.post("/documents", json=docs)
    assert r.status_code == 202 and r.json() == {"received": 2, "enqueued": 2}
    assert (await client.post("/documents", json=docs)).json()["enqueued"] == 0
    stats = (await client.get("/ingest/stats")).json()
    assert stats["jobs"] == {"pending": 2} and stats["documents"] == 0 and stats["chunks"] == 0


async def test_ask_validates_input(client):
    assert (await client.post("/ask", json={"question": "hi"})).status_code == 422
    assert (
        await client.post("/ask", json={"question": "valid question", "k": 0})
    ).status_code == 422


async def test_ask_end_to_end_with_fake_provider(client, pool):
    emb = get_embedder()
    for source, title, body in [
        ("static.md", "Static Files", "Serve static files from a directory with StaticFiles."),
        ("cors.md", "CORS", "Allow cross origin requests from a browser frontend."),
    ]:
        await ingest_document(
            pool, emb, source=source, title=title, content=f"# {title}\n\n## Body\n\n{body}"
        )
    r = await client.post("/ask", json={"question": "how do I serve static files", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"][0]["source"] == "static.md"
    assert body["answer"].startswith("[fake llm]") and body["abstained"] is False
    assert body["citations"] == [] and body["usage"]["input_tokens"] > 0
    assert body["latency_s"] >= 0


async def test_exhausted_quota_becomes_http_429_with_retry_after(client, monkeypatch):
    import app.main as main_mod
    from app.quota import QuotaExhausted

    class Boom:
        async def generate(self, *a, **k):
            raise QuotaExhausted("llm", "daily budget of 100 used up")

    monkeypatch.setattr(main_mod, "get_llm", lambda: Boom())
    r = await client.post("/ask", json={"question": "how do I serve static files"})
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "quota_exhausted" and "used up" in body["detail"]
    assert 0 < body["retry_after_s"] <= 25 * 3600
    assert r.headers["retry-after"] == str(body["retry_after_s"])
