import json
import sys

import pytest

import app.providers as providers
import app.rag.retrieve as retrieve
from app.cache import DiskCache
from app.config import get_settings
from app.ingest.pipeline import ingest_document
from app.providers.fake import FakeReranker
from app.providers.governed import CachedReranker, GovernedReranker
from app.quota import Limits
from app.rag.retrieve import (
    Hit,
    hybrid_search,
    keyword_query,
    keyword_search,
    rerank_search,
    rrf_fuse,
    search,
)
from evals import run_eval

DOCS = {
    "static.md": ("Static Files", "Serve static files from a directory with StaticFiles."),
    "cors.md": ("CORS", "Allow cross origin requests from a browser frontend."),
    "errors.md": ("Errors", "Raise HTTPException to return an error response with a status code."),
    "testing.md": ("Testing", "Use TestClient to call your app in tests without a server."),
}


async def ingest_all(pool, emb):
    for source, (title, body) in DOCS.items():
        await ingest_document(
            pool, emb, source=source, title=title, content=f"# {title}\n\n## Body\n\n{body}"
        )


def hit(cid, source="s.md", score=0.0):
    return Hit(cid, source, "H", f"text {cid}", score)


# ---------- pure functions
def test_keyword_query_keeps_words_and_digits_and_drops_punctuation():
    assert keyword_query("How do I use HTTPException?") == "how | do | use | httpexception"
    assert keyword_query("response_model_exclude_unset!") == "response | model | exclude | unset"
    assert keyword_query("a b a Python3") == "python3"  # 1-letter words dropped, duplicates merged
    assert keyword_query("'); DROP TABLE x; --") == "drop | table"  # nothing injectable survives
    assert keyword_query("?! ... --") == ""


def test_rrf_rewards_agreement_between_lists():
    vec = [hit(1), hit(2), hit(3)]
    kw = [hit(3), hit(2), hit(9)]
    fused = rrf_fuse([vec, kw])
    assert [h.chunk_id for h in fused] == [3, 2, 1, 9]  # 3 and 2 are in both lists
    assert fused[0].score == pytest.approx(1 / 63 + 1 / 61)  # rank 3 in one list, 1 in the other
    assert fused[1].score == pytest.approx(2 / 62)  # rank 2 in both


def test_rrf_ties_keep_first_seen_order_and_single_list_is_unchanged():
    assert [h.chunk_id for h in rrf_fuse([[hit(1)], [hit(2)]])] == [1, 2]
    assert [h.chunk_id for h in rrf_fuse([[hit(5), hit(4), hit(3)]])] == [5, 4, 3]
    assert rrf_fuse([]) == [] and rrf_fuse([[], []]) == []


def test_rrf_does_not_mutate_its_inputs():
    original = hit(1, score=0.9)
    rrf_fuse([[original]])
    assert original.score == 0.9


async def test_fake_reranker_orders_by_overlap_and_honours_top_k():
    rr = FakeReranker()
    docs = ["cats and dogs", "static files directory", "serve static files"]
    out = await rr.rerank("serve static files", docs, top_k=2)
    assert [i for i, _ in out] == [2, 1] and out[0][1] > out[1][1]
    assert len(await rr.rerank("x", docs)) == 3 and rr.calls == 2
    assert [i for i, _ in await rr.rerank("nothing matches", docs)] == [
        0,
        1,
        2,
    ]  # ties: input order


# ---------- against Postgres
async def test_keyword_search_finds_exact_identifiers(pool, emb):
    await ingest_all(pool, emb)
    hits = await keyword_search(pool, "what does HTTPException do?", n=5)
    assert [h.source for h in hits] == ["errors.md"] and hits[0].score > 0


async def test_keyword_search_handles_no_usable_words_and_no_matches(pool, emb):
    await ingest_all(pool, emb)
    assert await keyword_search(pool, "?!", n=5) == []
    assert await keyword_search(pool, "zebra", n=5) == []
    assert await keyword_search(pool, "the of and", n=5) == []  # only stop words


async def test_keyword_search_respects_the_limit(pool, emb):
    await ingest_all(pool, emb)
    assert len(await keyword_search(pool, "files status requests tests", n=2)) <= 2


async def test_hybrid_search_combines_both_rankings(pool, emb, monkeypatch):
    await ingest_all(pool, emb)
    hits = await hybrid_search(pool, emb, "serve static files from a directory", k=3)
    assert hits[0].source == "static.md" and len(hits) <= 3
    assert all(h.score > 0 for h in hits)
    # both retrievers rank static.md first, so it earns the highest possible RRF score
    assert hits[0].score == pytest.approx(2 / 61)


async def test_hybrid_search_surfaces_a_keyword_only_hit(pool, emb, monkeypatch):
    await ingest_all(pool, emb)

    async def blind_vector(*a, **k):  # the embedder "misses" entirely
        return []

    monkeypatch.setattr(retrieve, "vector_search", blind_vector)
    hits = await hybrid_search(pool, emb, "HTTPException", k=3)
    assert [h.source for h in hits] == ["errors.md"]


async def test_hybrid_search_with_empty_database_returns_nothing(pool, emb):
    assert await hybrid_search(pool, emb, "anything at all", k=3) == []


class ReverseReranker:
    """Scores the LAST candidate highest, so reranking visibly changes the order."""

    def __init__(self):
        self.seen = None

    async def rerank(self, query, documents, top_k=None):
        self.seen = (query, list(documents), top_k)
        order = list(range(len(documents) - 1, -1, -1))
        out = [(i, 1.0 - n * 0.1) for n, i in enumerate(order)]
        return out[:top_k] if top_k is not None else out


async def test_rerank_search_reorders_candidates_and_sets_scores(pool, emb):
    await ingest_all(pool, emb)
    rr = ReverseReranker()
    first_stage = await hybrid_search(pool, emb, "static files status tests", k=4, candidates=4)
    out = await rerank_search(pool, emb, rr, "static files status tests", k=2, candidates=4)
    assert [h.chunk_id for h in out] == [h.chunk_id for h in reversed(first_stage)][:2]
    assert [h.score for h in out] == [1.0, 0.9]
    query, docs, top_k = rr.seen
    assert query == "static files status tests" and top_k == 2
    assert docs[0].startswith("Body") or "\n" in docs[0]  # heading + content are both sent


async def test_rerank_search_skips_the_reranker_when_nothing_is_found(pool, emb):
    rr = ReverseReranker()
    assert await rerank_search(pool, emb, rr, "nothing here", k=3) == [] and rr.seen is None


async def test_search_dispatch_and_errors(pool, emb):
    await ingest_all(pool, emb)
    rr = FakeReranker()
    for mode in ("vector", "hybrid", "rerank"):
        out = await search(mode, pool, emb, rr, "serve static files", k=2)
        assert out and out[0].source == "static.md"
    with pytest.raises(ValueError, match="needs a reranker"):
        await search("rerank", pool, emb, None, "q")
    with pytest.raises(ValueError, match="unknown retrieval mode"):
        await search("bm25", pool, emb, rr, "q")


# ---------- eval and API wiring
@pytest.mark.parametrize("mode", ["hybrid", "rerank"])
async def test_eval_runs_the_new_modes(pool, emb, tmp_path, monkeypatch, mode):
    await ingest_all(pool, emb)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--mode", mode, "--limit", "3", "--k", "2"])
    await run_eval.main()
    (out,) = tmp_path.glob(f"{mode}-[0-9]*.json")
    data = json.loads(out.read_text())
    assert data["summary"]["mode"] == mode and len(data["rows"]) == 3


async def test_eval_retrieve_rejects_unknown_mode(pool, emb):
    with pytest.raises(SystemExit, match="choose from vector, hybrid, rerank"):
        await run_eval.retrieve("bm25", pool, emb, "q", 3)


@pytest.mark.parametrize("mode", ["hybrid", "rerank"])
async def test_api_uses_the_configured_retrieval_mode(pool, emb, monkeypatch, mode):
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    await ingest_all(pool, providers.get_embedder())
    monkeypatch.setenv("RETRIEVAL_MODE", mode)
    get_settings.cache_clear()
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            r = await c.post("/ask", json={"question": "serve static files", "k": 2})
    assert r.status_code == 200 and r.json()["retrieved"][0]["source"] == "static.md"


# ---------- governed / cached / wiring
class CountingReranker:
    def __init__(self):
        self.calls = []

    async def rerank(self, query, documents, top_k=None):
        self.calls.append((query, list(documents), top_k))
        return [(i, 1.0 / (i + 1)) for i in range(len(documents))][:top_k]


class RecordingGovernor:
    def __init__(self):
        self.calls = []

    async def call(self, fn, *, units=1, tokens=0):
        self.calls.append((units, tokens))
        return await fn()


async def test_governed_reranker_counts_one_request_and_all_tokens():
    gov, inner = RecordingGovernor(), CountingReranker()
    rr = GovernedReranker(inner, gov, limits=Limits(3, 10_000, 100))
    out = await rr.rerank("q" * 40, ["d" * 400, "e" * 400], top_k=1)
    assert out == [(0, 1.0)] and gov.calls == [(1, 14 + 134 + 134)]  # estimate: 3 chars per token


async def test_governed_reranker_drops_the_tail_that_does_not_fit_the_token_budget(caplog):
    gov, inner = RecordingGovernor(), CountingReranker()
    rr = GovernedReranker(inner, gov, limits=Limits(3, 300, 100))
    docs = ["d" * 400, "e" * 400, "f" * 400]  # 134 estimated tokens each
    await rr.rerank("q", docs)
    assert len(inner.calls[0][1]) == 2  # indexes still refer to the original list
    assert "2 of 3 candidates fit" in caplog.text
    inner.calls.clear()
    await GovernedReranker(inner, gov, limits=Limits(3, 5, 100)).rerank("q", docs)
    assert len(inner.calls[0][1]) == 1  # always keeps at least the best candidate


async def test_cached_reranker_pays_once_per_input():
    inner = CountingReranker()
    rr = CachedReranker(inner, DiskCache(":memory:"), "m")
    a = await rr.rerank("q", ["x", "y"], top_k=1)
    b = await rr.rerank("q", ["x", "y"], top_k=1)
    assert a == b == [(0, 1.0)] and len(inner.calls) == 1 and (rr.hits, rr.misses) == (1, 1)
    await rr.rerank("q", ["x", "y"], top_k=2)  # a different top_k is a different question
    await rr.rerank("q", ["x", "z"], top_k=1)
    assert len(inner.calls) == 3


@pytest.fixture
def clean_providers():
    get_settings.cache_clear()
    providers._cache.clear()
    yield
    get_settings.cache_clear()
    providers._cache.clear()


def test_reranker_defaults_follow_the_embedding_provider(monkeypatch, clean_providers):
    monkeypatch.setenv("PROVIDER", "fake")
    s = get_settings()
    assert s.rerank_backend == "fake" and s.usage_key("rerank") == "rerank:fake"
    rr = providers.get_reranker()
    assert isinstance(rr, FakeReranker) and providers.get_reranker() is rr


def test_voyage_reranker_is_governed_and_cached(monkeypatch, clean_providers):
    monkeypatch.setenv("EMBED_PROVIDER", "fake")
    monkeypatch.setenv("RERANK_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    s = get_settings()
    assert s.quota("rerank") == (3, 10_000, 1_000_000)
    rr = providers.get_reranker()
    assert isinstance(rr, CachedReranker) and isinstance(rr.inner, GovernedReranker)
    assert rr.model == "rerank-2.5-lite"


def test_voyage_reranker_without_cache_and_with_custom_limits(monkeypatch, clean_providers):
    monkeypatch.setenv("RERANK_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    monkeypatch.setenv("API_CACHE", "false")
    monkeypatch.setenv("RERANK_RPM", "30")
    assert get_settings().quota("rerank") == (30, 10_000, 1_000_000)
    assert isinstance(providers.get_reranker(), GovernedReranker)


def test_reranker_errors_are_clear(monkeypatch, clean_providers):
    monkeypatch.setenv("RERANK_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        providers.get_reranker()
    providers._cache.clear()
    monkeypatch.setenv("RERANK_PROVIDER", "cohere")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="unknown rerank provider"):
        providers.get_reranker()
