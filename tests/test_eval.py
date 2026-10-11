import json
import sys

import pytest

from app.ingest.pipeline import ingest_document
from app.providers import get_embedder
from evals import run_eval


async def _seed(pool):
    emb = get_embedder()
    for source, title, body in [
        ("advanced/events.md", "Events", "Run code at startup and shutdown with lifespan."),
        ("tutorial/cors.md", "CORS", "Allow a browser frontend on another origin to call the API."),
        ("tutorial/static-files.md", "Static Files", "Serve static files from a directory."),
    ]:
        await ingest_document(pool, emb, source=source, title=title, content=f"# {title}\n\n## Body\n\n{body}")


async def test_eval_with_generation_writes_results(pool, tmp_path, monkeypatch, capsys):
    await _seed(pool)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--k", "3", "--generate"])
    await run_eval.main()
    (out,) = tmp_path.glob("vector-gen-*.json")
    s = json.loads(out.read_text())["summary"]
    assert s["n_answerable"] == 17 and s["hit@k"] > 0
    assert s["abstain_rate_unanswerable"] == 0.0  # the fake LLM never abstains
    assert s["input_tokens"] > 0 and "avg_latency_s" in s
    assert "hit@k" in capsys.readouterr().out


async def test_eval_retrieval_only_with_limit(pool, tmp_path, monkeypatch):
    await _seed(pool)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--limit", "3", "--k", "2"])
    await run_eval.main()
    (out,) = tmp_path.glob("vector-[0-9]*.json")
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 3 and "cited_ok" not in data["summary"]


async def test_unknown_mode_is_rejected(pool, emb):
    with pytest.raises(SystemExit):
        await run_eval.retrieve("hybrid-not-built-yet", pool, emb, "q", 3)


def test_rank_of_first():
    assert run_eval._rank_of_first(["b"], ["a", "b", "c"]) == 2
    assert run_eval._rank_of_first(["z"], ["a", "b"]) is None
    assert run_eval._rank_of_first([], ["a"]) is None
