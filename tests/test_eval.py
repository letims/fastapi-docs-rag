import json
import sys
from pathlib import Path

import jsonschema
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
        await ingest_document(
            pool, emb, source=source, title=title, content=f"# {title}\n\n## Body\n\n{body}"
        )


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
    out = capsys.readouterr().out
    assert "hit@k" in out
    assert "[1/" in out and "left" in out  # progress counter: [i/N] ... elapsed, ETA


async def test_eval_retrieval_only_with_limit(pool, tmp_path, monkeypatch):
    await _seed(pool)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--limit", "3", "--k", "2"])
    await run_eval.main()
    (out,) = [p for p in tmp_path.glob("vector-*.json") if "-gen-" not in p.name]
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 3 and "cited_ok" not in data["summary"]
    schema = json.loads(
        (Path(run_eval.__file__).parent / "schemas/eval-run.schema.json").read_text()
    )
    jsonschema.validate(data, schema)  # what run_eval writes must match what compare.py relies on
    prov = data["summary"]["provenance"]
    assert prov["corpus"] == {"documents": 3, "chunks": 3}
    assert len(prov["settings_hash"]) == 12 and prov["settings"]["mode"] == "vector"
    assert prov["prices_as_of"] is None and prov["settings"]["llm"] is None  # retrieval only
    assert data["summary"]["retrieval_p95_s"] >= data["summary"]["retrieval_p50_s"] >= 0


async def test_eval_with_judge_scores_groundedness(pool, tmp_path, monkeypatch):
    """The fake LLM's reply is not judge JSON: grounded must be None (unknown), never a pass."""
    await _seed(pool)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--k", "2", "--judge", "--limit", "2"])
    await run_eval.main()
    (out,) = tmp_path.glob("vector-gen-*.json")
    data = json.loads(out.read_text())
    s = data["summary"]
    assert s["grounded_rate"] is None and s["judged"] == "0/2" and s["judge_tokens"] > 0
    assert s["cost_per_query_usd"] == 0.0  # fake:fake is priced at zero
    assert all(r["grounded"] is None for r in data["rows"])
    assert s["provenance"]["settings"]["llm"] == "fake:fake"


def test_percentile_and_git_helpers(monkeypatch):
    assert run_eval._pct([], 0.5) is None
    assert run_eval._pct([3.0, 1.0, 2.0], 0.5) == 2.0
    assert run_eval._pct([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0

    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(run_eval.subprocess, "run", boom)
    assert run_eval._git("rev-parse", "HEAD") == ""


async def test_unknown_mode_is_rejected(pool, emb):
    with pytest.raises(SystemExit):
        await run_eval.retrieve("hybrid-not-built-yet", pool, emb, "q", 3)


def test_rank_of_first():
    assert run_eval._rank_of_first(["b"], ["a", "b", "c"]) == 2
    assert run_eval._rank_of_first(["z"], ["a", "b"]) is None
    assert run_eval._rank_of_first([], ["a"]) is None


async def test_eval_stops_cleanly_when_the_quota_runs_out(pool, tmp_path, monkeypatch, capsys):
    from app.providers.base import LLMResult
    from app.quota import QuotaExhausted

    await _seed(pool)

    class FlakyQuotaLLM:
        """Answers 3 questions, then the daily budget is gone."""

        def __init__(self):
            self.calls = 0

        async def generate(self, prompt, *, system="", json=False, temperature=0.0):
            self.calls += 1
            if self.calls > 3:
                raise QuotaExhausted("llm", "daily budget of 100 used up")
            return LLMResult("answer [1]", 10, 5, cached=self.calls == 1)

    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(run_eval, "get_llm", lambda: FlakyQuotaLLM())
    monkeypatch.setattr(sys, "argv", ["run_eval", "--k", "3", "--generate"])
    with pytest.raises(SystemExit) as exit_info:
        await run_eval.main()
    assert exit_info.value.code == 2
    (out,) = tmp_path.glob("vector-gen-*.json")
    data = json.loads(out.read_text())
    assert len(data["rows"]) == 3  # the finished questions are scored and kept
    assert "used up" in data["summary"]["interrupted"]
    assert data["summary"]["llm_api_requests"] == 2  # one of the three came from the cache
    assert "STOPPED before q04" in capsys.readouterr().out


async def test_agent_eval_records_the_loop_and_supports_only(pool, tmp_path, monkeypatch):
    await _seed(pool)
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--k", "3", "--agent", "--only", "q01,q02,nope"])
    await run_eval.main()
    (out,) = tmp_path.glob("agent-vector-gen-*.json")
    data = json.loads(out.read_text())
    s = data["summary"]
    assert s["agent"] is True and len(data["rows"]) == 2
    # the fake LLM cannot grade, so every question fails open after one round
    assert s["stop_reasons"] == {"grader_error": 2} and s["avg_rounds"] == 1.0
    assert s["tokens_per_query"] > 0
    assert {"rounds", "stop_reason", "queries", "trace"} <= set(data["rows"][0])


async def test_only_with_no_match_exits(pool, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eval", "--only", "zzz"])
    with pytest.raises(SystemExit):
        await run_eval.main()
