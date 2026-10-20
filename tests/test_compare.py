import json

import pytest

from evals import compare

PROV = {
    "questions_hash": "q1",
    "corpus": {"documents": 147, "chunks": 1161},
    "settings_hash": "s1",
    "settings": {"mode": "rerank", "k": 5},
    "api_cache": False,
}


def _run(prov=PROV, rows=(), **summary):
    base = {"n_answerable": 17, "hit@k": 1.0, "full_recall@k": 0.941, "mrr": 0.941}
    return {"summary": {**base, **summary, "provenance": prov}, "rows": list(rows)}


def _write(path, data):
    path.write_text(json.dumps(data))
    return path


def test_one_question_is_noise_two_is_a_regression():
    base = _run()
    _, _, regressed = compare.compare(_run(**{"full_recall@k": 0.882}), base)  # -1 question
    assert not regressed
    rows, _, regressed = compare.compare(_run(**{"full_recall@k": 0.824}), base)  # -2 questions
    assert regressed and {r["metric"]: r["status"] for r in rows}["full_recall@k"] == "REGRESSION"


def test_directions_tolerances_and_report_only():
    base = _run(
        false_abstain=0.0,
        abstain_rate_unanswerable=1.0,
        cost_per_query_usd=0.0001,
        p95_latency_s=2.0,
        grounded_rate=0.9,
    )
    run = _run(
        false_abstain=0.0,
        abstain_rate_unanswerable=0.75,  # any drop in abstention fails
        cost_per_query_usd=0.00011,  # +10%: ok
        p95_latency_s=9.0,  # latency is never gated
        grounded_rate=0.88,  # within 0.05
    )
    rows, _, regressed = compare.compare(run, base)
    status = {r["metric"]: r["status"] for r in rows}
    assert regressed and status["abstain_rate_unanswerable"] == "REGRESSION"
    assert status["cost_per_query_usd"] == "ok" and status["p95_latency_s"] == "report"
    assert status["grounded_rate"] == "ok" and "cited_ok" not in status  # absent in both: skipped
    _, _, regressed = compare.compare(
        _run(cost_per_query_usd=0.0002), _run(cost_per_query_usd=0.0001)
    )
    assert regressed  # cost doubled
    rows, _, _ = compare.compare(_run(**{"full_recall@k": 1.0}), _run(**{"full_recall@k": 0.882}))
    assert {r["metric"]: r["status"] for r in rows}["full_recall@k"] == "better"


def test_provenance_notes():
    other = {
        **PROV,
        "questions_hash": "q2",
        "corpus": {"documents": 40, "chunks": 300},
        "settings_hash": "s2",
        "settings": {"mode": "rerank", "k": 8},
        "api_cache": True,
    }
    _, notes, _ = compare.compare(_run(prov=other), _run())
    text = "\n".join(notes)
    assert "NOT COMPARABLE" in text and "corpus differs" in text
    assert "settings changed: k" in text and "API cache on" in text
    _, notes, _ = compare.compare(_run(prov=None), _run())
    assert "no provenance" in notes[0]


def test_question_changes():
    base = _run(
        rows=[{"id": "q1", "expected_sources": ["a"], "hit": True, "full_recall": False, "rank": 1}]
    )
    run = _run(
        rows=[
            {"id": "q1", "expected_sources": ["a"], "hit": True, "full_recall": True, "rank": 3},
            {"id": "q9", "expected_sources": ["b"], "hit": True, "full_recall": True, "rank": 1},
            {"id": "u1", "expected_sources": [], "hit": False, "full_recall": False, "rank": None},
        ]
    )
    assert compare.question_changes(run, base) == [
        "q1 full_recall: False -> True",
        "q1 first right doc rank: 1 -> 3",
    ]


def test_tags_latest_and_promote(tmp_path):
    assert compare.run_tag(tmp_path / "rerank-gen-1791832103.json") == "rerank-gen"
    assert (
        compare.run_tag(tmp_path / "agent-rerank-gen-ab12cd3-1791832103.json") == "agent-rerank-gen"
    )
    assert compare.run_tag(tmp_path / "vector-nogit-1791832103.json") == "vector"
    with pytest.raises(SystemExit):
        compare.run_tag(tmp_path / "notes.json")
    for name in [
        "rerank-1791000001.json",
        "rerank-abc1234-1791000009.json",
        "rerank-gen-1791999999.json",
    ]:
        _write(tmp_path / name, _run())
    assert compare.latest("rerank", tmp_path).name == "rerank-abc1234-1791000009.json"
    with pytest.raises(SystemExit):
        compare.latest("hybrid", tmp_path)
    run = _run(rows=[{"id": "q1", "hit": True, "answer": "long text", "trace": [1]}])
    out = compare.promote(_write(tmp_path / "rerank-abc1234-1791000010.json", run), tmp_path / "bl")
    saved = json.loads(out.read_text())
    assert out.name == "rerank.json" and saved["rows"] == [{"id": "q1", "hit": True}]
    assert saved["source_run"] == "rerank-abc1234-1791000010.json"
    named = compare.promote(tmp_path / "rerank-abc1234-1791000010.json", tmp_path / "bl", "ci-x")
    assert named.name == "ci-x.json"


def test_main_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, "BASELINES", tmp_path / "bl")
    good = _write(tmp_path / "rerank-abc1234-1791000001.json", _run())
    assert compare.main([str(good)]) == 3  # no baseline yet
    assert compare.main(["--promote", str(good)]) == 0
    assert compare.main([str(good)]) == 0
    bad = _write(tmp_path / "rerank-abc1234-1791000002.json", _run(mrr=0.5))
    assert compare.main([str(bad)]) == 1
    assert "REGRESSION" in capsys.readouterr().out
    moved = _write(
        tmp_path / "rerank-abc1234-1791000003.json", _run(prov={**PROV, "questions_hash": "x"})
    )
    assert compare.main([str(moved)]) == 3
    monkeypatch.setattr(compare, "RESULTS", tmp_path)
    assert compare.main(["--latest", "rerank"]) == 3  # newest is the not-comparable one
    with pytest.raises(SystemExit):
        compare.main([])
