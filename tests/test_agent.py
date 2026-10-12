import pytest

import app.rag.agent as agent
from app.providers.base import LLMResult
from app.rag.agent import (
    GRADER_SYSTEM,
    build_grader_prompt,
    parse_verdict,
    plan_next,
    run_agent,
    select_context,
)
from app.rag.retrieve import Hit


def H(i, source=None):
    return Hit(i, source or f"d{i}.md", f"H{i}", f"text {i}", 1.0 / i)


class ScriptedLLM:
    """Replies to grader calls from a script and to answer calls with a fixed cited answer."""

    def __init__(self, verdicts, answer="Use it [1]."):
        self.verdicts = list(verdicts)
        self.answer = answer
        self.grader_prompts = []
        self.answer_calls = 0

    async def generate(self, prompt, *, system="", json=False, temperature=0.0):
        if system == GRADER_SYSTEM:
            self.grader_prompts.append(prompt)
            return LLMResult(self.verdicts.pop(0), input_tokens=100, output_tokens=10)
        self.answer_calls += 1
        return LLMResult(self.answer, input_tokens=50, output_tokens=5)


def fake_search(table):
    """search() replacement: table maps a query to its hits; unknown queries find nothing."""
    calls = []

    async def search(mode, pool, embedder, reranker, q, k, candidates):
        calls.append((q, k, mode))
        return table.get(q, [])[:k]

    search.calls = calls
    return search


V_OK = '{"sufficient": true, "missing": [], "rewrite": ""}'


def V(missing=(), rewrite=""):
    import json

    return json.dumps({"sufficient": False, "missing": list(missing), "rewrite": rewrite})


# --- pure helpers -------------------------------------------------------------------------------


def test_parse_verdict_plain_and_fenced():
    assert parse_verdict(V_OK) == {"sufficient": True, "missing": [], "rewrite": ""}
    v = parse_verdict(
        'Sure!\n```json\n{"sufficient": false, "missing": ["a", "a", " b ", 3, ""],'
        ' "rewrite": " r "}\n```'
    )
    assert v == {"sufficient": False, "missing": ["a", "b"], "rewrite": "r"}


@pytest.mark.parametrize(
    "text",
    ["", "no json", "{broken", '{"sufficient": "yes"}', "[1]", "} {", '{"missing": []}'],
)
def test_parse_verdict_rejects_unusable(text):
    assert parse_verdict(text) is None


def test_parse_verdict_limits_and_odd_types():
    many = [f"q{i}" for i in range(9)]
    import json

    v = parse_verdict(
        json.dumps({"sufficient": False, "missing": many + ["x" * 500], "rewrite": 5})
    )
    assert v["missing"] == ["q0", "q1", "q2"] and v["rewrite"] == ""
    v = parse_verdict(json.dumps({"sufficient": False, "missing": ["x" * 500]}))
    assert len(v["missing"][0]) == 200


def test_grader_prompt_truncates_and_handles_empty():
    p = build_grader_prompt("Q?", [Hit(1, "a.md", "A", "z" * 2000, 1.0)])
    assert p.startswith("Question: Q?") and "[1] (a.md :: A)" in p
    assert p.count("z") == agent.GRADER_PASSAGE_CHARS
    assert "no passages were found" in build_grader_prompt("Q?", [])


def test_select_context_round_robin_dedupes_and_limits():
    runs = [{"hits": [H(1), H(2), H(3)]}, {"hits": [H(9), H(2)]}, {"hits": []}]
    assert [h.chunk_id for h in select_context(runs, 10)] == [1, 9, 2, 3]
    assert [h.chunk_id for h in select_context(runs, 2)] == [1, 9]
    assert select_context([], 5) == []


def test_plan_next_actions():
    v = {"missing": ["A", "B"], "rewrite": "R"}
    assert plan_next(v, [], "Q", False) == ("decompose", ["A", "B"])
    assert plan_next(v, ["a"], "Q", False) == ("decompose", ["B"])  # case-insensitive, no repeats
    assert plan_next(v, ["a", "b"], "Q", False) == ("rewrite", ["R"])
    assert plan_next(v, ["a", "b", "r"], "Q", False) == ("widen", ["Q"])
    assert plan_next(v, ["a", "b", "r"], "Q", True) == ("none", [])
    assert plan_next({"missing": [], "rewrite": ""}, [], "Q", False) == ("widen", ["Q"])


# --- the loop -----------------------------------------------------------------------------------


async def go(monkeypatch, table, llm, **kw):
    s = fake_search(table)
    monkeypatch.setattr(agent, "search", s)
    q = kw.pop("question", "Q?")
    return s, await run_agent(None, None, llm, None, q, **kw)


async def test_sufficient_first_round_costs_one_grade_and_one_answer(monkeypatch):
    llm = ScriptedLLM([V_OK])
    s, res = await go(monkeypatch, {"Q?": [H(1), H(2)]}, llm)
    assert res.stop_reason == "sufficient" and res.rounds == 1
    assert llm.answer_calls == 1 and len(llm.grader_prompts) == 1
    assert (res.input_tokens, res.output_tokens) == (150, 15)
    assert [h.chunk_id for h in res.context] == [1, 2]
    assert res.answer.citations and res.queries == ["Q?"]
    assert [t["step"] for t in res.trace] == ["retrieve", "grade", "answer"]
    assert s.calls == [("Q?", 5, "vector")]


async def test_insufficient_decomposes_then_answers(monkeypatch):
    table = {"Q?": [H(1), H(2)], "part A": [H(10), H(1)], "part B": [H(20)]}
    llm = ScriptedLLM([V(missing=["part A", "part B"]), V_OK])
    s, res = await go(monkeypatch, table, llm, mode="hybrid", per_query=2)
    assert res.stop_reason == "sufficient" and res.rounds == 2
    assert res.queries == ["Q?", "part A", "part B"]
    assert [c[1] for c in s.calls] == [5, 2, 2] and {c[2] for c in s.calls} == {"hybrid"}
    assert [h.chunk_id for h in res.context] == [1, 10, 20, 2]
    actions = [t["action"] for t in res.trace if t["step"] == "plan"]
    assert actions == ["decompose"]


async def test_rewrite_is_used_when_nothing_specific_is_missing(monkeypatch):
    table = {"Q?": [H(1)], "better words": [H(2)]}
    llm = ScriptedLLM([V(rewrite="better words"), V_OK])
    _, res = await go(monkeypatch, table, llm)
    assert res.rounds == 2 and "better words" in res.queries
    assert [t["action"] for t in res.trace if t["step"] == "plan"] == ["rewrite"]


async def test_max_rounds_stops_and_still_answers(monkeypatch):
    table = {"Q?": [H(1)], "a": [H(2)], "b": [H(3)], "c": [H(4)]}
    llm = ScriptedLLM([V(missing=["a"]), V(missing=["b"]), V(missing=["c"])])
    _, res = await go(monkeypatch, table, llm, max_rounds=3)
    assert res.stop_reason == "max_rounds" and res.rounds == 3
    assert len(llm.grader_prompts) == 2 and llm.answer_calls == 1
    assert sorted(h.chunk_id for h in res.context) == [1, 2, 3]


async def test_no_progress_stops(monkeypatch):
    table = {"Q?": [H(1)], "a": [H(1)]}
    llm = ScriptedLLM([V(missing=["a"])])
    _, res = await go(monkeypatch, table, llm)
    assert res.stop_reason == "no_progress" and res.rounds == 2 and llm.answer_calls == 1


async def test_token_budget_stops(monkeypatch):
    table = {"Q?": [H(1)], "a": [H(2)]}
    llm = ScriptedLLM([V(missing=["a"])])
    _, res = await go(monkeypatch, table, llm, token_budget=100)
    assert res.stop_reason == "token_budget" and res.rounds == 2


async def test_unusable_grader_reply_fails_open(monkeypatch):
    llm = ScriptedLLM(["I think it is fine"])
    _, res = await go(monkeypatch, {"Q?": [H(1)]}, llm)
    assert res.stop_reason == "grader_error" and res.rounds == 1 and llm.answer_calls == 1
    assert res.trace[1]["sufficient"] is None


async def test_widen_once_then_no_new_idea(monkeypatch):
    table = {"Q?": [H(1), H(2), H(3), H(4), H(5), H(6), H(7), H(8)]}
    llm = ScriptedLLM([V(), V()])
    s, res = await go(monkeypatch, table, llm, per_query=3, max_rounds=5)
    # round 1 k=5; the grader has no idea, so Q? is searched again deeper (2 x per_query) ...
    assert [c[1] for c in s.calls] == [5, 6]
    assert res.stop_reason == "no_new_idea" and res.rounds == 2
    assert [t["action"] for t in res.trace if t["step"] == "plan"] == ["widen", "none"]
    assert len(res.context) == 6


async def test_follow_up_round_with_only_known_documents_stops(monkeypatch):
    # a new chunk, but from a document already in hand: the same material reworded
    table = {"Q?": [H(1, "a.md")], "a": [H(2, "a.md")]}
    llm = ScriptedLLM([V(missing=["a"])])
    _, res = await go(monkeypatch, table, llm)
    assert res.stop_reason == "no_progress" and res.rounds == 2
    assert res.trace[3]["new_chunks"] == 1 and res.trace[3]["new_sources"] == 0


def test_parse_verdict_overrides_sufficient_when_a_part_has_no_passage():
    import json

    text = json.dumps(
        {
            "parts": [
                {"part": "yield dependencies", "passage": 1},
                {"part": "background tasks usage", "passage": None},
                {"part": "exit code timing", "passage": "x"},
                {"part": "  "},
                "junk",
            ],
            "sufficient": True,
            "missing": "not a list",
        }
    )
    v = parse_verdict(text)
    assert v["sufficient"] is False
    assert v["missing"] == ["background tasks usage", "exit code timing"]


def test_parse_verdict_keeps_sufficient_when_every_part_is_covered():
    import json

    text = json.dumps(
        {"parts": [{"part": "a", "passage": 1}, {"part": "b", "passage": 2.0}], "sufficient": True}
    )
    assert parse_verdict(text)["sufficient"] is True
    assert parse_verdict('{"sufficient": true, "parts": "none"}')["sufficient"] is True


async def test_grader_override_triggers_a_decompose_round(monkeypatch):
    import json

    liar = json.dumps(
        {"parts": [{"part": "bg tasks", "passage": None}], "sufficient": True, "missing": []}
    )
    table = {"Q?": [H(1)], "bg tasks": [H(2)]}
    llm = ScriptedLLM([liar, V_OK])
    _, res = await go(monkeypatch, table, llm)
    assert res.rounds == 2 and res.queries == ["Q?", "bg tasks"] and res.stop_reason == "sufficient"
