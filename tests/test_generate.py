from app.providers.base import LLMResult
from app.rag.generate import NO_ANSWER, SYSTEM, build_prompt, generate_answer, parse_citations
from app.rag.retrieve import Hit


def make_hits():
    return [Hit(1, "a.md", "A > X", "alpha text", 0.9), Hit(2, "b.md", "B", "beta text", 0.8)]


class StubLLM:
    def __init__(self, text):
        self.text = text
        self.prompt = self.system = None

    async def generate(self, prompt, *, system="", json=False, temperature=0.0):
        self.prompt, self.system = prompt, system
        return LLMResult(self.text, input_tokens=11, output_tokens=7)


def test_prompt_numbers_passages_and_includes_question():
    p = build_prompt("What is X?", make_hits())
    assert "[1] (a.md :: A > X)\nalpha text" in p
    assert "[2] (b.md :: B)\nbeta text" in p
    assert p.endswith("Question: What is X?")
    assert NO_ANSWER in SYSTEM


async def test_citations_are_deduped_sorted_and_range_checked():
    llm = StubLLM("Use X [2] and Y [1] [2]. Bogus [9] and [0].")
    ans = await generate_answer(llm, "q", make_hits())
    assert [h.source for h in ans.citations] == ["a.md", "b.md"]
    assert not ans.abstained
    assert (ans.input_tokens, ans.output_tokens) == (11, 7)
    assert ans.latency_s >= 0 and len(ans.retrieved) == 2
    assert llm.system == SYSTEM and "alpha text" in llm.prompt


async def test_abstention_is_detected_and_has_no_citations():
    ans = await generate_answer(
        StubLLM(f"  {NO_ANSWER} The docs never mention gRPC."), "q", make_hits()
    )
    assert ans.abstained and ans.citations == []


async def test_answer_without_citations():
    ans = await generate_answer(StubLLM("Some answer."), "q", make_hits())
    assert not ans.abstained and ans.citations == []


def test_parse_citations_accepts_common_bracket_variants():
    assert parse_citations("a [1] b 【2】 c [1][2]", 3) == [1, 2]
    assert parse_citations("see [1, 3] and 【2，3】 and [2,3]", 3) == [1, 2, 3]
    assert parse_citations("[ 2 ]", 3) == [2]


def test_parse_citations_ignores_non_citations_and_out_of_range():
    assert parse_citations("no refs, list[int], (1), array[0], [9]", 3) == []
    assert parse_citations("", 3) == []


async def test_fullwidth_brackets_still_count_as_a_citation():
    ans = await generate_answer(StubLLM("Use TestClient【2】."), "q", make_hits())
    assert [h.source for h in ans.citations] == ["b.md"]


def test_prompt_demands_plain_brackets_and_citations_on_partial_answers():
    assert "plain ASCII square brackets" in SYSTEM
    assert "Not found:" in SYSTEM


async def test_answer_is_logged_and_an_uncited_answer_is_flagged(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.rag.generate")
    await generate_answer(StubLLM("Use X [2]."), "q", make_hits())
    await generate_answer(StubLLM("An answer with no citation."), "q", make_hits())
    await generate_answer(StubLLM(NO_ANSWER + " missing info"), "q", make_hits())
    levels = [(r.levelname, r.getMessage()) for r in caplog.records]
    assert any("cited [2], answered" in m for _, m in levels)
    assert any("ABSTAINED" in m for _, m in levels)
    assert any(lv == "WARNING" and "cites no passage" in m for lv, m in levels)
