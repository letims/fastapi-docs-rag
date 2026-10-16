"""Phase 3: an agentic RAG loop. Retrieve, check whether the evidence is enough, and if it is not,
decide what to do about it before answering.

    retrieve -> grade -+-> answer                       (evidence covers the question)
                       +-> plan -> retrieve -> ...       (decompose / rewrite / widen, then again)

The loop is a LangGraph state graph, but every call inside it goes through this project's own
providers (`get_llm`, `get_embedder`, `get_reranker`), so the quota governor, the disk cache and
the daily budget apply exactly as in the plain pipeline. LangGraph only draws the flowchart.

Hard limits stop it from looping or spending without bound: a maximum number of retrieval rounds,
a token budget for the extra LLM calls, and a no-progress rule (a follow-up round that brings no document not seen before
ends the loop). Whatever stops it, the last step is the same `generate_answer` as Phase 1, which
answers what the evidence supports with citations, says what is "Not found:", or abstains.
"""

import json
import logging
import operator
import time
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from app.providers.base import LLM, Embedder, Reranker
from app.rag.generate import Answer, generate_answer
from app.rag.retrieve import Hit, search

log = logging.getLogger(__name__)

MAX_SUBQUERIES = 3  # per planning step
MAX_QUERY_CHARS = 200
GRADER_PASSAGE_CHARS = 700  # how much of each passage the grader reads

GRADER_SYSTEM = """You check whether numbered documentation passages contain what is needed to \
answer a question. Reply with ONLY a JSON object and no other text:
{"parts": [{"part": "one distinct thing the question asks for", "passage": <number of the best passage, or null>}],
 "sufficient": true or false,
 "missing": ["a short, self-contained search query for each part with no passage"],
 "rewrite": "one better search query, or an empty string"}
Rules:
- First split the question into its distinct parts (for example "can X be combined with Y" has a part \
for X, a part for Y, and one for how they interact). Then find a passage for each part.
- A passage covers a part only if that passage is about it: its source file or heading names the \
topic, or its text states the answer. A passing mention of the topic inside a passage about \
something else does NOT cover it; use null.
- sufficient is true only if EVERY part has a passage.
- Each "missing" query must make sense on its own: no pronouns, include the product terms. At most 3.
- If the passages are about a different topic, leave "missing" empty and give a "rewrite" that uses \
the words the documentation itself would use.
- Never answer the question. Never add text outside the JSON."""


class AgentState(TypedDict, total=False):
    """What the graph carries from step to step."""

    question: str
    queries: list[str]  # queries to run in the next retrieval step
    runs: Annotated[list[dict], operator.add]  # every retrieval: {"query", "hits"}
    tried: Annotated[list[str], operator.add]  # every query already run (never repeated)
    trace: Annotated[list[dict], operator.add]  # what happened, for people to read
    rounds: int
    new_chunks: int  # chunks the last retrieval step added to the evidence
    new_sources: int  # documents among them that no earlier step had returned
    widened: bool
    tokens_in: int
    tokens_out: int
    verdict: dict | None
    stop_reason: str
    answer: Answer


@dataclass
class AgentResult:
    """The outcome of one agent run, with enough detail to evaluate and to explain it."""

    answer: Answer
    rounds: int
    stop_reason: (
        str  # sufficient | max_rounds | token_budget | no_progress | no_new_idea | grader_error
    )
    input_tokens: int  # every LLM call in the loop, grader included
    output_tokens: int
    latency_s: float  # wall time of the whole loop
    trace: list[dict]
    queries: list[str]
    context: list[Hit]  # the passages the answer was written from, in order


def parse_verdict(text: str) -> dict | None:
    """Parse the grader's JSON reply into a verdict dict, or None if it is unusable.

    The verdict has `parts` (each distinct thing the question asks, with the passage number that
    covers it or None), `sufficient`, `missing` (self-contained sub-queries) and `rewrite`.
    `overridden` is True when the model said sufficient but one of its own parts has no passage:
    the parser then flips `sufficient` to False and turns those parts into `missing` queries.

    Tolerates prose or code fences around the JSON (models add them even when told not to). Only
    the first `{` to the last `}` is read. A reply without a boolean `sufficient` is unusable.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("sufficient"), bool):
        return None
    missing = raw.get("missing")
    rewrite = raw.get("rewrite")
    queries = [
        m.strip()[:MAX_QUERY_CHARS]
        for m in (missing if isinstance(missing, list) else [])
        if isinstance(m, str) and m.strip()
    ]
    sufficient = raw["sufficient"]
    # Guard: the model must not call the evidence sufficient while its own part-by-part check
    # found a part with no passage. Those parts become the queries to search.
    raw_parts = raw.get("parts")
    parts = [
        {
            "part": p["part"].strip()[:MAX_QUERY_CHARS],
            "passage": p["passage"] if isinstance(p.get("passage"), int | float) else None,
        }
        for p in (raw_parts if isinstance(raw_parts, list) else [])
        if isinstance(p, dict) and isinstance(p.get("part"), str) and p["part"].strip()
    ]
    uncovered = [p["part"] for p in parts if p["passage"] is None]
    overridden = bool(sufficient and uncovered)
    if overridden:
        sufficient, queries = False, queries + uncovered
    return {
        "parts": parts,
        "overridden": overridden,
        "sufficient": sufficient,
        "missing": list(dict.fromkeys(queries))[:MAX_SUBQUERIES],
        "rewrite": rewrite.strip()[:MAX_QUERY_CHARS] if isinstance(rewrite, str) else "",
    }


def build_grader_prompt(question: str, hits: list[Hit]) -> str:
    """Format the question and the evidence collected so far for the grader."""
    if not hits:
        passages = "(no passages were found)"
    else:
        passages = "\n\n".join(
            f"[{i}] ({h.source} :: {h.heading})\n{h.content[:GRADER_PASSAGE_CHARS]}"
            for i, h in enumerate(hits, 1)
        )
    return f"Question: {question}\n\nPassages:\n{passages}"


def select_context(runs: list[dict], limit: int) -> list[Hit]:
    """Pick the passages for the answer by taking turns between the retrievals.

    Round robin (best hit of each retrieval, then the second best, ...) so that every part of a
    multi-part question is represented, instead of the first retrieval filling the whole context.
    Duplicates are dropped; the first run (the original question) goes first.
    """
    out: list[Hit] = []
    seen: set[int] = set()
    depth = max((len(r["hits"]) for r in runs), default=0)
    for i in range(depth):
        for r in runs:
            if i < len(r["hits"]):
                h = r["hits"][i]
                if h.chunk_id not in seen and len(out) < limit:
                    seen.add(h.chunk_id)
                    out.append(h)
    return out


def plan_next(
    verdict: dict, tried: list[str], question: str, widened: bool
) -> tuple[str, list[str]]:
    """Turn the grader's diagnosis into an action and the queries to run.

    - **decompose**: the grader named parts of the question the evidence does not cover; search
      each part on its own.
    - **rewrite**: nothing specific is missing but the evidence is off topic; search again with a
      better-worded query.
    - **widen**: no new idea; look deeper for the original question, once.
    - **none**: nothing new left to try, so the loop ends.

    Queries already tried are never repeated.
    """
    done = {t.lower() for t in tried}
    missing = [m for m in verdict["missing"] if m.lower() not in done]
    if missing:
        return "decompose", missing
    rewrite = verdict["rewrite"]
    if rewrite and rewrite.lower() not in done:
        return "rewrite", [rewrite]
    if not widened:
        return "widen", [question]
    return "none", []


async def run_agent(
    pool: AsyncConnectionPool,
    embedder: Embedder,
    llm: LLM,
    reranker: Reranker | None,
    question: str,
    *,
    mode: str = "vector",
    k: int = 5,
    candidates: int = 20,
    max_rounds: int = 3,
    token_budget: int = 12_000,
    per_query: int = 3,
    extra_context: int = 4,
) -> AgentResult:
    """Answer a question with the retrieve, grade, plan, retrieve, answer loop.

    Args:
        pool: Open connection pool.
        embedder: Provider used to embed queries.
        llm: Provider for the grader and the final answer.
        reranker: Needed only when `mode` is "rerank".
        question: The user's question.
        mode: First-stage retriever for every search: vector, hybrid or rerank.
        k: Passages retrieved for the original question.
        candidates: First-stage depth for hybrid and rerank.
        max_rounds: Most retrieval steps (the first one included).
        token_budget: Most LLM tokens the grading steps may spend; the final answer is always made.
        per_query: Passages kept from each follow-up query.
        extra_context: How many more passages than `k` the answer may be written from.

    Returns:
        An `AgentResult`; `answer` is the same type Phase 1 returns.
    """
    t0 = time.perf_counter()

    async def retrieve_node(state: AgentState) -> dict:
        """Run every pending query through the configured retriever and record what was new.

        Round 1 retrieves `k` passages for the question; later rounds keep `per_query` per follow-up
        query (double after a "widen" plan). `new_chunks` / `new_sources` feed the no-progress rule.
        """
        rounds = state.get("rounds", 0) + 1
        queries = state["queries"]
        runs, tried, before = [], [], {h.chunk_id for r in state.get("runs", []) for h in r["hits"]}
        widen = state.get("widened", False) and rounds > 1
        log.info(
            "agent round %d: %d quer%s%s: %s",
            rounds,
            len(queries),
            "y" if len(queries) == 1 else "ies",
            ", widened" if widen else "",
            [q[:60] for q in queries],
        )
        for q in queries:
            n = k if rounds == 1 else per_query * (2 if widen else 1)
            hits = await search(mode, pool, embedder, reranker, q, n, candidates)
            runs.append({"query": q, "hits": hits})
            tried.append(q)
        added = {h.chunk_id for r in runs for h in r["hits"]} - before
        seen_sources = {h.source for r in state.get("runs", []) for h in r["hits"]}
        new_sources = {h.source for r in runs for h in r["hits"]} - seen_sources
        log.info(
            "agent round %d: found %d new chunks, %d from documents not seen before",
            rounds,
            len(added),
            len(new_sources),
        )
        step = {
            "step": "retrieve",
            "round": rounds,
            "queries": queries,
            "new_chunks": len(added),
            "new_sources": len(new_sources),
            "sources": list(dict.fromkeys(h.source for r in runs for h in r["hits"])),
        }
        return {
            "rounds": rounds,
            "runs": runs,
            "tried": tried,
            "new_chunks": len(added),
            "new_sources": len(new_sources),
            "trace": [step],
        }

    def route_after_retrieve(state: AgentState) -> str:
        """Hard stops come first (max rounds, no new document, token budget); otherwise grade."""
        used = state.get("tokens_in", 0) + state.get("tokens_out", 0)
        if state["rounds"] >= max_rounds:
            log.info("agent stop: reached the limit of %d rounds", max_rounds)
            return "stop_max_rounds"
        # Follow-up rounds must bring a document not seen before: more chunks from documents
        # already in hand are usually the same material reworded (measured on q17).
        if state["rounds"] > 1 and state["new_sources"] == 0:
            log.info("agent stop: round %d brought no document not seen before", state["rounds"])
            return "stop_no_progress"
        if used > token_budget:
            log.info("agent stop: %d tokens used, over the budget of %d", used, token_budget)
            return "stop_token_budget"
        return "grade"

    async def grade_node(state: AgentState) -> dict:
        """Ask the LLM whether the evidence covers every part of the question (one JSON verdict).

        An unusable reply fails open: stop with `grader_error` and answer with what was found.
        """
        evidence = select_context(state["runs"], k + extra_context)
        res = await llm.generate(
            build_grader_prompt(state["question"], evidence), system=GRADER_SYSTEM
        )
        verdict = parse_verdict(res.text)
        update: dict = {
            "tokens_in": state.get("tokens_in", 0) + res.input_tokens,
            "tokens_out": state.get("tokens_out", 0) + res.output_tokens,
            "verdict": verdict,
        }
        if verdict is None:
            log.warning("agent: unusable grader reply, answering with what was found")
            update["trace"] = [{"step": "grade", "sufficient": None, "note": "unusable reply"}]
            update["stop_reason"] = "grader_error"
        else:
            update["trace"] = [
                {
                    "step": "grade",
                    "sufficient": verdict["sufficient"],
                    "parts": verdict["parts"],  # what the grader said covers each part
                    "overridden": verdict["overridden"],  # parser turned "sufficient" into "no"
                    "missing": verdict["missing"],
                    "rewrite": verdict["rewrite"],
                }
            ]
            uncovered = [p["part"][:60] for p in verdict["parts"] if p["passage"] is None]
            log.info(
                "agent grade: %s%s; uncovered parts: %s",
                "evidence is sufficient" if verdict["sufficient"] else "evidence is NOT sufficient",
                " (overridden: the grader said sufficient but left a part without a passage)"
                if verdict["overridden"]
                else "",
                uncovered or "none",
            )
            if verdict["sufficient"]:
                update["stop_reason"] = "sufficient"
        return update

    def route_after_grade(state: AgentState) -> str:
        """Plan another round only when the grader said the evidence is not enough."""
        return "plan" if state.get("verdict") and not state["verdict"]["sufficient"] else "answer"

    async def plan_node(state: AgentState) -> dict:
        """Turn the grader's verdict into the next queries (decompose, rewrite or widen), or stop."""
        action, queries = plan_next(
            state["verdict"], state.get("tried", []), state["question"], state.get("widened", False)
        )
        log.info(
            "agent plan: %s%s",
            action,
            f" -> {[q[:60] for q in queries]}" if queries else " (no new query to try; answering)",
        )
        update: dict = {
            "queries": queries,
            "trace": [{"step": "plan", "action": action, "queries": queries}],
        }
        if action == "widen":
            update["widened"] = True
        if action == "none":
            update["stop_reason"] = "no_new_idea"
        return update

    def route_after_plan(state: AgentState) -> str:
        """Retrieve again if the plan produced queries; otherwise go straight to the answer."""
        return "retrieve" if state["queries"] else "answer"

    def stopper(reason: str):
        """Build a node that records why the loop stopped, then lets the graph continue to the answer."""

        async def node(state: AgentState) -> dict:
            return {
                "stop_reason": reason,
                "trace": [{"step": "stop", "reason": reason}],
            }

        return node

    async def answer_node(state: AgentState) -> dict:
        """Write the final answer from the best passages found across all rounds (same step as Phase 1)."""
        ctx = select_context(state["runs"], k + extra_context)
        ans = await generate_answer(llm, state["question"], ctx)
        return {
            "answer": ans,
            "tokens_in": state.get("tokens_in", 0) + ans.input_tokens,
            "tokens_out": state.get("tokens_out", 0) + ans.output_tokens,
            "trace": [
                {
                    "step": "answer",
                    "passages": len(ctx),
                    "abstained": ans.abstained,
                    "cited": [h.source for h in ans.citations],
                }
            ],
        }

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("plan", plan_node)
    g.add_node("answer", answer_node)
    for reason in ("max_rounds", "no_progress", "token_budget"):
        g.add_node(f"stop_{reason}", stopper(reason))
        g.add_edge(f"stop_{reason}", "answer")
    g.add_edge(START, "retrieve")
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        ["grade", "stop_max_rounds", "stop_no_progress", "stop_token_budget"],
    )
    g.add_conditional_edges("grade", route_after_grade, ["plan", "answer"])
    g.add_conditional_edges("plan", route_after_plan, ["retrieve", "answer"])
    g.add_edge("answer", END)

    final = await g.compile().ainvoke({"question": question, "queries": [question]})
    ctx = select_context(final["runs"], k + extra_context)
    log.info(
        "agent done: stopped because %s after %d round(s), %d tokens, %.1fs",
        final.get("stop_reason", "sufficient"),
        final["rounds"],
        final.get("tokens_in", 0) + final.get("tokens_out", 0),
        time.perf_counter() - t0,
    )
    return AgentResult(
        answer=final["answer"],
        rounds=final["rounds"],
        stop_reason=final.get("stop_reason", "sufficient"),
        input_tokens=final.get("tokens_in", 0),
        output_tokens=final.get("tokens_out", 0),
        latency_s=time.perf_counter() - t0,
        trace=final["trace"],
        queries=final["tried"],
        context=ctx,
    )
