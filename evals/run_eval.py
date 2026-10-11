"""Evaluate retrieval (and optionally generation) against evals/questions.jsonl.

    python -m evals.run_eval --mode vector                # retrieval metrics only (cheap)
    python -m evals.run_eval --mode vector --generate     # + answers, citations, abstention

Metrics (answerable questions: single / multi / partial):
  hit@k          any expected source appears in the top-k
  full_recall@k  ALL expected sources appear in the top-k (matters for multi-hop)
  MRR            1/rank of the first expected source
With --generate:
  cited_ok       answer cites at least one expected source
  false_abstain  model said INSUFFICIENT_CONTEXT although the answer exists
Unanswerable questions (--generate): abstain_rate should be 100%.
Every run is saved to evals/results/ so ablations (vector -> hybrid -> rerank -> agent) can be
compared side by side."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from app.db import close_pool, open_pool
from app.providers import get_embedder, get_llm
from app.rag.generate import generate_answer
from app.rag.retrieve import vector_search

QUESTIONS = Path(__file__).parent / "questions.jsonl"
RESULTS = Path(__file__).parent / "results"


async def retrieve(mode: str, pool, embedder, question: str, k: int):
    """Run the retriever selected by `mode` and return its hits.

    Only "vector" exists so far; hybrid, rerank and agent modes will be added here as phases land.
    Exits with an error for unknown modes.
    """
    if mode == "vector":
        return await vector_search(pool, embedder, question, k)
    raise SystemExit(f"unknown mode {mode!r}")  # hybrid / rerank / agent land in later phases


def _rank_of_first(expected: list[str], sources: list[str]) -> int | None:
    """1-based rank of the first retrieved source that is expected, or None if none is."""
    for i, s in enumerate(sources, 1):
        if s in expected:
            return i
    return None


async def main() -> None:
    """Run the eval: retrieve (and optionally generate) for each question, then score and save.

    Prints one line per question and a JSON summary, and writes the summary plus per-question
    rows to `evals/results/`. Questions with no expected sources are the unanswerable ones,
    scored only by abstention rate.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vector")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    questions = [json.loads(l) for l in QUESTIONS.read_text().splitlines() if l.strip()]
    if args.limit:
        questions = questions[: args.limit]
    pool, embedder, llm = await open_pool(), get_embedder(), get_llm()

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        hits = await retrieve(args.mode, pool, embedder, q["question"], args.k)
        sources = list(dict.fromkeys(h.source for h in hits))
        exp = q["expected_sources"]
        row = {
            **q,
            "retrieved": sources,
            "hit": bool(exp) and any(s in sources for s in exp),
            "full_recall": bool(exp) and all(s in sources for s in exp),
            "rank": _rank_of_first(exp, sources) if exp else None,
            "retrieval_s": round(time.perf_counter() - t0, 3),
        }
        if args.generate:
            ans = await generate_answer(llm, q["question"], hits)
            cited = {h.source for h in ans.citations}
            row.update(
                answer=ans.text,
                abstained=ans.abstained,
                cited_ok=bool(cited & set(exp)),
                input_tokens=ans.input_tokens,
                output_tokens=ans.output_tokens,
                latency_s=round(ans.latency_s, 2),
            )
        rows.append(row)
        print(f"{q['id']} {q['type']:<12} hit={row['hit']!s:<5} rank={row['rank']}")

    answerable = [r for r in rows if r["expected_sources"]]
    unans = [r for r in rows if not r["expected_sources"]]
    n = max(len(answerable), 1)
    summary = {
        "mode": args.mode,
        "k": args.k,
        "n_answerable": len(answerable),
        "hit@k": round(sum(r["hit"] for r in answerable) / n, 3),
        "full_recall@k": round(sum(r["full_recall"] for r in answerable) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in answerable if r["rank"]) / n, 3),
    }
    if args.generate:
        summary.update(
            cited_ok=round(sum(r["cited_ok"] for r in answerable) / n, 3),
            false_abstain=round(sum(r["abstained"] for r in answerable) / n, 3),
            abstain_rate_unanswerable=round(
                sum(r["abstained"] for r in unans) / max(len(unans), 1), 3
            ),
            input_tokens=sum(r["input_tokens"] for r in rows),
            output_tokens=sum(r["output_tokens"] for r in rows),
            avg_latency_s=round(sum(r["latency_s"] for r in rows) / max(len(rows), 1), 2),
        )
    print(json.dumps(summary, indent=2))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.mode}-{'gen-' if args.generate else ''}{int(time.time())}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("saved", out)
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
