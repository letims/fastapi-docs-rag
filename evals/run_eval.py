"""Evaluate retrieval (and optionally generation) against evals/questions.jsonl.

    python -m evals.run_eval --mode vector                # retrieval metrics only (cheap)
    python -m evals.run_eval --mode hybrid                # vector + full-text, fused with RRF
    python -m evals.run_eval --mode rerank                # hybrid candidates, cross-encoder rerank
    python -m evals.run_eval --mode vector --generate     # + answers, citations, abstention
    python -m evals.run_eval --mode rerank --agent        # Phase 3 loop on top of that retriever
    python -m evals.run_eval --mode rerank --agent --only q14,q16,q17   # just those questions

Metrics (answerable questions: single / multi / partial):
  hit@k          any expected source appears in the top-k
  full_recall@k  ALL expected sources appear in the top-k (matters for multi-hop)
  MRR            1/rank of the first expected source
With --generate (implied by --agent):
  cited_ok       answer cites at least one expected source
  false_abstain  model said INSUFFICIENT_CONTEXT although the answer exists
Unanswerable questions (--generate): abstain_rate should be 100%.
With --judge (implies --generate): grounded = share of cited claims an LLM judge finds supported
by the cited passage (evals/judge.py); None when the judge reply is unusable.
Cost: cost_usd per row is the list-price equivalent of the LLM tokens (PRICES in app/config.py).
Every summary carries provenance (git SHA, settings hash, model IDs, corpus size) so a baseline
comparison (evals/compare.py) can tell a code change from a config or corpus change.
Every run is saved to evals/results/ so ablations (vector -> hybrid -> rerank -> agent) can be
compared side by side."""

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from app.config import PRICES_AS_OF, cost_usd, get_settings, llm_model_key
from app.db import close_pool, open_pool
from app.logging_config import setup_logging
from app.providers import get_embedder, get_llm, get_reranker
from app.quota import QuotaExhausted
from app.rag.agent import run_agent
from app.rag.generate import generate_answer
from app.rag.retrieve import MODES, search
from evals.judge import judge_answer

QUESTIONS = Path(__file__).parent / "questions.jsonl"
RESULTS = Path(__file__).parent / "results"


def _mmss(seconds: float) -> str:
    """Format seconds as m:ss for the progress line."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


async def retrieve(
    mode: str, pool, embedder, question: str, k: int, reranker=None, candidates: int = 20
):
    """Run the retriever selected by `mode` ("vector", "hybrid" or "rerank") and return its hits.

    Exits with an error for unknown modes. The agent is not a mode: `--agent` runs the loop on top
    of one of these retrievers.
    """
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}: choose from {', '.join(MODES)}")
    return await search(mode, pool, embedder, reranker, question, k, candidates)


def _pct(values: list[float], q: float) -> float | None:
    """q-th percentile (0..1) of `values`, nearest-rank; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(q * len(ordered)) - 1)], 3)


def _git(*args: str) -> str:
    """Output of a git command, or "" outside a repo (CI tarballs, tests in tmp dirs)."""
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


async def provenance(pool, s, args) -> dict:
    """What produced this run: code, retrieval settings, models, corpus and question set.

    `settings_hash` covers everything that changes retrieval or answers, so two runs with equal
    hashes and SHAs differ only by provider-side drift (model updates, nondeterminism)."""
    knobs = {
        "mode": args.mode,
        "k": args.k,
        "candidates": args.candidates,
        "agent": args.agent,
        "rerank_max_per_source": s.rerank_max_per_source,
        "chunk_max_chars": s.chunk_max_chars,
        "embed": f"{s.embed_backend}:{s.voyage_embed_model if s.embed_backend == 'voyage' else s.gemini_embed_model}@{s.embed_dim}",
        "rerank": s.voyage_rerank_model if args.mode == "rerank" else None,
        "llm": llm_model_key(s) if args.generate else None,
        "agent_caps": [
            s.agent_max_rounds,
            s.agent_token_budget,
            s.agent_per_query,
            s.agent_context_extra,
        ]
        if args.agent
        else None,
    }
    async with pool.connection() as c:
        cur = await c.execute(
            "SELECT (SELECT count(*) FROM documents) AS docs, (SELECT count(*) FROM chunks) AS chunks"
        )
        row = await cur.fetchone()
    docs, chunks = (row["docs"], row["chunks"]) if isinstance(row, dict) else row
    return {
        "git_sha": _git("rev-parse", "--short", "HEAD") or "nogit",
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "settings": knobs,
        "settings_hash": hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()[
            :12
        ],
        "questions_hash": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:12],
        "corpus": {"documents": docs, "chunks": chunks},
        "api_cache": s.api_cache,
        "prices_as_of": PRICES_AS_OF if args.generate else None,
    }


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

    If the daily API budget runs out mid-run, the finished questions are still scored and saved,
    the summary carries an `interrupted` note, and the process exits with status 2.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vector", choices=MODES)
    ap.add_argument(
        "--candidates", type=int, default=20, help="first-stage depth for hybrid and rerank"
    )
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--agent",
        action="store_true",
        help="Phase 3 loop (implies --generate); --mode is its retriever",
    )
    ap.add_argument("--only", default="", help="comma-separated question ids, e.g. q14,q16,q17")
    ap.add_argument(
        "--judge",
        action="store_true",
        help="score groundedness with an LLM judge (implies --generate; one extra call per answer)",
    )
    args = ap.parse_args()
    # Show the limiter's "waiting Ns" lines: with a low RPM the run is slow, and silence looks like a hang.
    setup_logging()

    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            raise SystemExit(f"--only {args.only!r} matches no question id")
    if args.limit:
        questions = questions[: args.limit]
    args.generate = args.generate or args.agent or args.judge
    pool, embedder, llm = await open_pool(), get_embedder(), get_llm()
    reranker = get_reranker() if args.mode == "rerank" else None
    s = get_settings()
    price_key = llm_model_key(s)

    rows = []
    interrupted = None
    started = time.perf_counter()
    for i, q in enumerate(questions, 1):
        try:
            t0 = time.perf_counter()
            res = None
            if args.agent:
                res = await run_agent(
                    pool,
                    embedder,
                    llm,
                    reranker,
                    q["question"],
                    mode=args.mode,
                    k=args.k,
                    candidates=args.candidates,
                    max_rounds=s.agent_max_rounds,
                    token_budget=s.agent_token_budget,
                    per_query=s.agent_per_query,
                    extra_context=s.agent_context_extra,
                )
                hits = res.context
            else:
                hits = await retrieve(
                    args.mode, pool, embedder, q["question"], args.k, reranker, args.candidates
                )
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
                ans = res.answer if res else await generate_answer(llm, q["question"], hits)
                cited = {h.source for h in ans.citations}
                tok_in = res.input_tokens if res else ans.input_tokens
                tok_out = res.output_tokens if res else ans.output_tokens
                row.update(
                    answer=ans.text,
                    abstained=ans.abstained,
                    cited_ok=bool(cited & set(exp)),
                    cached=ans.cached,
                    input_tokens=tok_in,
                    output_tokens=tok_out,
                    latency_s=round(res.latency_s if res else ans.latency_s, 2),
                    cost_usd=cost_usd(price_key, tok_in, tok_out),
                )
                if args.judge:
                    v = await judge_answer(llm, q["question"], ans.text, hits)
                    row.update(
                        grounded=v.score,
                        claims_supported=v.supported,
                        claims_unsupported=v.unsupported,
                        least_supported=v.worst,
                        judge_tokens=v.input_tokens + v.output_tokens,
                    )
                if res:
                    row.update(
                        rounds=res.rounds,
                        stop_reason=res.stop_reason,
                        queries=res.queries,
                        trace=res.trace,
                    )
        except QuotaExhausted as e:
            # Stop cleanly: score what finished, save it, and say how to resume. Finished answers
            # are in the disk cache, so the re-run only pays for the remaining questions.
            interrupted = str(e)
            print(f"STOPPED before {q['id']}: {e}")
            break
        rows.append(row)
        elapsed = time.perf_counter() - started
        eta = elapsed / i * (len(questions) - i)
        print(
            f"[{i}/{len(questions)}] {q['id']} {q['type']:<8} hit={row['hit']!s:<5} "
            f"rank={row['rank']!s:<4} elapsed {_mmss(elapsed)}, about {_mmss(eta)} left"
        )

    answerable = [r for r in rows if r["expected_sources"]]
    unans = [r for r in rows if not r["expected_sources"]]
    n = max(len(answerable), 1)
    summary = {
        "mode": args.mode,
        "agent": args.agent,
        "rerank_max_per_source": s.rerank_max_per_source if args.mode == "rerank" else None,
        "k": args.k,
        "n_answerable": len(answerable),
        "hit@k": round(sum(r["hit"] for r in answerable) / n, 3),
        "full_recall@k": round(sum(r["full_recall"] for r in answerable) / n, 3),
        "mrr": round(sum(1 / r["rank"] for r in answerable if r["rank"]) / n, 3),
        "retrieval_p50_s": _pct([r["retrieval_s"] for r in rows], 0.5),
        "retrieval_p95_s": _pct([r["retrieval_s"] for r in rows], 0.95),
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
            llm_api_requests=sum(not r["cached"] for r in rows),  # what this run cost in quota
            p50_latency_s=_pct([r["latency_s"] for r in rows], 0.5),
            p95_latency_s=_pct([r["latency_s"] for r in rows], 0.95),
        )
        costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
        summary["cost_per_query_usd"] = (
            round(statistics.fmean(costs), 7) if costs and len(costs) == len(rows) else None
        )
    if args.judge:
        scored = [r["grounded"] for r in rows if r.get("grounded") is not None]
        answered = [r for r in rows if not r["abstained"]]
        summary.update(
            grounded_rate=round(statistics.fmean(scored), 3) if scored else None,
            judged=f"{len(scored)}/{len(answered)}",  # unparseable judge replies are not counted as passes
            judge_tokens=sum(r.get("judge_tokens", 0) for r in rows),
        )
    if args.agent and rows:
        stops: dict[str, int] = {}
        for r in rows:
            stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
        summary.update(
            avg_rounds=round(sum(r["rounds"] for r in rows) / len(rows), 2),
            tokens_per_query=round(
                sum(r["input_tokens"] + r["output_tokens"] for r in rows) / len(rows)
            ),
            stop_reasons=stops,
            note="hit/recall are over the agent's final context (k + extra), not the top k",
        )
    if interrupted:
        summary["interrupted"] = interrupted
    summary["provenance"] = await provenance(pool, s, args)
    print(json.dumps(summary, indent=2))
    RESULTS.mkdir(exist_ok=True)
    tag = ("agent-" if args.agent else "") + args.mode + ("-gen" if args.generate else "")
    out = RESULTS / f"{tag}-{summary['provenance']['git_sha']}-{int(time.time())}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("saved", out)
    await close_pool()
    if interrupted:
        print(
            "Quota used up: re-run after the reset (see `make quota`). Finished questions are cached."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
