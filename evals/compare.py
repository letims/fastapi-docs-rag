"""Compare an eval run against a stored baseline and fail on regressions (drift detection).

    python -m evals.compare RUN.json [--baseline evals/baseline/rerank.json]
    python -m evals.compare --latest rerank           # newest results/rerank-*.json vs baseline/rerank.json
    python -m evals.compare --promote RUN.json        # make RUN the baseline for its tag
    python -m evals.compare --promote RUN.json --name ci-dev40-rerank   # the CI baseline

Exit codes: 0 no regression, 1 regression, 3 not comparable (different question set).

Tolerances follow docs/evaluation.md: with 17 answerable questions one question moves a rate by
~0.06, which is noise, so retrieval/answer rates fail only when they drop by MORE than one question.
Abstention on unanswerable questions is a safety property: any drop fails. Cost/query fails above
+25%. Latency is reported, never gated (free-tier latency is mostly queueing on the provider side).
The table is Markdown so CI can append it to the job summary as is."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
BASELINES = ROOT / "baseline"

# metric: (direction, tolerance kind). "q" = one question's worth; "abs:x" absolute; "rel:x" relative.
METRICS = {
    "hit@k": ("up", "q"),
    "full_recall@k": ("up", "q"),
    "mrr": ("up", "q"),
    "cited_ok": ("up", "q"),
    "false_abstain": ("down", "q"),
    "abstain_rate_unanswerable": ("up", "abs:0"),
    "grounded_rate": ("up", "abs:0.05"),
    "cost_per_query_usd": ("down", "rel:0.25"),
    "tokens_per_query": ("down", "rel:0.25"),
    "retrieval_p95_s": ("down", "report"),
    "p95_latency_s": ("down", "report"),
}

_NAME = re.compile(r"^(?P<tag>.+?)-(?:(?:[0-9a-f]{7,40}|nogit)-)?\d{9,}\.json$")


def run_tag(path: Path) -> str:
    """`rerank-gen` from `rerank-gen-<sha>-<ts>.json` (or the older `rerank-gen-<ts>.json`)."""
    m = _NAME.match(path.name)
    if not m:
        raise SystemExit(f"not an eval result file name: {path.name}")
    return m.group("tag")


def latest(tag: str, results: Path | None = None) -> Path:
    """Newest result file whose tag is exactly `tag` (so `rerank` never picks `rerank-gen`)."""
    results = results or RESULTS
    runs = [p for p in results.glob(f"{tag}-*.json") if run_tag(p) == tag]
    if not runs:
        raise SystemExit(f"no {tag} runs in {results}: run `make eval MODE=...` first")
    return max(runs, key=lambda p: int(p.stem.rsplit("-", 1)[1]))


def _tolerance(kind: str, base: float, n_answerable: int) -> float | None:
    if kind == "q":
        return 1 / max(n_answerable, 1) + 0.0015  # slack for the 3-decimal rounding in summaries
    if kind.startswith("abs:"):
        return float(kind[4:]) + 1e-9
    if kind.startswith("rel:"):
        return abs(base) * float(kind[4:]) + 1e-12
    return None  # report only


def compare(run: dict, base: dict) -> tuple[list[dict], list[str], bool]:
    """Return (metric rows, notes, regressed). Only metrics present in both summaries are compared."""
    rs, bs = run["summary"], base["summary"]
    notes: list[str] = []
    rp, bp = rs.get("provenance"), bs.get("provenance")
    if rp and bp:
        if rp["questions_hash"] != bp["questions_hash"]:
            notes.append("NOT COMPARABLE: the question set changed. Promote a new baseline.")
        if rp["corpus"] != bp["corpus"]:
            notes.append(f"corpus differs: run {rp['corpus']} vs baseline {bp['corpus']}")
        if rp["settings_hash"] != bp["settings_hash"]:
            changed = sorted(
                k for k in rp["settings"] if rp["settings"].get(k) != bp["settings"].get(k)
            )
            notes.append(f"settings changed: {', '.join(changed)}")
        if rp.get("api_cache") or bp.get("api_cache"):
            notes.append("API cache on: answers may be replayed, run-to-run variance is hidden")
    else:
        notes.append("baseline or run has no provenance (pre-provenance run): checks skipped")

    rows, regressed = [], False
    n = rs.get("n_answerable") or bs.get("n_answerable") or 1
    for metric, (direction, kind) in METRICS.items():
        if rs.get(metric) is None or bs.get(metric) is None:
            continue
        r, b = float(rs[metric]), float(bs[metric])
        delta = r - b
        worse = -delta if direction == "up" else delta
        tol = _tolerance(kind, b, n)
        if tol is None:
            status = "report"
        elif worse > tol:
            status, regressed = "REGRESSION", True
        elif worse < -tol:
            status = "better"
        else:
            status = "ok"
        rows.append({"metric": metric, "baseline": b, "run": r, "delta": delta, "status": status})
    return rows, notes, regressed


def question_changes(run: dict, base: dict) -> list[str]:
    """Per-question flips in hit / full_recall / rank: the 'which question moved' view."""
    before = {r["id"]: r for r in base.get("rows", [])}
    out = []
    for r in run.get("rows", []):
        b = before.get(r["id"])
        if not b or not r.get("expected_sources"):
            continue
        for field in ("hit", "full_recall"):
            if b.get(field) != r.get(field):
                out.append(f"{r['id']} {field}: {b.get(field)} -> {r.get(field)}")
        if b.get("rank") != r.get("rank"):
            out.append(f"{r['id']} first right doc rank: {b.get('rank')} -> {r.get('rank')}")
    return out


def render(run_path: Path, base_path: Path, rows, notes, changes, regressed) -> str:
    def fmt(x):
        return f"{x:.4g}" if isinstance(x, float) else str(x)

    lines = [
        f"### Eval vs baseline: {'❌ regression' if regressed else '✅ no regression'}",
        f"run `{run_path.name}` · baseline `{base_path.name}`",
        "",
        "| metric | baseline | run | Δ | status |",
        "|---|---|---|---|---|",
        *(
            f"| {r['metric']} | {fmt(r['baseline'])} | {fmt(r['run'])} | {r['delta']:+.4g} | {r['status']} |"
            for r in rows
        ),
    ]
    if changes:
        lines += ["", "Questions that moved:", *(f"- {c}" for c in changes)]
    if notes:
        lines += ["", *(f"> {n}" for n in notes)]
    return "\n".join(lines)


def promote(run_path: Path, baselines: Path | None = None, name: str | None = None) -> Path:
    """Store the run's summary plus a slim per-question view as the baseline for its tag."""
    data = json.loads(run_path.read_text())
    keep = ("id", "type", "expected_sources", "retrieved", "hit", "full_recall", "rank", "grounded")
    slim = {
        "source_run": run_path.name,
        "summary": data["summary"],
        "rows": [{k: r[k] for k in keep if k in r} for r in data["rows"]],
    }
    baselines = baselines or BASELINES
    baselines.mkdir(parents=True, exist_ok=True)
    out = baselines / f"{name or run_tag(run_path)}.json"
    out.write_text(json.dumps(slim, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", nargs="?", type=Path, help="result file to check")
    ap.add_argument("--latest", metavar="TAG", help="use the newest results/TAG-*.json")
    ap.add_argument("--baseline", type=Path, help="default: evals/baseline/<tag>.json")
    ap.add_argument("--promote", type=Path, metavar="RUN", help="make RUN the baseline for its tag")
    ap.add_argument(
        "--name", help="with --promote: baseline name instead of the tag (e.g. ci-dev40-rerank)"
    )
    args = ap.parse_args(argv)

    if args.promote:
        print("baseline written:", promote(args.promote, name=args.name))
        return 0
    run_path = latest(args.latest) if args.latest else args.run
    if run_path is None:
        ap.error("give a RUN file, --latest TAG or --promote RUN")
    base_path = args.baseline or BASELINES / f"{run_tag(run_path)}.json"
    if not base_path.exists():
        print(f"no baseline at {base_path}: promote one with --promote {run_path}")
        return 3
    run, base = json.loads(run_path.read_text()), json.loads(base_path.read_text())
    rows, notes, regressed = compare(run, base)
    print(render(run_path, base_path, rows, notes, question_changes(run, base), regressed))
    if any(n.startswith("NOT COMPARABLE") for n in notes):
        return 3
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
