"""Load test for the ingestion pipeline itself: workers, queue, chunker and Postgres.

python -m scripts.load_test --docs 2000 --concurrency 1,4,8,16 --latency-ms 50

The embedder is the offline FakeEmbedder, so no API, no quota and no cache are involved: what is
measured is how fast the pipeline can go when the embedding provider is NOT the bottleneck. A
fixed `--latency-ms` per embedding call stands in for a real API round trip, because that is
what makes extra workers pay off (with 0 ms the test is CPU-bound in one Python process).

Documents come from the real corpus (default `corpus/fastapi`), repeated and rewritten so that
every chunk is unique: otherwise chunk-hash reuse would skip the embedding step and the test
would measure the shortcut instead of the work. Without a corpus it falls back to synthetic text.

By default a throwaway Postgres is started with the `pgserver` dev dependency (no Docker, no risk
to your data). `--database-url` uses your own server instead; the app tables there are DROPPED
before every run, so it needs `--drop-tables`.
"""

import argparse
import asyncio
import contextlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import app.db as db
from app.config import get_settings
from app.ingest import queue
from app.providers.fake import FakeEmbedder
from app.worker import worker_loop

PARAGRAPHS_PER_SYNTHETIC_DOC = 12
WORDS = (
    "request response router dependency model schema field query header cookie middleware "
    "session token cache worker queue index vector chunk document async await handler"
).split()


class LatencyEmbedder:
    """Wrap an embedder and sleep before each call, to imitate a network round trip."""

    def __init__(self, inner: FakeEmbedder, latency_s: float):
        self.inner = inner
        self.latency_s = latency_s

    @property
    def calls(self) -> int:
        return self.inner.calls

    async def embed(self, texts, *, kind="document"):
        """Sleep `latency_s`, then embed with the wrapped embedder."""
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        return await self.inner.embed(texts, kind=kind)


def uniquify(text: str, tag: str) -> str:
    """Prefix every plain paragraph with `tag` so each chunk of each variant hashes differently.

    Headings and code fences are left alone so the structure the chunker sees stays realistic.
    """
    out = []
    for i, para in enumerate(text.split("\n\n")):
        keep = not para.strip() or para.lstrip().startswith(("#", "```", "~~~"))
        out.append(para if keep else f"{tag}-{i} {para}")
    return "\n\n".join(out)


def synthetic_docs(n: int, seed: int = 7) -> list[tuple[str, str, str]]:
    """Generate `n` markdown documents of random words, used when no corpus is available."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        parts = [f"# Synthetic document {i}"]
        for p in range(PARAGRAPHS_PER_SYNTHETIC_DOC):
            if p % 4 == 0:
                parts.append(f"## Section {p // 4}")
            parts.append(" ".join(rng.choices(WORDS, k=rng.randint(40, 160))))
        docs.append((f"synthetic/{i}.md", f"Synthetic document {i}", "\n\n".join(parts)))
    return docs


def load_base_docs(corpus: Path) -> list[tuple[str, str, str]]:
    """Read every markdown file under `corpus` that ingestion would read (honouring excludes)."""
    from app.ingest.chunker import extract_title

    exclude = set(get_settings().corpus_exclude)
    docs = []
    for p in sorted(corpus.rglob("*.md")) if corpus.is_dir() else []:
        if p.name not in exclude:
            text = p.read_text(encoding="utf-8")
            docs.append((p.relative_to(corpus).as_posix(), extract_title(text, p.stem), text))
    return docs


def make_docs(n: int, corpus: Path) -> tuple[list[tuple[str, str, str]], str]:
    """Build `n` documents with unique sources and unique chunks.

    Returns:
        (documents, description of where they came from).
    """
    base = load_base_docs(corpus)
    origin = f"{len(base)} real docs from {corpus}"
    if not base:
        base, origin = synthetic_docs(min(n, 50)), "synthetic text (no corpus found)"
    docs = []
    for i in range(n):
        source, title, text = base[i % len(base)]
        docs.append((f"load/{i:06d}/{source}", title, uniquify(text, f"v{i}")))
    return docs, f"{n} docs built from {origin}, every chunk unique"


async def reset(pool) -> None:
    """Drop and recreate the app tables so each run starts from an empty database."""
    async with pool.connection() as c:
        await c.execute("DROP TABLE IF EXISTS chunks, documents, ingest_jobs, api_usage CASCADE")
    await db.init_schema(pool)


async def run_once(pool, docs, concurrency: int, latency_ms: float = 0.0) -> dict:
    """Ingest `docs` with `concurrency` workers into an empty schema and measure it.

    Args:
        pool: Open pool, at least `concurrency + 2` connections.
        docs: (source, title, content) tuples.
        concurrency: Number of worker tasks.
        latency_ms: Simulated embedding API round trip per call.

    Returns:
        Timings and counters; `ingest_s` covers claim to last commit, not index build.
    """
    await reset(pool)
    t0 = time.perf_counter()
    await queue.enqueue_many(pool, docs)
    enqueue_s = time.perf_counter() - t0

    embedder = LatencyEmbedder(FakeEmbedder(get_settings().embed_dim), latency_ms / 1000)
    counters = {
        k: 0
        for k in (
            "ingested",
            "skipped",
            "retried",
            "failed",
            "paused",
            "chunks_embedded",
            "chunks_reused",
        )
    }
    tasks = [
        asyncio.create_task(worker_loop(f"w{i}", pool, embedder, counters, drain=False))
        for i in range(concurrency)
    ]
    t0 = time.perf_counter()
    while counters["ingested"] + counters["skipped"] + counters["failed"] < len(docs):
        await asyncio.sleep(0.01)
    ingest_s = time.perf_counter() - t0
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    t0 = time.perf_counter()
    await db.build_vector_index(pool)
    index_s = time.perf_counter() - t0
    async with pool.connection() as c:
        n_chunks = (await (await c.execute("SELECT count(*) AS n FROM chunks")).fetchone())["n"]
    return {
        "concurrency": concurrency,
        "latency_ms": latency_ms,
        "docs": len(docs),
        "chunks": n_chunks,
        "embed_calls": embedder.calls,
        "failed": counters["failed"],
        "enqueue_s": round(enqueue_s, 2),
        "ingest_s": round(ingest_s, 2),
        "index_s": round(index_s, 2),
        "docs_per_s": round(len(docs) / ingest_s, 1),
        "chunks_per_s": round(n_chunks / ingest_s, 1),
    }


def format_table(rows: list[dict]) -> str:
    """Render results as a fixed-width text table."""
    cols = [
        ("workers", "concurrency"),
        ("latency ms", "latency_ms"),
        ("docs/s", "docs_per_s"),
        ("chunks/s", "chunks_per_s"),
        ("ingest s", "ingest_s"),
        ("index s", "index_s"),
        ("embed calls", "embed_calls"),
        ("failed", "failed"),
    ]
    widths = [max(len(h), *(len(str(r[k])) for r in rows)) for h, k in cols]
    line = "  ".join(h.rjust(w) for (h, _), w in zip(cols, widths, strict=True))
    body = [
        "  ".join(str(r[k]).rjust(w) for (_, k), w in zip(cols, widths, strict=True)) for r in rows
    ]
    return "\n".join([line, "-" * len(line), *body])


@contextlib.contextmanager
def throwaway_database():
    """Yield the URL of a temporary Postgres (pgvector included) that is deleted afterwards."""
    try:
        import pgserver
    except ImportError as e:  # pragma: no cover - dev dependency is always present in CI
        raise SystemExit("pgserver is missing: run `uv sync`, or pass --database-url.") from e
    with tempfile.TemporaryDirectory() as d:
        srv = pgserver.get_server(Path(d), cleanup_mode="delete")
        try:
            yield srv.get_uri()
        finally:
            srv.cleanup()


async def run_all(url: str, docs, levels: list[int], latency_ms: float, dim: int) -> list[dict]:
    """Run every concurrency level against `url`; returns one result row per level."""
    os.environ.update(DATABASE_URL=url, EMBED_DIM=str(dim), PROVIDER="fake")
    get_settings.cache_clear()
    db._pool = None
    pool = await db.open_pool(max_size=max(levels) + 2)
    try:
        rows = []
        for level in levels:
            row = await run_once(pool, docs, level, latency_ms)
            print(f"  {level} workers: {row['docs_per_s']} docs/s", flush=True)
            rows.append(row)
        return rows
    finally:
        await db.close_pool()


def main() -> None:
    """CLI entry point; see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--concurrency", default="1,4,8,16", help="comma-separated worker counts")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="simulated API round trip")
    ap.add_argument("--dim", type=int, default=1024, help="embedding dimension (match EMBED_DIM)")
    ap.add_argument("--corpus", type=Path, default=Path("corpus/fastapi"))
    ap.add_argument("--database-url", help="use this Postgres instead of a throwaway one")
    ap.add_argument(
        "--drop-tables", action="store_true", help="required with --database-url: allow DROP TABLE"
    )
    ap.add_argument("--json", type=Path, help="also write the results to this file")
    args = ap.parse_args()
    if args.database_url and not args.drop_tables:
        raise SystemExit("--database-url drops the app tables on every run; add --drop-tables.")

    levels = [int(x) for x in args.concurrency.split(",")]
    docs, origin = make_docs(args.docs, args.corpus)
    print(f"{origin}; dim {args.dim}; simulated API latency {args.latency_ms} ms")

    def go(url: str) -> list[dict]:
        return asyncio.run(run_all(url, docs, levels, args.latency_ms, args.dim))

    if args.database_url:
        rows = go(args.database_url)
    else:
        with throwaway_database() as url:
            rows = go(url)
    print()
    print(format_table(rows))
    best = max(rows, key=lambda r: r["docs_per_s"])
    hours = 50_000 / best["docs_per_s"] / 3600
    print(
        f"\nBest: {best['docs_per_s']} docs/s with {best['concurrency']} workers; "
        f"50,000 docs would take about {hours:.1f} h at this rate (embedding calls excluded)."
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
