"""Queue every markdown file under a directory for ingestion.

python -m scripts.enqueue_corpus corpus/fastapi [--limit N]
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from app.config import get_settings
from app.db import close_pool, init_schema, open_pool
from app.ingest.chunker import extract_title
from app.ingest.queue import enqueue_many

QUESTIONS = Path(__file__).parent.parent / "evals" / "questions.jsonl"


def select_subset(docs: list[tuple[str, str, str]], expected: set[str], total: int, seed: int = 7):
    """Pick a small, deterministic dev corpus for tight API quotas.

    Always keeps every document the eval questions expect (so the eval stays meaningful), then
    fills up to `total` with seeded random distractors so retrieval still has to discriminate.

    Args:
        docs: (source, title, text) for every candidate document.
        expected: Sources named in `evals/questions.jsonl`.
        total: Target number of documents (at least len(expected) are always kept).
        seed: Seed for the distractor choice; same seed, same subset.

    Returns:
        The chosen documents, sorted by source.
    """
    must = [d for d in docs if d[0] in expected]
    rest = [d for d in docs if d[0] not in expected]
    random.Random(seed).shuffle(rest)
    return sorted(must + rest[: max(total - len(must), 0)])


async def main() -> None:
    """Walk `root` for .md files (skipping `corpus_exclude`) and enqueue them as ingest jobs.

    The source is the path relative to `root`; the title comes from the first level-1 heading,
    falling back to the file name. Documents that already have a pending/running job are skipped.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--subset",
        type=int,
        default=0,
        help="dev corpus of N docs: every eval-expected doc + seeded distractors",
    )
    args = ap.parse_args()

    root = Path(args.root)
    exclude = set(get_settings().corpus_exclude)
    docs = []
    for p in sorted(root.rglob("*.md")):
        if p.name in exclude:
            continue
        text = p.read_text(encoding="utf-8")
        source = p.relative_to(root).as_posix()
        docs.append((source, extract_title(text, p.stem), text))
    if args.subset:
        expected = {
            s
            for line in QUESTIONS.read_text().splitlines()
            if line.strip()
            for s in json.loads(line)["expected_sources"]
        }
        docs = select_subset(docs, expected, args.subset)
    if args.limit:
        docs = docs[: args.limit]

    pool = await open_pool()
    await init_schema(pool)
    created = await enqueue_many(pool, docs)
    print(f"{len(docs)} docs found, {created} new jobs enqueued")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
