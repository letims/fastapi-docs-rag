"""Queue every markdown file under a directory for ingestion.

    python -m scripts.enqueue_corpus corpus/fastapi [--limit N]
"""

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.db import close_pool, init_schema, open_pool
from app.ingest.chunker import extract_title
from app.ingest.queue import enqueue_many


async def main() -> None:
    """Walk `root` for .md files (skipping `corpus_exclude`) and enqueue them as ingest jobs.

    The source is the path relative to `root`; the title comes from the first level-1 heading,
    falling back to the file name. Documents that already have a pending/running job are skipped.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=0)
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
    if args.limit:
        docs = docs[: args.limit]

    pool = await open_pool()
    await init_schema(pool)
    created = await enqueue_many(pool, docs)
    print(f"{len(docs)} docs found, {created} new jobs enqueued")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
