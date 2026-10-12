"""Offline retrieval experiments: no API calls, no Docker, no quota.

    PYTHONPATH=. uv run python experiments/phase2_offline.py

Rebuilds the whole corpus in a throwaway Postgres using the REAL Voyage vectors stored in the
on-disk API cache (`.cache/api_cache.sqlite`; a cache miss is an error, never an API call), then
scores retrieval variants on the eval questions. The vector and hybrid baselines reproduce the
saved eval runs exactly (0.882 / 0.971 and 0.882 / 0.730), which is what makes the other rows
trustworthy. Needs the corpus (`make corpus`) and a finished ingest with the default models
(voyage-4-lite, 1024 dims). Reranking cannot be tested here because it needs the Voyage API.

Exploratory tooling, not part of the product: it is outside the coverage gate on purpose."""

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

import pgserver

root = Path.cwd()
srv = pgserver.get_server(Path(tempfile.mkdtemp()), cleanup_mode="delete")
os.environ.update(DATABASE_URL=srv.get_uri(), EMBED_DIM="1024", PROVIDER="fake")

import app.db as db
from app.cache import DiskCache, make_key, unpack_vector
from app.config import get_settings
from app.ingest.chunker import extract_title
from app.ingest.pipeline import ingest_document
from app.rag import retrieve as R

MODEL, DIM = "voyage-4-lite", 1024
cache = DiskCache(root / ".cache/api_cache.sqlite")


class CacheOnly:
    dim = DIM

    async def embed(self, texts, *, kind="document"):
        out = []
        for t in texts:
            blob = cache.get(make_key("emb", MODEL, str(DIM), kind, t))
            if blob is None:
                raise KeyError(f"cache miss ({kind}): {t[:60]!r}")
            out.append(unpack_vector(blob))
        return out


QS = [json.loads(l) for l in (root / "evals/questions.jsonl").read_text().splitlines() if l.strip()]


async def main():
    get_settings.cache_clear()
    pool = await db.open_pool()
    await db.init_schema(pool)
    emb = CacheOnly()
    exclude = set(get_settings().corpus_exclude)
    n = 0
    for p in sorted((root / "corpus/fastapi").rglob("*.md")):
        if p.name in exclude:
            continue
        text = p.read_text(encoding="utf-8")
        await ingest_document(
            pool,
            emb,
            source=p.relative_to(root / "corpus/fastapi").as_posix(),
            title=extract_title(text, p.stem),
            content=text,
        )
        n += 1
    await db.build_vector_index(pool)
    async with pool.connection() as c:
        r = await (
            await c.execute(
                "SELECT (SELECT count(*) FROM documents) d, (SELECT count(*) FROM chunks) c"
            )
        ).fetchone()
    print("docs", r["d"], "chunks", r["c"], flush=True)

    # document frequency of each lexeme (for IDF-style filtering)
    async with pool.connection() as c:
        rows = await (
            await c.execute("SELECT word, ndoc FROM ts_stat('SELECT tsv FROM chunks')")
        ).fetchall()
    df = {r["word"]: r["ndoc"] for r in rows}
    total = r["c"]

    async def lexemes(word):
        async with pool.connection() as c:
            row = await (
                await c.execute(
                    "SELECT tsvector_to_array(to_tsvector('english', %s)) AS a", (word,)
                )
            ).fetchone()
        return row["a"]

    async def kw(question, n=20, mode="or", max_df=None):
        words = list(
            dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 1)
        )
        if mode == "ident":
            raw = re.findall(r"[A-Za-z0-9_.]+", question)
            ident = [
                w
                for w in raw
                if (
                    "_" in w
                    or re.search(r"[a-z][A-Z]", w)
                    or re.search(r"[A-Z]{2,}", w)
                    or "." in w.strip(".")
                )
            ]
            words = list(
                dict.fromkeys(
                    x for w in ident for x in re.findall(r"[a-z0-9]+", w.lower()) if len(x) > 1
                )
            )
        if max_df is not None:
            keep = []
            for w in words:
                lx = await lexemes(w)
                if lx and all(df.get(l, 0) / total <= max_df for l in lx):
                    keep.append(w)
            words = keep
        if not words:
            return []
        q = " | ".join(words)
        async with pool.connection() as c:
            cur = await c.execute(
                """SELECT c.id, d.source, c.heading, c.content, ts_rank_cd(c.tsv, query) AS score
                     FROM chunks c JOIN documents d ON d.id=c.document_id, to_tsquery('english', %s) query
                    WHERE c.tsv @@ query ORDER BY score DESC, c.id LIMIT %s""",
                (q, n),
            )
            rows = await cur.fetchall()
        return [
            R.Hit(x["id"], x["source"], x["heading"], x["content"], float(x["score"])) for x in rows
        ]

    def fuse(lists, weights, k=60):
        scores, first = {}, {}
        for lst, w in zip(lists, weights):
            for rank, h in enumerate(lst, 1):
                scores[h.chunk_id] = scores.get(h.chunk_id, 0) + w / (k + rank)
                first.setdefault(h.chunk_id, h)
        order = sorted(scores, key=lambda i: -scores[i])
        from dataclasses import replace

        return [replace(first[i], score=scores[i]) for i in order]

    def dedupe_sources(hits):
        return list(dict.fromkeys(h.source for h in hits))

    async def run(name, fn, k=5):
        hit = full = 0
        mrr = 0.0
        na = 0
        bad = []
        for q in QS:
            exp = q["expected_sources"]
            if not exp:
                continue
            hits = await fn(q["question"])
            srcs = list(dict.fromkeys(h.source for h in hits[:k]))
            na += 1
            h = any(s in srcs for s in exp)
            f = all(s in srcs for s in exp)
            hit += h
            full += f
            rk = next((i for i, s in enumerate(srcs, 1) if s in exp), None)
            mrr += 1 / rk if rk else 0
            if not f:
                bad.append(q["id"])
        print(
            f"{name:46s} hit {hit}/{na}  full {full}/{na} ({full / na:.3f})  MRR {mrr / na:.3f}  miss-full {bad}",
            flush=True,
        )

    async def recall_at(name, fn, depth):
        # full recall among the top-`depth` chunks (what a reranker could still pick from)
        full = na = 0
        bad = []
        for q in QS:
            exp = q["expected_sources"]
            if not exp:
                continue
            hits = await fn(q["question"])
            srcs = {h.source for h in hits[:depth]}
            na += 1
            ok = all(s in srcs for s in exp)
            full += ok
            if not ok:
                bad.append(q["id"])
        print(f"  candidates {name:34s} full-recall@{depth} {full}/{na}  miss {bad}", flush=True)

    vec = lambda qn, n=5: R.vector_search(pool, emb, qn, n)
    print("--- baselines (should match saved runs: vector 0.882/0.971, hybrid 0.882/0.73)")
    await run("vector", lambda q: vec(q, 5))

    async def hyb(q, wk=1.0, mode="or", max_df=None, depth=20):
        v = await vec(q, depth)
        k_ = await kw(q, depth, mode, max_df)
        return fuse([v, k_], [1.0, wk])

    await run("hybrid (current: OR, w=1)", lambda q: hyb(q))
    print("--- candidate pools (top 20) by keyword weight")
    for wk in (1.0, 0.5, 0.3, 0.15):
        await recall_at(f"hybrid OR w {wk}", lambda q, wk=wk: hyb(q, wk), 20)
    await recall_at("vector top 20", lambda q: vec(q, 20), 20)
    for d in (10, 15, 20, 30):
        await recall_at(f"hybrid OR w 1.0 depth {d}", lambda q, d=d: hyb(q, 1.0, "or", None, d), d)
    await db.close_pool()
    srv.cleanup()


asyncio.run(main())
