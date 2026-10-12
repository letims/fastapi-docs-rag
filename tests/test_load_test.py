import json
import sys

import pytest

import app.db as db
from app.config import get_settings
from scripts import load_test
from tests.conftest import DIM


def test_uniquify_changes_plain_paragraphs_but_not_headings_or_fences():
    text = "# Title\n\nplain text\n\n```python\ncode\n```\n\n\n\nlast"
    out = load_test.uniquify(text, "v9")
    assert out.startswith("# Title\n\nv9-1 plain text")
    assert "```python\ncode\n```" in out
    assert out.endswith("v9-4 last")
    assert load_test.uniquify(text, "v1") != out


def test_make_docs_cycles_the_corpus_with_unique_sources_and_content(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nalpha body")
    (tmp_path / "b.md").write_text("# B\n\nbeta body")
    (tmp_path / "release-notes.md").write_text("# excluded")
    docs, origin = load_test.make_docs(5, tmp_path)
    assert len(docs) == 5 and "2 real docs" in origin
    assert len({d[0] for d in docs}) == 5 and len({d[2] for d in docs}) == 5
    assert docs[0][0] == "load/000000/a.md" and docs[2][0] == "load/000002/a.md"


def test_make_docs_falls_back_to_synthetic_text_without_a_corpus(tmp_path):
    docs, origin = load_test.make_docs(3, tmp_path / "missing")
    assert len(docs) == 3 and "synthetic" in origin
    assert all(d[2].startswith("# Synthetic document") for d in docs)


async def test_latency_embedder_delays_and_delegates():
    inner = load_test.FakeEmbedder(8)
    slow = load_test.LatencyEmbedder(inner, 0.01)
    out = await slow.embed(["a b", "c"], kind="query")
    assert len(out) == 2 and slow.calls == 1


async def test_run_once_ingests_every_doc_and_reports_throughput(pool, tmp_path):
    docs, _ = load_test.make_docs(12, tmp_path / "missing")
    res = await load_test.run_once(pool, docs, concurrency=3, latency_ms=1)
    assert res["docs"] == 12 and res["failed"] == 0 and res["chunks"] >= 12
    assert res["embed_calls"] >= 12 and res["docs_per_s"] > 0
    async with pool.connection() as c:
        n = (await (await c.execute("SELECT count(*) AS n FROM documents")).fetchone())["n"]
    assert n == 12


def test_format_table_aligns_columns():
    rows = [
        {
            "concurrency": 1,
            "latency_ms": 0.0,
            "docs_per_s": 10.5,
            "chunks_per_s": 80.0,
            "ingest_s": 1.2,
            "index_s": 0.3,
            "embed_calls": 12,
            "failed": 0,
        }
    ]
    lines = load_test.format_table(rows).splitlines()
    assert lines[0].split() == ["workers", "latency", "ms", "docs/s", "chunks/s", "ingest", "s"] + [
        "index",
        "s",
        "embed",
        "calls",
        "failed",
    ]
    assert len(lines) == 3 and len(lines[0]) == len(lines[2])


def _run_main(monkeypatch, *argv):
    # main() writes DATABASE_URL, EMBED_DIM and PROVIDER into os.environ; setenv first so the
    # monkeypatch restores whatever was there when the test ends.
    for k, v in {"DATABASE_URL": "x", "EMBED_DIM": str(DIM), "PROVIDER": "fake"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["load_test", *argv])
    db._pool = None
    load_test.main()
    get_settings.cache_clear()


def test_main_against_given_database_prints_table_and_writes_json(
    db_url, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "res.json"
    _run_main(
        monkeypatch,
        "--docs", "6",
        "--concurrency", "1,2",
        "--dim", str(DIM),
        "--corpus", str(tmp_path / "missing"),
        "--database-url", db_url,
        "--drop-tables",
        "--json", str(out),
    )  # fmt: skip
    text = capsys.readouterr().out
    assert "workers" in text and "Best:" in text and "50,000 docs" in text
    assert [r["concurrency"] for r in json.loads(out.read_text())] == [1, 2]


def test_main_refuses_to_drop_tables_without_the_flag(monkeypatch, db_url):
    monkeypatch.setattr(sys, "argv", ["load_test", "--database-url", db_url])
    with pytest.raises(SystemExit, match="--drop-tables"):
        load_test.main()


def test_main_starts_a_throwaway_database_by_default(tmp_path, monkeypatch, capsys):
    _run_main(
        monkeypatch,
        "--docs", "4",
        "--concurrency", "2",
        "--dim", str(DIM),
        "--corpus", str(tmp_path / "missing"),
    )  # fmt: skip
    assert "Best:" in capsys.readouterr().out
