import sys

import app.db as db
from scripts import enqueue_corpus


def _make_corpus(root):
    (root / "a.md").write_text("# Alpha\n\ntext")
    (root / "sub").mkdir()
    (root / "sub" / "b.md").write_text("no h1 here")
    (root / "release-notes.md").write_text("# excluded by config")
    (root / "notes.txt").write_text("not markdown")


async def _jobs():
    p = await db.open_pool()  # main() closed the global pool; this reopens it
    async with p.connection() as c:
        return await (await c.execute("SELECT source, title FROM ingest_jobs ORDER BY source")).fetchall()


async def test_enqueue_corpus_skips_excluded_and_non_markdown(pool, tmp_path, monkeypatch, capsys):
    _make_corpus(tmp_path)
    monkeypatch.setattr(sys, "argv", ["enqueue_corpus", str(tmp_path)])
    await enqueue_corpus.main()
    assert "2 docs found, 2 new jobs enqueued" in capsys.readouterr().out
    rows = await _jobs()
    assert [(r["source"], r["title"]) for r in rows] == [("a.md", "Alpha"), ("sub/b.md", "b")]


async def test_enqueue_corpus_limit_and_rerun_is_idempotent(pool, tmp_path, monkeypatch, capsys):
    _make_corpus(tmp_path)
    monkeypatch.setattr(sys, "argv", ["enqueue_corpus", str(tmp_path), "--limit", "1"])
    await enqueue_corpus.main()
    assert "1 docs found, 1 new jobs enqueued" in capsys.readouterr().out
    await db.open_pool()
    await enqueue_corpus.main()
    assert "1 docs found, 0 new jobs enqueued" in capsys.readouterr().out
