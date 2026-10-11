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
        return await (
            await c.execute("SELECT source, title FROM ingest_jobs ORDER BY source")
        ).fetchall()


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


def test_select_subset_keeps_expected_docs_and_is_deterministic():
    docs = [(f"d{i}.md", f"D{i}", "x") for i in range(20)]
    expected = {"d3.md", "d17.md", "missing.md"}  # a stale expectation must not break anything
    a = enqueue_corpus.select_subset(docs, expected, total=6)
    assert len(a) == 6 and {"d3.md", "d17.md"} <= {d[0] for d in a}
    assert a == enqueue_corpus.select_subset(docs, expected, total=6)  # same seed, same subset
    assert a != enqueue_corpus.select_subset(docs, expected, total=6, seed=99)
    assert a == sorted(a)
    small = enqueue_corpus.select_subset(docs, expected, total=1)  # never drops expected docs
    assert {d[0] for d in small} == {"d3.md", "d17.md"}


async def test_enqueue_corpus_subset_flag_uses_the_eval_questions(
    pool, tmp_path, monkeypatch, capsys
):
    (tmp_path / "advanced").mkdir()
    (tmp_path / "advanced" / "events.md").write_text("# Events\n\nstartup")  # named by eval q01
    for i in range(5):
        (tmp_path / f"other{i}.md").write_text(f"# Other {i}\n\ntext")
    monkeypatch.setattr(sys, "argv", ["enqueue_corpus", str(tmp_path), "--subset", "3"])
    await enqueue_corpus.main()
    assert "3 docs found, 3 new jobs enqueued" in capsys.readouterr().out
    assert "advanced/events.md" in [r["source"] for r in await _jobs()]


async def test_quota_status_script(pool, capsys):
    from app.config import get_settings
    from app.quota import DailyBudget
    from scripts import quota

    # The counter is per provider (the test env runs the fake one): "llm:<backend>"
    await DailyBudget(get_settings().usage_key("llm"), 100, db.open_pool).reserve(7)
    usage = await quota.usage_today(pool)
    assert usage["llm"] == {"used": 7, "limit": 100, "remaining": 93}
    assert usage["embed"] == {"used": 0, "limit": 100, "remaining": 100}
    await quota.main()
    out = capsys.readouterr().out
    assert "llm" in out and "used    7 / 100" in out and "resets in" in out


async def test_quota_status_reads_the_configured_provider_counter(pool, monkeypatch):
    from app.config import get_settings
    from app.quota import DailyBudget
    from scripts import quota

    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    get_settings.cache_clear()
    await DailyBudget("llm:groq", 1000, db.open_pool).reserve(5)
    await DailyBudget("llm:gemini", 100, db.open_pool).reserve(50)  # other vendor: not counted
    usage = await quota.usage_today(pool)
    assert usage["llm"] == {"used": 5, "limit": 1000, "remaining": 995}
    assert usage["embed"]["limit"] == 1_000_000
