#!/usr/bin/env python3
"""PreToolUse guard: block destructive commands and protected paths (exit 2 = block, reason on stderr).

Defense in depth behind `permissions.deny` in .claude/settings.json: deny rules match the command
Claude writes, this also catches wrapped forms (`bash -c "..."`, `/usr/bin/curl`, `git -C . push -f`).
Git policy: Claude may commit (and amend, rebase, merge...) freely on a feature branch, so work is
reviewable as a branch diff and undoable with one reset. It never pushes, never writes to main/master,
never deletes or force-moves branches. Stdlib only, so it runs with any python3 and stays fast."""

import json
import re
import subprocess
import sys

# `git` as a command word (start, after ; & | ( `, or inside `sh -c '...'`), then any global options.
_GIT = r"(^|[;&|(`]\s*|-c\s+['\"]?)git(\s+(-[Cc]\s+\S+|--?[\w-]+(=\S+)?))*\s+"
# Commands that create commits or move the current branch: fine on a feature branch, never on main.
COMMIT_CMD = re.compile(_GIT + r"(commit|am|cherry-pick|rebase|merge|revert|pull)\b")
PROTECTED_BRANCHES = {"main", "master"}
SWITCH_TO_PROTECTED = re.compile(_GIT + r"(checkout|switch)\s+(main|master)\b")

# (pattern, reason). Matched against the whole Bash command, case-sensitive unless (?i).
BLOCKED_COMMANDS = [
    (
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|--recursive\s+--force|--force\s+--recursive)\b",
        "rm -rf",
    ),
    (
        r"\bgit\b[^;&|]*\bpush\b[^;&|]*(\s--force\b|\s--force-with-lease\b|\s-f\b|\s\+\S)",
        "force-push",
    ),
    (r"\bgit\b[^;&|]*\breset\s+--hard\b", "git reset --hard"),
    # The user pushes herself. Branch deletes / force-moves and ref rewrites can lose her work.
    (
        _GIT + r"(push|update-ref|branch\s+(-[a-zA-Z]*[Dfm]|--delete|--force|--move))\b",
        "push / branch rewrite",
    ),
    (
        r"(?i)\b(drop\s+(table|database|schema)|truncate\s+table|truncate\s+\w+\s*;)",
        "destructive SQL",
    ),
    (
        r"\bdocker\s+(compose\s+down\b[^;&|]*\s-v\b|volume\s+(rm|prune)\b)",
        "deleting the Postgres volume",
    ),
    # As a command word: start, after ; & | ( `, as a path (/usr/bin/curl) or inside `sh -c '...'`.
    (
        r"(^|[;&|(`]\s*|/|-c\s+['\"]?)(curl|wget)(\s|$)",
        "network fetch outside the sandbox allowlist",
    ),
    # docker runs outside the sandbox (excludedCommands), so `docker run -v ~:/h` could read ~/.ssh.
    # `docker compose up/down/ps/logs` (make up/down) stay allowed.
    (
        r"(^|[;&|(`]\s*|/|-c\s+['\"]?)docker(-compose|\s+compose|\s+container)?\s+(run|cp|exec)\b",
        "docker run/cp/exec (runs outside the sandbox and can mount your home directory)",
    ),
]

# Search/listing tools take paths in different fields; check only fields that hold a path or glob
# (not Grep's `pattern`, which is a regex over file contents).
PATH_FIELDS = {
    "Read": ("file_path",),
    "Edit": ("file_path",),
    "Write": ("file_path",),
    "NotebookEdit": ("notebook_path",),
    "Grep": ("path", "glob"),
    "Glob": ("path", "pattern"),
}

# Paths Claude must never read or write, by tool or by shell. `.env.example` stays readable.
PROTECTED_PATH = re.compile(
    r"(^|[\s/'\"=<>(:])("
    r"\.env(\.local|\.[\w-]+\.local)?(?![\w.-])"  # .env, .env.local, .env.prod.local
    r"|secrets/"
    r"|\.ssh/|\.ssh(?![\w.-])"
    r"|\.aws/|\.aws(?![\w.-])"
    r"|\.config/gcloud"
    r"|\.cache/api_cache\.sqlite"
    r")"
)


def current_branch(cwd: str | None) -> str | None:
    """Checked-out branch in `cwd`; "" when detached, None outside a git repo."""
    try:
        res = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def check(event: dict) -> str | None:
    """Return a block reason for this tool call, or None to let it through."""
    tool = event.get("tool_name", "")
    args = event.get("tool_input") or {}
    if tool == "Bash":
        cmd = args.get("command", "")
        for pattern, reason in BLOCKED_COMMANDS:
            if re.search(pattern, cmd):
                return f"blocked {reason}: `{cmd[:120]}`"
        if COMMIT_CMD.search(cmd):
            branch = current_branch(event.get("cwd"))
            if branch in PROTECTED_BRANCHES or SWITCH_TO_PROTECTED.search(cmd):
                return (
                    f"blocked: commits on {branch if branch in PROTECTED_BRANCHES else 'main'} are the user's. "
                    "Create a feature branch first (git switch -c claude/<topic>)"
                )
            if not branch:  # detached HEAD, not a repo, or git failed: fail closed
                return (
                    "blocked: cannot tell which branch is checked out (detached HEAD or git error). "
                    "Create a feature branch first (git switch -c claude/<topic>)"
                )
        if PROTECTED_PATH.search(cmd):
            return f"blocked: command touches a protected path (.env, secrets, credentials): `{cmd[:120]}`"
    elif tool in PATH_FIELDS:
        for field in PATH_FIELDS[tool]:
            path = str(args.get(field) or "")
            if PROTECTED_PATH.search(path):
                return f"blocked: {tool} on protected path {path}"
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("pre_tool_guard: unreadable hook input, blocking to be safe", file=sys.stderr)
        return 2
    try:
        reason = check(event)
    except Exception as exc:  # a crashing guard must block, not silently allow (only exit 2 blocks)
        print(
            f"pre_tool_guard: internal error ({type(exc).__name__}), blocking to be safe",
            file=sys.stderr,
        )
        return 2
    if reason:
        print(
            f"{reason}. Ask the user to do this themselves if it is really needed.", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
