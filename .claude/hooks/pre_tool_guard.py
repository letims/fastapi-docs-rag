#!/usr/bin/env python3
"""PreToolUse guard: block destructive commands and protected paths (exit 2 = block, reason on stderr).

Defense in depth behind `permissions.deny` in .claude/settings.json: deny rules match the command
Claude writes, this also catches wrapped forms (`bash -c "..."`, `/usr/bin/curl`, `git -C . push -f`).
Stdlib only, so it runs with any python3 and stays fast (<50 ms)."""

import json
import re
import sys

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
    # The user commits herself: Claude never creates commits or rewrites history (CLAUDE.md, Workflow).
    (
        r"(^|[;&|(`]\s*|-c\s+['\"]?)git(\s+(-[Cc]\s+\S+|--?[\w-]+(=\S+)?))*\s+(commit|am|cherry-pick|rebase|merge|revert|push)\b",
        "git commit/history change (the user reviews and commits herself)",
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
]

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


def check(event: dict) -> str | None:
    """Return a block reason for this tool call, or None to let it through."""
    tool = event.get("tool_name", "")
    args = event.get("tool_input") or {}
    if tool == "Bash":
        cmd = args.get("command", "")
        for pattern, reason in BLOCKED_COMMANDS:
            if re.search(pattern, cmd):
                return f"blocked {reason}: `{cmd[:120]}`"
        if PROTECTED_PATH.search(cmd):
            return f"blocked: command touches a protected path (.env, secrets, credentials): `{cmd[:120]}`"
    elif tool in {"Read", "Edit", "Write"}:
        path = args.get("file_path", "")
        if PROTECTED_PATH.search(path):
            return f"blocked: {tool} on protected path {path}"
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        print("pre_tool_guard: unreadable hook input, blocking to be safe", file=sys.stderr)
        return 2
    reason = check(event)
    if reason:
        print(
            f"{reason}. Ask the user to do this themselves if it is really needed.", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
