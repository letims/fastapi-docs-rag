#!/usr/bin/env python3
"""Stop hook: run the fast unit tests (no Postgres, ~5 s) when Python files changed; block on failure.

Exit 2 keeps Claude working with the failure on stderr. `stop_hook_active` is true when Claude is
already continuing because of this hook: then we let it stop, so a test it cannot fix never loops."""

import json
import subprocess
import sys

UNIT_TESTS = [
    "uv",
    "run",
    "--quiet",
    "pytest",
    "-q",
    "-x",
    "-m",
    "not db",
    "-p",
    "no:cacheprovider",
]


def python_changed() -> bool:
    """True if any tracked or untracked .py file differs from HEAD."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True
    ).stdout
    return any(line.rstrip().endswith(".py") for line in out.splitlines())


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}
    if event.get("stop_hook_active"):
        return 0
    if not python_changed():
        return 0
    res = subprocess.run(UNIT_TESTS, capture_output=True, text=True)
    if res.returncode != 0:
        tail = "\n".join((res.stdout + res.stderr).splitlines()[-40:])
        print(
            f"Unit tests fail (`{' '.join(UNIT_TESTS)}`). Fix before finishing:\n{tail}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
