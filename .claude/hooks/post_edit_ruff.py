#!/usr/bin/env python3
"""PostToolUse (Edit|Write): ruff-fix and format the edited .py file; report what ruff can't fix.

Exit 2 shows stderr to Claude (the edit already happened), so it fixes the rest in the same turn."""

import json
import subprocess
import sys
from pathlib import Path


def ruff(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", "run", "--quiet", "ruff", *args], capture_output=True, text=True)


def main() -> int:
    try:
        path = Path(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))
    except json.JSONDecodeError:
        return 0
    if path.suffix != ".py" or not path.is_file():
        return 0
    ruff("check", "--fix", "--quiet", str(path))
    ruff("format", "--quiet", str(path))
    left = ruff("check", "--quiet", str(path))
    if left.returncode != 0:
        print(
            f"ruff found issues it could not fix in {path}:\n{left.stdout[-2000:]}", file=sys.stderr
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
