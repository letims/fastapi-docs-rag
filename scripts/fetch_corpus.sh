#!/usr/bin/env bash
# Fetch the English FastAPI docs (markdown only) into ./corpus/fastapi (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="$(mktemp -d)"
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi "$TMP/fastapi"
git -C "$TMP/fastapi" sparse-checkout set docs/en/docs
rm -rf corpus/fastapi && mkdir -p corpus/fastapi
(cd "$TMP/fastapi/docs/en/docs" && find . -type f -name '*.md' -print0 | cpio -0pdm "$OLDPWD/corpus/fastapi" 2>/dev/null)
rm -rf "$TMP"
echo "Fetched $(find corpus/fastapi -name '*.md' | wc -l) markdown files, $(du -sh corpus/fastapi | cut -f1)"
