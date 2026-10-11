from pathlib import Path

import pytest

from app.ingest.chunker import chunk_markdown, doc_hash, extract_title

SAMPLE = """---
title: x
---
# Dependencies { #dependencies }

Intro paragraph.

## What is DI { #what-is-di }

Some text.

```python
# not a heading, inside a fence

def f(): ...
```

{* ../../docs_src/x.py hl[1] *}

/// tip | Nice

Tip body.

///

### Deep { #deep }

Deep text.
"""


def test_title_and_anchor_stripping():
    assert extract_title(SAMPLE, "fallback") == "Dependencies"
    chunks = chunk_markdown(SAMPLE, "Dependencies")
    assert not any("{ #" in c.heading for c in chunks)
    assert any(c.heading == "Dependencies > What is DI > Deep" for c in chunks)


def test_fence_is_not_split_and_hash_comment_is_not_a_heading():
    chunks = chunk_markdown(SAMPLE, "Dependencies")
    code = [c for c in chunks if "def f()" in c.content]
    assert len(code) == 1
    assert "# not a heading, inside a fence" in code[0].content
    assert not any(c.heading.endswith("not a heading, inside a fence") for c in chunks)


def test_docs_syntax_noise_removed():
    text = "\n".join(c.content for c in chunk_markdown(SAMPLE, "Dependencies"))
    assert "docs_src" not in text
    assert "///" not in text
    assert "Tip:" in text


def test_oversize_code_block_is_split_with_balanced_fences():
    big = "# T\n\n## S\n\n```python\n" + "\n".join(f"x{i} = {i}" for i in range(400)) + "\n```\n"
    chunks = chunk_markdown(big, "T", max_chars=500)
    assert len(chunks) > 3
    for c in chunks:
        assert len(c.content) <= 520
        assert c.content.count("```") % 2 == 0


def test_chunk_hash_is_stable_and_content_sensitive():
    a = chunk_markdown(SAMPLE, "Dependencies")
    b = chunk_markdown(SAMPLE, "Dependencies")
    assert [c.content_hash for c in a] == [c.content_hash for c in b]
    c = chunk_markdown(SAMPLE.replace("Some text.", "Other text."), "Dependencies")
    assert {x.content_hash for x in a} != {x.content_hash for x in c}


def test_doc_hash_ignores_line_endings_and_outer_whitespace():
    assert doc_hash("a\r\nb\r\n") == doc_hash("a\nb\n")


CORPUS = Path(__file__).parent.parent / "corpus" / "fastapi"


@pytest.mark.skipif(not CORPUS.exists(), reason="run scripts/fetch_corpus.sh first")
def test_whole_corpus_chunks_sanely():
    files = list(CORPUS.rglob("*.md"))
    total = 0
    for p in files:
        text = p.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, extract_title(text, p.stem), 1800)
        total += len(chunks)
        for c in chunks:
            assert c.content.strip()
            assert len(c.content) <= 1900, (p, len(c.content))
            assert c.content.count("```") % 2 == 0, p
    assert total > len(files)
