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


def test_unclosed_front_matter_is_left_alone():
    chunks = chunk_markdown("---\nnot closed\n\n# T\n\nbody", "T")
    assert any("not closed" in c.content for c in chunks)


def test_extract_title_falls_back_when_no_h1():
    assert extract_title("no heading at all", "fb") == "fb"
    assert extract_title("## only a subsection\n", "fb") == "fb"


def test_skipped_heading_levels_still_build_a_breadcrumb():
    chunks = chunk_markdown("# T\n\n### Deep\n\ntext", "T")
    assert chunks[0].heading == "T > Deep"


def test_single_pathological_line_is_hard_split_without_loss():
    chunks = chunk_markdown("# T\n\n## S\n\n" + "x" * 5000, "T", max_chars=1000)
    assert all(len(c.content) <= 1000 for c in chunks)
    assert sum(len(c.content.replace("\n", "")) for c in chunks) == 5000


def test_oversize_plain_paragraph_splits_on_line_boundaries():
    para = "\n".join(f"line {i:03d} " + "y" * 90 for i in range(30))
    chunks = chunk_markdown(f"# T\n\n## S\n\n{para}", "T", max_chars=500)
    assert len(chunks) > 3 and all(len(c.content) <= 500 for c in chunks)


def test_tilde_fences_are_kept_balanced_when_split():
    big = "# T\n\n## S\n\n~~~text\n" + "\n".join(f"row {i}" for i in range(300)) + "\n~~~\n"
    chunks = chunk_markdown(big, "T", max_chars=400)
    assert len(chunks) > 3 and all(c.content.count("~~~") % 2 == 0 for c in chunks)


def test_blank_document_yields_no_chunks():
    assert chunk_markdown("\n\n   \n", "T") == []
