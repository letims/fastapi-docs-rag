"""Markdown-aware chunker.

Goals: never split inside a fenced code block unless it is too big for one chunk, keep the
heading breadcrumb with every chunk (it is embedded together with the text, which is a cheap
form of "contextual" chunking), and drop docs-site syntax that is noise for retrieval."""

import hashlib
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")  # "{ #some-anchor }" suffix on every heading
_INCLUDE = re.compile(r"^\{\*.*\*\}\s*$")  # {* ../../docs_src/x.py hl[1] *}: code not in corpus
_FENCE = re.compile(r"^\s*(```|~~~)")
_ADMONITION = re.compile(r"^///\s*(\w+)?")  # "/// tip | Title" ... "///"


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a document.

    Attributes:
        index: Position of the chunk within its document, from 0.
        heading: Breadcrumb of the document title and section headings.
        content: The chunk text (without the heading).
    """

    index: int
    heading: str  # "Title > Section > Subsection"
    content: str

    @property
    def embed_text(self) -> str:
        """The text that gets embedded: heading breadcrumb, blank line, content."""
        return f"{self.heading}\n\n{self.content}"

    @property
    def content_hash(self) -> str:
        """SHA-256 of `embed_text`; the key for reusing a stored embedding."""
        return sha256(self.embed_text)


def sha256(text: str) -> str:
    """Hex SHA-256 digest of `text` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Convert CRLF to LF and strip surrounding whitespace, so hashes ignore line-ending noise."""
    return text.replace("\r\n", "\n").strip()


def doc_hash(text: str) -> str:
    """Hash of the normalized document text; used to skip documents that have not changed."""
    return sha256(normalize(text))


def _strip_front_matter(text: str) -> str:
    """Remove a leading `---` YAML block, if there is one."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :]
    return text


def extract_title(text: str, fallback: str) -> str:
    """Find the document title: the first level-1 heading, with any `{ #anchor }` removed.

    Args:
        text: Markdown source.
        fallback: Returned when the document has no level-1 heading (e.g. the file name).

    Returns:
        The title text.
    """
    for line in _strip_front_matter(normalize(text)).split("\n"):
        m = _HEADING.match(line)
        if m and len(m.group(1)) == 1:
            return _ANCHOR.sub("", m.group(2)).strip()
    return fallback


def _clean_line(line: str) -> str | None:
    """Return the cleaned line, or None to drop it."""
    if _INCLUDE.match(line):
        return None
    m = _ADMONITION.match(line)
    if m:
        return f"{m.group(1).capitalize()}:" if m.group(1) else None
    return line


def _sections(text: str) -> list[tuple[list[str], list[str]]]:
    """Split into (heading_path, blocks). Headings inside code fences are not headings.

    A block is a run of non-blank lines (a paragraph, list or whole fenced code block). The
    heading path excludes the level-1 title, which the caller adds back. Docs-site syntax is
    cleaned on the way (see `_clean_line`).
    """
    sections: list[tuple[list[str], list[list[str]]]] = []
    path: list[str] = []
    blocks: list[list[str]] = []
    cur: list[str] = []
    in_fence = False

    def end_block():
        nonlocal cur
        if any(l.strip() for l in cur):
            blocks.append(cur)
        cur = []

    def end_section():
        end_block()
        if blocks:
            sections.append((list(path), list(blocks)))
        blocks.clear()

    for raw in _strip_front_matter(normalize(text)).split("\n"):
        line = _clean_line(raw)
        if line is None:
            continue
        if _FENCE.match(line):
            in_fence = not in_fence
            cur.append(line)
            continue
        if in_fence:
            cur.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            end_section()
            level, title = len(m.group(1)), _ANCHOR.sub("", m.group(2)).strip()
            path = [] if level == 1 else path[: level - 2] + [title]
            continue
        if not line.strip():
            end_block()
        else:
            cur.append(line)
    end_section()
    return [(p, ["\n".join(b) for b in bl]) for p, bl in sections]  # type: ignore[misc]


def _split_oversize(block: str, max_chars: int) -> list[str]:
    """Split a block longer than `max_chars` on line boundaries.

    A fenced code block is split into several complete fenced blocks, so each piece still
    renders as code. A single line longer than the budget is cut mid-line as a last resort.

    Returns:
        The block itself in a one-item list if it already fits, otherwise the pieces.
    """
    if len(block) <= max_chars:
        return [block]
    lines = block.split("\n")
    fence = lines[0] if _FENCE.match(lines[0]) else None
    if fence:
        lines = lines[1:]
        if lines and _FENCE.match(lines[-1]):
            lines = lines[:-1]
    budget = max_chars - (2 * len(fence) + 2 if fence else 0)
    pieces, cur, size = [], [], 0
    for line in lines:
        while len(line) > budget:  # pathological single long line
            if cur:
                pieces.append("\n".join(cur))
                cur, size = [], 0
            pieces.append(line[:budget])
            line = line[budget:]
        if size + len(line) + 1 > budget and cur:
            pieces.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        pieces.append("\n".join(cur))
    return [f"{fence}\n{p}\n{fence.strip()[:3]}" for p in pieces] if fence else pieces


def chunk_markdown(text: str, title: str, max_chars: int = 1800) -> list[Chunk]:
    """Split a markdown document into chunks of roughly `max_chars` characters.

    Chunks never cross a heading. Within a section, blocks are packed together until adding
    another would exceed `max_chars`. Changing this function (or `max_chars`) changes chunk
    hashes and so forces a full re-embed of the corpus.

    Args:
        text: Markdown source.
        title: Document title, used as the first part of every chunk's heading.
        max_chars: Soft limit on content length per chunk.

    Returns:
        Chunks in document order, with `index` running from 0.
    """
    chunks: list[Chunk] = []
    for path, blocks in _sections(text):
        heading = " > ".join([title, *path])
        buf: list[str] = []
        size = 0

        def flush():
            nonlocal buf, size
            if buf:
                chunks.append(Chunk(len(chunks), heading, "\n\n".join(buf)))
            buf, size = [], 0

        for block in blocks:
            for piece in _split_oversize(block, max_chars):
                if buf and size + len(piece) + 2 > max_chars:
                    flush()
                buf.append(piece)
                size += len(piece) + 2
        flush()
    return chunks
