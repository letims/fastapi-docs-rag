"""Persistent on-disk cache for API results (SQLite, stdlib only).

With 100 requests/day, every call must be paid for at most once, EVEN across database resets,
`docker compose down -v`, and re-runs of the eval. Embeddings are keyed by (model, dim, kind, text)
and LLM answers by (model, temperature, json, system, prompt), so identical inputs are free forever.

Caveat: caching LLM answers hides run-to-run variance. Disable (API_CACHE=false) for final numbers."""

import array
import hashlib
import sqlite3
import threading
from pathlib import Path


def make_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def pack_vector(v: list[float]) -> bytes:
    return array.array("f", v).tobytes()  # float32: half the space, plenty for cosine search


def unpack_vector(blob: bytes) -> list[float]:
    a = array.array("f")
    a.frombytes(blob)
    return a.tolist()


class DiskCache:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        self._db.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: bytes) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)", (key, value))
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT count(*) FROM kv").fetchone()[0]

    def close(self) -> None:
        self._db.close()
