from app.cache import DiskCache, make_key, pack_vector, unpack_vector


def test_vector_roundtrip_is_float32_accurate():
    v = [0.1, -0.5, 3.25, 0.0]
    got = unpack_vector(pack_vector(v))
    assert len(got) == 4 and all(abs(a - b) < 1e-6 for a, b in zip(got, v, strict=True))


def test_make_key_depends_on_every_part():
    assert make_key("a", "b") == make_key("a", "b")
    assert make_key("a", "b") != make_key("a", "c")
    assert make_key("ab", "") != make_key("a", "b")  # parts are delimited, not just concatenated


def test_cache_persists_across_reopen(tmp_path):
    path = tmp_path / "sub" / "cache.sqlite"  # parent dir is created on demand
    c = DiskCache(path)
    assert c.get("k") is None and len(c) == 0
    c.set("k", b"v1")
    c.set("k", b"v2")  # replace, not duplicate
    assert c.get("k") == b"v2" and len(c) == 1
    c.close()
    assert DiskCache(path).get("k") == b"v2"  # survives a restart: the whole point


def test_in_memory_cache():
    c = DiskCache(":memory:")
    c.set("a", b"x")
    assert c.get("a") == b"x"
