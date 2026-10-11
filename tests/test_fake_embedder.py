import math

from app.providers.fake import fake_embedding


def cos(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_unit_norm_and_similarity_ordering():
    q = fake_embedding("how do I run code at startup", 256)
    near = fake_embedding("run code at startup and shutdown with lifespan", 256)
    far = fake_embedding("serving static files from a directory", 256)
    assert math.isclose(sum(x * x for x in q), 1.0, rel_tol=1e-6)
    assert cos(q, near) > cos(q, far)
