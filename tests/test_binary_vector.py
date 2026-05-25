"""Binary-protocol vector tests. Exercises VINDEX/VSET/VGET/VDEL/VSEARCH
against an in-RAM flat index (the disk-backed Vamana path requires an
offline build, so the unit tests go through `BACKEND_FLAT`)."""
from __future__ import annotations

import random

import pytest

from skeg import BinaryClient, Hit, VectorBackend, VectorKind


def _client(server: dict) -> BinaryClient:
    return BinaryClient.connect(server["host"], server["port"])  # type: ignore[arg-type]


def _rand_vec(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    # L2-normalize so cosine and dot are consistent.
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm > 0 else v


def test_vindex_create_and_drop(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("test-create-drop", dim=8, kind="int8", backend="flat")
        c.vindex_drop("test-create-drop")


def test_vset_vget_roundtrip(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("rt", dim=8, kind="f32", backend="flat")
        v = _rand_vec(8, seed=1)
        c.vset("rt", 42, v)
        got = c.vget("rt", 42)
        assert got is not None
        assert len(got) == 8
        # f32 round-trip: relax tolerance to account for IEEE-754 quirks.
        for a, b in zip(v, got):
            assert abs(a - b) < 1e-6
        c.vindex_drop("rt")


def test_vget_missing_returns_none(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("missing", dim=4, kind="f32", backend="flat")
        assert c.vget("missing", 999) is None
        c.vindex_drop("missing")


def test_vdel_then_vget_returns_none(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("dv", dim=4, kind="f32", backend="flat")
        c.vset("dv", 1, _rand_vec(4, seed=2))
        assert c.vget("dv", 1) is not None
        assert c.vdel("dv", 1) is True
        assert c.vget("dv", 1) is None
        c.vindex_drop("dv")


def test_vsearch_finds_self_at_top_score(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("self", dim=16, kind="f32", backend="flat")
        v = _rand_vec(16, seed=3)
        c.vset("self", 1, v)
        hits = c.vsearch("self", v, k=1)
        assert len(hits) == 1
        assert hits[0].id == 1
        # Cosine score for a unit vector with itself = 1.0.
        assert hits[0].score > 0.99
        c.vindex_drop("self")


def test_vsearch_returns_k_results(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("kr", dim=16, kind="f32", backend="flat")
        for i in range(20):
            c.vset("kr", i, _rand_vec(16, seed=i + 100))
        hits = c.vsearch("kr", _rand_vec(16, seed=999), k=5)
        assert len(hits) == 5
        assert all(isinstance(h, Hit) for h in hits)
        # Cosine score: higher = closer, so the list should be sorted
        # in descending order (best match first).
        for a, b in zip(hits, hits[1:]):
            assert a.score >= b.score - 1e-6
        c.vindex_drop("kr")


def test_vector_enums_accept_strings(binary_server: dict) -> None:
    # Pass kind/backend as strings - the client should map them.
    with _client(binary_server) as c:
        c.vindex_create("enums", dim=4, kind="int8", backend="flat")
        c.vindex_drop("enums")


def test_vector_enums_accept_enum_values(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create(
            "enums2", dim=4,
            kind=VectorKind.BINARY, backend=VectorBackend.FLAT,
        )
        c.vindex_drop("enums2")
