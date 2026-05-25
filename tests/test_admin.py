"""Admin / observability commands: VINDEX LIST + SHARDS.

These commands ship with skeg ≥ v0.2 fase 1. They power skeg-top and
any external dashboard tool.
"""
from __future__ import annotations

import random

import pytest

from skeg import BinaryClient


# Test-only convention: all VINDEX created by this suite start with this
# prefix, so the cleanup loop can scope its drops and not wipe unrelated
# data on a server that happens to host real VINDEX. The library API has
# no notion of this prefix - end users name VINDEX however they want.
_TEST_PREFIX = "pytst-"


def _client(server: dict) -> BinaryClient:
    return BinaryClient.connect(server["host"], server["port"])  # type: ignore[arg-type]


def _vec(dim: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def test_shards_returns_one_row_per_shard(binary_server: dict) -> None:
    with _client(binary_server) as c:
        rows = c.shards()
        assert len(rows) >= 1
        for r in rows:
            assert set(r.keys()) >= {
                "shard_id", "cache_bytes", "cache_evictions",
                "n_keys", "cache_budget",
            }
            assert r["cache_budget"] > 0


def test_shards_sum_matches_total_keys_after_writes(binary_server: dict
                                                     ) -> None:
    with _client(binary_server) as c:
        for i in range(50):
            c.set(f"sk-{i}".encode(), b"v")
        rows = c.shards()
        total = sum(r["n_keys"] for r in rows)
        # All 50 keys are stored exactly once across the shards.
        assert total >= 50


def test_vindex_list_returns_only_owned_after_cleanup(binary_server: dict) -> None:
    # After we drop every test-owned VINDEX, no entry with our prefix
    # remains. Other VINDEX (created outside this suite) are left
    # untouched - we never inspect them.
    with _client(binary_server) as c:
        for v in c.vindex_list():
            if v["name"].startswith(_TEST_PREFIX):
                try:
                    c.vindex_drop(v["name"])
                except Exception:
                    pass
        owned = [r for r in c.vindex_list() if r["name"].startswith(_TEST_PREFIX)]
        assert owned == []


def test_vindex_list_reports_flat_and_disk(binary_server: dict) -> None:
    with _client(binary_server) as c:
        # Clean slate (in-fixture server may have leftovers).
        # Defensive cleanup: only drop VINDEX names this test suite
        # owns (prefix-matched). Avoids wiping unrelated indexes if a
        # developer accidentally runs the suite against a populated
        # server. Library callers can use any names they like; this is
        # test-only convention.
        for v in c.vindex_list():
            if v["name"].startswith(_TEST_PREFIX):
                try:
                    c.vindex_drop(v["name"])
                except Exception:
                    pass
        c.vindex_create(_TEST_PREFIX + "vl-flat", dim=4, kind="f32", backend="flat")
        c.vindex_create(_TEST_PREFIX + "vl-disk", dim=8, kind="int8", backend="disk_vamana")
        rows = c.vindex_list()
        # Filter to the entries this test owns; the rest of the list
        # might include unrelated VINDEX created outside the suite.
        owned = [r["name"] for r in rows if r["name"].startswith(_TEST_PREFIX)]
        assert _TEST_PREFIX + "vl-flat" in owned
        assert _TEST_PREFIX + "vl-disk" in owned
        # Order: VINDEX LIST returns alphabetical, per server contract.
        assert owned == sorted(owned)
        flat = next(r for r in rows if r["name"] == _TEST_PREFIX + "vl-flat")
        assert flat["dim"] == 4
        assert flat["backend"] == 0  # flat
        disk = next(r for r in rows if r["name"] == _TEST_PREFIX + "vl-disk")
        assert disk["dim"] == 8
        assert disk["backend"] == 1  # disk-vamana
        c.vindex_drop(_TEST_PREFIX + "vl-flat")
        c.vindex_drop(_TEST_PREFIX + "vl-disk")


def test_vindex_list_n_vectors_reflects_vset(binary_server: dict) -> None:
    with _client(binary_server) as c:
        # Defensive cleanup: only drop VINDEX names this test suite
        # owns (prefix-matched). Avoids wiping unrelated indexes if a
        # developer accidentally runs the suite against a populated
        # server. Library callers can use any names they like; this is
        # test-only convention.
        for v in c.vindex_list():
            if v["name"].startswith(_TEST_PREFIX):
                try:
                    c.vindex_drop(v["name"])
                except Exception:
                    pass
        c.vindex_create(_TEST_PREFIX + "vl-count", dim=4, kind="f32", backend="flat")
        for i in range(7):
            c.vset(_TEST_PREFIX + "vl-count", i, _vec(4, seed=i + 1000))
        rows = c.vindex_list()
        row = next(r for r in rows if r["name"] == _TEST_PREFIX + "vl-count")
        assert row["n_vectors"] == 7
        c.vindex_drop(_TEST_PREFIX + "vl-count")


def test_vindex_list_after_drop(binary_server: dict) -> None:
    with _client(binary_server) as c:
        # Defensive cleanup: only drop VINDEX names this test suite
        # owns (prefix-matched). Avoids wiping unrelated indexes if a
        # developer accidentally runs the suite against a populated
        # server. Library callers can use any names they like; this is
        # test-only convention.
        for v in c.vindex_list():
            if v["name"].startswith(_TEST_PREFIX):
                try:
                    c.vindex_drop(v["name"])
                except Exception:
                    pass
        c.vindex_create(_TEST_PREFIX + "vl-drop", dim=2, kind="f32", backend="flat")
        assert any(r["name"] == _TEST_PREFIX + "vl-drop" for r in c.vindex_list())
        c.vindex_drop(_TEST_PREFIX + "vl-drop")
        assert all(r["name"] != _TEST_PREFIX + "vl-drop" for r in c.vindex_list())
