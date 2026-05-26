"""PyO3 backend smoke tests.

We only mark these collectable when the native module imports cleanly;
otherwise pytest skips the whole file. This way pure-Python installs
continue to pass without a Rust toolchain.
"""
from __future__ import annotations

import pytest

skeg = pytest.importorskip("skeg")

if not getattr(skeg, "_HAS_NATIVE", False):
    pytest.skip("PyO3 backend not built; pip install -e . or maturin develop",
                allow_module_level=True)

from skeg._native import BinaryClient as NativeClient  # noqa: E402


def _native(server: dict) -> NativeClient:
    return NativeClient(server["host"], server["port"])  # type: ignore[arg-type]


def test_native_ping(binary_server: dict) -> None:
    c = _native(binary_server)
    c.ping()
    c.close()


def test_native_set_get_roundtrip(binary_server: dict) -> None:
    c = _native(binary_server)
    try:
        c.set(b"native-k", b"native-v")
        assert c.get(b"native-k") == b"native-v"
    finally:
        c.close()


def test_native_get_missing_returns_none(binary_server: dict) -> None:
    c = _native(binary_server)
    try:
        assert c.get(b"never-set-by-native-12345") is None
    finally:
        c.close()


def test_native_mget(binary_server: dict) -> None:
    c = _native(binary_server)
    try:
        c.set(b"nm1", b"a")
        c.set(b"nm3", b"c")
        out = c.mget([b"nm1", b"nm2", b"nm3"])
        assert out == [b"a", None, b"c"]
    finally:
        c.close()


def test_native_vector_roundtrip(binary_server: dict) -> None:
    c = _native(binary_server)
    try:
        c.vindex_create("native-vix", 8, "f32", "flat")
        v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        c.vset("native-vix", 1, v)
        got = c.vget("native-vix", 1)
        assert got is not None
        for a, b in zip(v, got):
            assert abs(a - b) < 1e-6
        hits = c.vsearch("native-vix", v, 1)
        assert len(hits) == 1
        assert hits[0].id == 1
        assert hits[0].score > 0.99
        c.vindex_drop("native-vix")
    finally:
        c.close()


def test_native_context_manager(binary_server: dict) -> None:
    with _native(binary_server) as c:
        c.set(b"ctx-native", b"ok")
        assert c.get(b"ctx-native") == b"ok"
