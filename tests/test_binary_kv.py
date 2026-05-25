"""Binary-protocol KV tests against a live skeg-server."""
from __future__ import annotations

import pytest

from skeg import BinaryClient, NotConnected, ServerError


def _client(server: dict) -> BinaryClient:
    return BinaryClient.connect(server["host"], server["port"])  # type: ignore[arg-type]


def test_ping(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.ping()


def test_set_get_roundtrip(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"hello", b"world")
        assert c.get(b"hello") == b"world"


def test_get_missing_returns_none(binary_server: dict) -> None:
    with _client(binary_server) as c:
        assert c.get(b"definitely-not-set-1234") is None


def test_overwrite_key(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"k", b"v1")
        c.set(b"k", b"v2")
        assert c.get(b"k") == b"v2"


def test_delete_existing_returns_true(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"to-be-deleted", b"x")
        assert c.delete(b"to-be-deleted") is True
        assert c.get(b"to-be-deleted") is None


def test_delete_missing_returns_false(binary_server: dict) -> None:
    with _client(binary_server) as c:
        assert c.delete(b"never-existed") is False


def test_mget_preserves_order_and_marks_missing(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"a", b"av")
        c.set(b"c", b"cv")
        out = c.mget([b"a", b"b", b"c"])
        assert out == [b"av", None, b"cv"]


def test_large_value_roundtrip(binary_server: dict) -> None:
    big = b"x" * 65536
    with _client(binary_server) as c:
        c.set(b"big", big)
        assert c.get(b"big") == big


def test_no_reply_set_skips_ack(binary_server: dict) -> None:
    # Fire-and-forget SET. Subsequent GET should still see the value
    # because the shard mailbox serialises ops.
    with _client(binary_server) as c:
        c.set(b"async-key", b"async-val", no_reply=True)
        # Force a sync round-trip after the no_reply: PING flushes any
        # pending writes through the shard.
        c.ping()
        assert c.get(b"async-key") == b"async-val"


def test_closed_client_raises(binary_server: dict) -> None:
    c = _client(binary_server)
    c.close()
    with pytest.raises(NotConnected):
        c.ping()


def test_context_manager_closes(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"ctx-mgr", b"ok")
    with pytest.raises(NotConnected):
        c.ping()
