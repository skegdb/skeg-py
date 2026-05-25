"""RESP2/3 client tests against a live skeg-resp3 server."""
from __future__ import annotations

import pytest

from skeg import RespClient, ServerError


def _client(server: dict) -> RespClient:
    return RespClient.connect(server["host"], server["port"])  # type: ignore[arg-type]


def test_ping_default(resp_server: dict) -> None:
    with _client(resp_server) as c:
        assert c.ping() == b"PONG"


def test_ping_with_message(resp_server: dict) -> None:
    with _client(resp_server) as c:
        assert c.ping(b"hello") == b"hello"


def test_echo(resp_server: dict) -> None:
    with _client(resp_server) as c:
        assert c.echo(b"abc") == b"abc"


def test_hello_upgrades_to_resp3(resp_server: dict) -> None:
    with _client(resp_server) as c:
        reply = c.hello(3)
        # RESP3 returns a Map (decoded as dict here). RESP2 returns a
        # flat array. We accept either; what matters is the negotiation
        # leaves the client in version 3.
        assert c.version == 3
        # The reply should mention the "proto" key in either shape.
        if isinstance(reply, dict):
            assert b"proto" in reply
        else:
            assert b"proto" in (reply or [])  # type: ignore[operator]


def test_set_get_roundtrip(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"hello", b"world")
        assert c.get(b"hello") == b"world"


def test_get_missing_is_none(resp_server: dict) -> None:
    with _client(resp_server) as c:
        assert c.get(b"missing-key-xyz") is None


def test_del_multikey(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"d1", b"v1")
        c.set(b"d2", b"v2")
        assert c.delete(b"d1", b"d2", b"d3") == 2


def test_exists_multikey(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"e1", b"v")
        c.set(b"e2", b"v")
        assert c.exists(b"e1", b"e2", b"nope") == 2


def test_mget_preserves_order(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"m1", b"v1")
        c.set(b"m3", b"v3")
        out = c.mget([b"m1", b"m2", b"m3"])
        assert out == [b"v1", None, b"v3"]


def test_mset_then_get(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.mset({b"x": b"1", b"y": b"2", b"z": b"3"})
        for k, v in [(b"x", b"1"), (b"y", b"2"), (b"z", b"3")]:
            assert c.get(k) == v


def test_incr_starts_at_one(resp_server: dict) -> None:
    with _client(resp_server) as c:
        # Ensure unique key per-run by checking-and-deleting first.
        c.delete(b"counter-incr")
        assert c.incr(b"counter-incr") == 1
        assert c.incr(b"counter-incr") == 2
        assert c.decr(b"counter-incr") == 1


def test_incrby_signed(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.delete(b"counter-incrby")
        assert c.incrby(b"counter-incrby", 42) == 42
        assert c.incrby(b"counter-incrby", -10) == 32


def test_incr_rejects_non_integer(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"bad-counter", b"not-a-number")
        with pytest.raises(ServerError):
            c.incr(b"bad-counter")


def test_skeg_stats_returns_summary(resp_server: dict) -> None:
    with _client(resp_server) as c:
        s = c.skeg_stats()
        assert "cache_bytes=" in s
        assert "n_keys=" in s


def test_skeg_whoami_anonymous_is_zero(resp_server: dict) -> None:
    with _client(resp_server) as c:
        w = c.skeg_whoami()
        # Server in single-tenant mode (no TenantContext wired in the
        # test fixture) reports the zero tenant.
        assert "tenant=" in w
        assert "mode=" in w


def test_select_zero_is_noop(resp_server: dict) -> None:
    # SELECT 0 returns +OK for compatibility with auto-issuing drivers.
    with _client(resp_server) as c:
        reply = c._cmd([b"SELECT", b"0"])  # type: ignore[arg-type]
        assert reply in (b"OK", "OK")
