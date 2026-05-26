"""RESP2/3 client edge cases.

Covers protocol negotiation, large values, unicode, error frames,
multi-key MGET sizing, integer overflow on INCR, and SKEG.* namespace
behaviour for unknown verbs.
"""
from __future__ import annotations

import pytest

from skeg import RespClient, ServerError


def _client(server: dict) -> RespClient:
    return RespClient.connect(server["host"], server["port"])  # type: ignore[arg-type]


# ── negotiation ──────────────────────────────────────────────────────


def test_resp2_default_before_hello(resp_server: dict) -> None:
    with _client(resp_server) as c:
        assert c.version == 2


def test_hello_3_returns_protocol_map(resp_server: dict) -> None:
    with _client(resp_server) as c:
        reply = c.hello(3)
        assert c.version == 3
        # RESP3 HELLO reply is a map; in RESP2 it's flattened to a list.
        assert isinstance(reply, (dict, list))


def test_hello_2_keeps_resp2(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.hello(2)
        assert c.version == 2


def test_ping_empty_default_message(resp_server: dict) -> None:
    with _client(resp_server) as c:
        # PING with no arg returns the literal +PONG.
        assert c.ping() == b"PONG"


# ── data shape ───────────────────────────────────────────────────────


def test_empty_value_via_resp(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"resp-empty", b"")
        assert c.get(b"resp-empty") == b""


def test_unicode_key_via_resp(resp_server: dict) -> None:
    k = "città".encode("utf-8")
    v = "caffè".encode("utf-8")
    with _client(resp_server) as c:
        c.set(k, v)
        assert c.get(k) == v


def test_binary_value_with_crlf(resp_server: dict) -> None:
    # RESP framing uses \r\n; the bulk-string encoding carries the
    # length prefix so embedded CRLFs must survive intact.
    v = b"line1\r\nline2\r\n\x00\xff"
    with _client(resp_server) as c:
        c.set(b"crlf-val", v)
        assert c.get(b"crlf-val") == v


def test_large_value_64kib_via_resp(resp_server: dict) -> None:
    # 64 KiB exercises the readexact recv loop without ballooning the
    # CI test time.
    v = b"R" * (64 * 1024)
    with _client(resp_server) as c:
        c.set(b"resp-64kib", v)
        assert c.get(b"resp-64kib") == v


# ── multi-key paths ──────────────────────────────────────────────────


def test_mget_empty_input_raises(resp_server: dict) -> None:
    with _client(resp_server) as c:
        # The server rejects MGET with zero keys; we surface that as
        # ServerError, not an empty list.
        with pytest.raises(ServerError):
            c.mget([])


def test_mget_64_keys(resp_server: dict) -> None:
    with _client(resp_server) as c:
        for i in range(64):
            c.set(f"resp-mget-{i}".encode(), f"v{i}".encode())
        out = c.mget([f"resp-mget-{i}".encode() for i in range(64)])
        assert out == [f"v{i}".encode() for i in range(64)]


def test_mset_then_mget_preserves_pairs(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.mset({b"mk1": b"v1", b"mk2": b"v2", b"mk3": b"v3"})
        out = c.mget([b"mk1", b"mk2", b"mk3"])
        assert out == [b"v1", b"v2", b"v3"]


def test_exists_partial_count(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"exists-a", b"x")
        c.set(b"exists-c", b"x")
        # 2 present, 1 missing -> count 2.
        assert c.exists(b"exists-a", b"exists-b", b"exists-c") == 2


def test_del_count_of_missing_is_zero(resp_server: dict) -> None:
    with _client(resp_server) as c:
        # Three keys all missing -> DEL returns 0.
        assert c.delete(b"never-a", b"never-b", b"never-c") == 0


# ── counter semantics ───────────────────────────────────────────────


def test_incr_then_get_returns_string(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.delete(b"counter-string")
        n = c.incr(b"counter-string")
        assert n == 1
        # The stored value is the integer rendered as a UTF-8 string.
        assert c.get(b"counter-string") == b"1"


def test_incrby_then_decrby_returns_to_zero(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.delete(b"counter-zero")
        assert c.incrby(b"counter-zero", 100) == 100
        assert c.incrby(b"counter-zero", -100) == 0


def test_incr_overflow_rejected(resp_server: dict) -> None:
    with _client(resp_server) as c:
        c.set(b"counter-max", b"9223372036854775807")  # i64::MAX
        with pytest.raises(ServerError):
            c.incr(b"counter-max")


# ── SKEG.* namespace ────────────────────────────────────────────────


def test_skeg_stats_contains_expected_fields(resp_server: dict) -> None:
    with _client(resp_server) as c:
        s = c.skeg_stats()
        for field in ("cache_bytes=", "evictions=", "n_keys=", "budget="):
            assert field in s


def test_skeg_unknown_verb_returns_error(resp_server: dict) -> None:
    with _client(resp_server) as c:
        with pytest.raises(ServerError, match="SKEG.WHATEVER"):
            c._cmd([b"SKEG.WHATEVER"])  # type: ignore[arg-type]
        # Connection survives a typed error from the server.
        c.ping()


def test_unknown_top_level_command_returns_error(resp_server: dict) -> None:
    with _client(resp_server) as c:
        with pytest.raises(ServerError):
            c._cmd([b"NONSENSE"])  # type: ignore[arg-type]
        c.ping()


# ── encoding paths ──────────────────────────────────────────────────


def test_resp_encode_array_round_trip() -> None:
    # Pure-Python encode/decode is exercised by every test above; here
    # we lock in the wire bytes for one canonical command so a future
    # encoder rewrite cannot silently change them.
    raw = RespClient._encode([b"SET", b"k", b"v"])
    assert raw == b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n"


def test_resp_encode_handles_empty_arg() -> None:
    raw = RespClient._encode([b"GET", b""])
    assert raw == b"*2\r\n$3\r\nGET\r\n$0\r\n\r\n"


def test_resp_select_zero_keeps_connection_alive(resp_server: dict) -> None:
    with _client(resp_server) as c:
        # Drivers that auto-issue SELECT 0 on connect should not lose
        # the session.
        reply = c._cmd([b"SELECT", b"0"])  # type: ignore[arg-type]
        assert reply in (b"OK", "OK")
        c.set(b"after-select", b"ok")
        assert c.get(b"after-select") == b"ok"


def test_resp_select_nonzero_returns_error(resp_server: dict) -> None:
    with _client(resp_server) as c:
        with pytest.raises(ServerError):
            c._cmd([b"SELECT", b"1"])  # type: ignore[arg-type]
        # And connection still usable.
        c.ping()
