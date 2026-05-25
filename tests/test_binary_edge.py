"""Binary-protocol edge cases.

Covers boundaries the smoke tests don't: empty payloads, oversized
inputs, unicode keys, repeated pings, error-class propagation, and
no_reply pipelining behaviour. None of these should regress between
releases without a deliberate decision.
"""
from __future__ import annotations

import threading

import pytest

from skeg import BinaryClient, NotConnected, ProtocolError, ServerError
from skeg import _wire as wire


def _client(server: dict) -> BinaryClient:
    return BinaryClient.connect(server["host"], server["port"])  # type: ignore[arg-type]


# ── value-boundary tests ─────────────────────────────────────────────


def test_empty_value_roundtrip(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"empty-val", b"")
        assert c.get(b"empty-val") == b""


def test_zero_byte_in_key_and_value(binary_server: dict) -> None:
    # Keys and values are arbitrary bytes; NULs must pass through cleanly.
    key = b"with\x00null"
    value = b"\x00\x01\x02\xff"
    with _client(binary_server) as c:
        c.set(key, value)
        assert c.get(key) == value


def test_unicode_key_via_utf8_encoding(binary_server: dict) -> None:
    key = "città".encode("utf-8")
    value = "café ☕".encode("utf-8")
    with _client(binary_server) as c:
        c.set(key, value)
        assert c.get(key) == value


def test_single_byte_key_and_value(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.set(b"k", b"v")
        assert c.get(b"k") == b"v"


def test_key_at_size_limit_minus_one(binary_server: dict) -> None:
    # u16 key_len: max 65535. We test 65534 to leave a margin for any
    # framing slack and not blow the test budget on a 64KB transfer
    # each run.
    key = b"k" * 65534
    with _client(binary_server) as c:
        c.set(key, b"x")
        assert c.get(key) == b"x"


def test_key_too_large_raises_clientside(binary_server: dict) -> None:
    # u16 key_len overflows past 65535. The encoder catches this BEFORE
    # we touch the network, so the client raises ValueError rather than
    # a ProtocolError.
    huge = b"k" * 70_000
    with _client(binary_server) as c:
        with pytest.raises(ValueError, match="key too large"):
            c.set(huge, b"x")
        # Connection still usable after a client-side validation error.
        c.ping()


def test_value_one_mib_roundtrip(binary_server: dict) -> None:
    # 1 MiB value: stresses the value_len u32 path and recv loop.
    big = b"V" * (1024 * 1024)
    with _client(binary_server) as c:
        c.set(b"1mib", big)
        out = c.get(b"1mib")
        assert out is not None and len(out) == 1024 * 1024
        # Spot-check; we don't want to assert byte-by-byte on 1MB.
        assert out[:8] == b"V" * 8
        assert out[-8:] == b"V" * 8


def test_mget_empty_list_returns_empty(binary_server: dict) -> None:
    with _client(binary_server) as c:
        assert c.mget([]) == []


def test_mget_all_missing_returns_all_none(binary_server: dict) -> None:
    with _client(binary_server) as c:
        out = c.mget([b"never-exist-1", b"never-exist-2", b"never-exist-3"])
        assert out == [None, None, None]


def test_mget_many_keys(binary_server: dict) -> None:
    # 256 keys exercises both the encoder loop and the response parser.
    with _client(binary_server) as c:
        for i in range(256):
            c.set(f"many-{i}".encode(), f"v{i}".encode())
        keys = [f"many-{i}".encode() for i in range(256)]
        out = c.mget(keys)
        assert len(out) == 256
        for i, v in enumerate(out):
            assert v == f"v{i}".encode()


# ── pipelined / fire-and-forget ──────────────────────────────────────


def test_no_reply_sequence_then_sync_get(binary_server: dict) -> None:
    # 32 no_reply SETs then a single sync GET. Tests that the client
    # internal req_id counter stays in sync with the server even when
    # no acks are read between requests.
    with _client(binary_server) as c:
        for i in range(32):
            c.set(f"async-{i}".encode(), f"v{i}".encode(), no_reply=True)
        c.ping()
        for i in range(32):
            assert c.get(f"async-{i}".encode()) == f"v{i}".encode()


def test_alternating_no_reply_and_get(binary_server: dict) -> None:
    # Interleave no_reply SET with sync GET: the req_id advances every
    # call but only the GETs read responses. If the client mis-counts
    # we see ProtocolError("req_id mismatch").
    with _client(binary_server) as c:
        for i in range(20):
            c.set(f"int-{i}".encode(), b"x", no_reply=True)
            if i % 4 == 0:
                # Sync read; no_reply set above must not have left a
                # pending frame on the wire.
                c.ping()
        c.ping()
        assert c.get(b"int-0") == b"x"


# ── error propagation ───────────────────────────────────────────────


def test_get_uncreated_vindex_returns_server_error(binary_server: dict) -> None:
    # A vector op on a non-existent index should surface a ServerError,
    # not crash the connection.
    with _client(binary_server) as c:
        with pytest.raises(ServerError):
            c.vsearch("definitely-no-such-index-xyz", [0.0, 0.0, 0.0, 0.0], k=1)
        # The connection survives a non-NotFound server error.
        c.ping()


def test_vindex_create_duplicate_surfaces_error(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("dup-test", dim=4, kind="f32", backend="flat")
        with pytest.raises(ServerError):
            c.vindex_create("dup-test", dim=4, kind="f32", backend="flat")
        c.vindex_drop("dup-test")


def test_vset_wrong_dim_surfaces_error(binary_server: dict) -> None:
    with _client(binary_server) as c:
        c.vindex_create("dim-test", dim=4, kind="f32", backend="flat")
        with pytest.raises(ServerError):
            c.vset("dim-test", 1, [0.1, 0.2])  # only 2 floats, want 4
        c.vindex_drop("dim-test")


def test_unknown_kind_string_raises_clientside(binary_server: dict) -> None:
    # The Python wrapper validates kind/backend strings before sending.
    with _client(binary_server) as c:
        with pytest.raises(ValueError, match="unknown vector kind"):
            c.vindex_create("never-created", dim=4, kind="nonsense",
                             backend="flat")


def test_unknown_backend_string_raises_clientside(binary_server: dict) -> None:
    with _client(binary_server) as c:
        with pytest.raises(ValueError, match="unknown backend"):
            c.vindex_create("never-created", dim=4, kind="f32",
                             backend="bogus")


# ── connection lifecycle ────────────────────────────────────────────


def test_double_close_is_idempotent(binary_server: dict) -> None:
    c = _client(binary_server)
    c.close()
    c.close()  # second close must not raise
    with pytest.raises(NotConnected):
        c.get(b"x")


def test_multiple_pings_share_one_connection(binary_server: dict) -> None:
    with _client(binary_server) as c:
        for _ in range(100):
            c.ping()


def test_req_id_advances_monotonically(binary_server: dict) -> None:
    # Verify the private counter advances as expected after 100 ops.
    # Done via attribute peek; if we ever change the implementation,
    # this test forces the rename to be deliberate.
    with _client(binary_server) as c:
        before = c._req_id
        for _ in range(50):
            c.ping()
        after = c._req_id
        assert after - before == 50


# ── concurrent client use ───────────────────────────────────────────


def test_one_client_used_by_two_threads(binary_server: dict) -> None:
    # The client serialises requests with a lock; we don't promise true
    # parallelism but we DO promise that two threads can't corrupt each
    # other's req_id matching.
    with _client(binary_server) as c:
        c.set(b"shared", b"v")

        results: dict[int, list[bytes | None]] = {0: [], 1: []}

        def worker(tid: int) -> None:
            for _ in range(50):
                results[tid].append(c.get(b"shared"))

        t1 = threading.Thread(target=worker, args=(0,))
        t2 = threading.Thread(target=worker, args=(1,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        for v in results[0] + results[1]:
            assert v == b"v"
