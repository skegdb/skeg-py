"""Wire-format encoder/decoder unit tests.

These run without a server: they pin the bytes-on-the-wire so a
serialiser change can't slip through silently. Every byte here is
load-bearing and matches the format spec in `_wire.py`.
"""
from __future__ import annotations

import struct

import pytest

from skeg import _wire as wire


# ── header ───────────────────────────────────────────────────────────


def test_encode_header_layout() -> None:
    raw = wire.encode_header(op=wire.OP_PING, req_id=42, payload_len=0)
    assert len(raw) == wire.HEADER_LEN
    # Layout: [magic u16][ver u8][op u8][flags u32][req_id u64][plen u32][reserved u32]
    magic = int.from_bytes(raw[0:2], "little")
    ver = raw[2]
    op = raw[3]
    flags = int.from_bytes(raw[4:8], "little")
    req_id = int.from_bytes(raw[8:16], "little")
    plen = int.from_bytes(raw[16:20], "little")
    assert magic == 0x564B
    assert ver == 1
    assert op == wire.OP_PING
    assert flags == 0
    assert req_id == 42
    assert plen == 0


def test_decode_header_roundtrip() -> None:
    raw = wire.encode_header(op=wire.OP_GET, req_id=1234, payload_len=99, flags=wire.FLAG_NO_REPLY)
    op, flags, req_id, plen = wire.decode_header(raw)
    assert op == wire.OP_GET
    assert flags == wire.FLAG_NO_REPLY
    assert req_id == 1234
    assert plen == 99


def test_decode_header_rejects_bad_magic() -> None:
    raw = bytearray(wire.encode_header(op=wire.OP_PING, req_id=1, payload_len=0))
    raw[0] = 0x00
    raw[1] = 0x00
    with pytest.raises(ValueError, match="bad magic"):
        wire.decode_header(bytes(raw))


def test_decode_header_rejects_unsupported_version() -> None:
    raw = bytearray(wire.encode_header(op=wire.OP_PING, req_id=1, payload_len=0))
    raw[2] = 99
    with pytest.raises(ValueError, match="unsupported version"):
        wire.decode_header(bytes(raw))


# ── payload encoders ────────────────────────────────────────────────


def test_encode_get_payload_layout() -> None:
    p = wire.encode_get_payload(b"hello")
    assert p == struct.pack("<H", 5) + b"hello"


def test_encode_set_payload_layout() -> None:
    p = wire.encode_set_payload(b"k", b"v_value")
    assert p == struct.pack("<HI", 1, 7) + b"k" + b"v_value"


def test_encode_set_payload_empty_value() -> None:
    p = wire.encode_set_payload(b"k", b"")
    assert p == struct.pack("<HI", 1, 0) + b"k"


def test_encode_set_payload_rejects_huge_key() -> None:
    with pytest.raises(ValueError, match="key too large"):
        wire.encode_set_payload(b"k" * 70_000, b"v")


def test_encode_mget_payload_layout() -> None:
    p = wire.encode_mget_payload([b"a", b"bb", b"ccc"])
    # Expected: n=3, then per-key: u16 len + bytes.
    expected = (
        struct.pack("<I", 3)
        + struct.pack("<H", 1) + b"a"
        + struct.pack("<H", 2) + b"bb"
        + struct.pack("<H", 3) + b"ccc"
    )
    assert p == expected


# ── response decoders ───────────────────────────────────────────────


def test_decode_value_response_basic() -> None:
    body = struct.pack("<I", 5) + b"hello"
    assert wire.decode_value_response(body) == b"hello"


def test_decode_value_response_empty_value() -> None:
    body = struct.pack("<I", 0)
    assert wire.decode_value_response(body) == b""


def test_decode_value_response_too_short() -> None:
    with pytest.raises(ValueError):
        wire.decode_value_response(b"\x00\x00")


def test_decode_value_response_truncated_body() -> None:
    body = struct.pack("<I", 10) + b"only5"
    with pytest.raises(ValueError, match="truncated"):
        wire.decode_value_response(body)


def test_decode_bool_response_true_and_false() -> None:
    assert wire.decode_bool_response(b"\x01") is True
    assert wire.decode_bool_response(b"\x00") is False
    assert wire.decode_bool_response(b"") is False  # tolerate empty


def test_decode_mget_response_mixed() -> None:
    body = (
        struct.pack("<I", 3)
        + b"\x00" + struct.pack("<I", 1) + b"a"
        + b"\x01"
        + b"\x00" + struct.pack("<I", 2) + b"bc"
    )
    out = wire.decode_mget_response(body)
    assert out == [b"a", None, b"bc"]


def test_decode_err_response_layout() -> None:
    body = bytes([wire.ERR_NOT_FOUND]) + bytes([9]) + b"not found"
    code, msg = wire.decode_err_response(body)
    assert code == wire.ERR_NOT_FOUND
    assert msg == "not found"


def test_decode_err_response_empty_body() -> None:
    code, msg = wire.decode_err_response(b"")
    assert code == 0
    assert msg == ""


# ── vector payloads ─────────────────────────────────────────────────


def test_encode_vindex_create_payload_layout() -> None:
    p = wire.encode_vindex_create_payload("notes", 4, wire.KIND_INT8, wire.BACKEND_FLAT)
    expected = (
        struct.pack("<H", 5) + b"notes"
        + struct.pack("<I", 4)
        + bytes([wire.KIND_INT8, wire.BACKEND_FLAT])
    )
    assert p == expected


def test_encode_vset_payload_layout() -> None:
    p = wire.encode_vset_payload("x", 7, [0.5, -0.25])
    expected = (
        struct.pack("<H", 1) + b"x"
        + struct.pack("<Q", 7)
        + struct.pack("<I", 2)
        + struct.pack("<2f", 0.5, -0.25)
    )
    assert p == expected


def test_encode_vsearch_payload_includes_l_search_trailing() -> None:
    p = wire.encode_vsearch_payload("ix", [0.1, 0.2], k=5, l_search=200)
    # name(2+2) + k(4) + dim(4) + 2 floats(8) + l_search(4) = 24 bytes
    assert len(p) == 24
    # l_search is the last 4 bytes.
    assert int.from_bytes(p[-4:], "little") == 200


def test_decode_vsearch_response_layout() -> None:
    body = (
        struct.pack("<I", 2)
        + struct.pack("<Qf", 42, 0.9)
        + struct.pack("<Qf", 7, 0.5)
    )
    out = wire.decode_vsearch_response(body)
    # f32 round-trip loses a few ulps; compare with tolerance.
    assert len(out) == 2
    assert out[0][0] == 42
    assert out[1][0] == 7
    assert abs(out[0][1] - 0.9) < 1e-6
    assert abs(out[1][1] - 0.5) < 1e-6


def test_decode_vsearch_response_truncated() -> None:
    body = struct.pack("<I", 2) + struct.pack("<Qf", 42, 0.9)  # missing second hit
    with pytest.raises(ValueError, match="truncated"):
        wire.decode_vsearch_response(body)


def test_flag_bit_positions_match_server() -> None:
    # If these constants ever drift, every binary client breaks. Lock them.
    assert wire.FLAG_WAIT_DURABLE == 1 << 0
    assert wire.FLAG_NO_REPLY == 1 << 1
    assert wire.FLAG_BATCH == 1 << 2
    assert wire.FLAG_BATCH_END == 1 << 3
