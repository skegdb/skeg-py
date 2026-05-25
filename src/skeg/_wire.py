"""Wire-format primitives for skeg-proto.

Header layout (24 bytes, little-endian):

    [magic u16 = 0x564B][version u8 = 1][op u8][flags u32]
    [req_id u64][payload_len u32][reserved u32]

KV payloads (request):

    GET:    [key_len u16][key]
    SET:    [key_len u16][value_len u32][key][value]
    DEL:    [key_len u16][key]
    MGET:   [n u32][[key_len u16][key]] * n

KV responses (Ok shape; missing keys come back as Err{NotFound}):

    GET:    [value_len u32][value]
    DEL:    [u8 bool]               # 1 if key existed
    MGET:   [n u32][per-result: [status u8 (0=found|1=missing)]
                                 [if found: [vlen u32][value]]] x n

Vector payloads (request):

    VINDEX_CREATE: [nlen u16][name][dim u32][kind u8][backend u8]
    VINDEX_DROP:   [nlen u16][name]
    VSET:          [nlen u16][name][vec_id u64][dim u32][f32 LE * dim]
    VGET:          [nlen u16][name][vec_id u64]
    VDEL:          [nlen u16][name][vec_id u64]
    VSEARCH:       [nlen u16][name][k u32][dim u32][f32 LE * dim]
                   [optional u32 l_search trailing field]

Vector responses (Ok shape):

    VGET:    [vlen u32][f32 LE * dim]   (vlen is in bytes; dim = vlen/4)
    VSEARCH: [n u32][[id u64][score f32 LE]] * n
    VDEL:    [u8 bool]

Err shape (any op): `[code u8][msg_len u8][msg UTF-8]`. NotFound = 0x01.
"""
from __future__ import annotations

import struct
from typing import Tuple

MAGIC: int = 0x564B
VERSION: int = 1
HEADER_LEN: int = 24

# Op codes (mirror skeg-proto/src/op.rs).
OP_GET = 0x01
OP_SET = 0x02
OP_DEL = 0x03
OP_MGET = 0x04
OP_EXISTS = 0x06

OP_VINDEX_CREATE = 0x10
OP_VINDEX_DROP = 0x11
OP_VSET = 0x12
OP_VGET = 0x13
OP_VDEL = 0x14
OP_VSEARCH = 0x15
OP_VINDEX_LIST = 0x16

OP_PING = 0x80
OP_STATS = 0x81
OP_SHARDS = 0x83

OP_OK = 0xC0
OP_ERR = 0xC1
OP_CONTINUED = 0xC2

# Flag bit positions (mirror skeg-proto/src/flags.rs).
FLAG_WAIT_DURABLE = 1 << 0
FLAG_NO_REPLY = 1 << 1
FLAG_BATCH = 1 << 2
FLAG_BATCH_END = 1 << 3
FLAG_COMPRESSED_LZ4 = 1 << 4
FLAG_CONTINUATION = 1 << 5
FLAG_SET_NX = 1 << 6
FLAG_SET_XX = 1 << 7

# Server error codes (skeg-proto::response::ErrCode).
ERR_NOT_FOUND = 0x01
ERR_INVALID_REQUEST = 0x02
ERR_INTERNAL = 0x03

# VectorKind discriminants (mirror skeg-vector::QuantKind on the wire).
KIND_F32 = 0
KIND_INT8 = 1
KIND_BINARY = 2
KIND_PQ = 3

# VectorBackend discriminants (mirror skeg-client::VectorBackend).
BACKEND_FLAT = 0
BACKEND_DISK_VAMANA = 1


def encode_header(op: int, req_id: int, payload_len: int, flags: int = 0) -> bytes:
    """Pack the 24-byte frame header."""
    return struct.pack(
        "<HBBIQII",
        MAGIC, VERSION, op, flags, req_id, payload_len, 0,
    )


def decode_header(buf: bytes) -> Tuple[int, int, int, int]:
    """Unpack a 24-byte header into (op, flags, req_id, payload_len)."""
    if len(buf) != HEADER_LEN:
        raise ValueError(f"header is {len(buf)} bytes, want {HEADER_LEN}")
    magic, version, op, flags, req_id, payload_len, _ = struct.unpack(
        "<HBBIQII", buf,
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic: 0x{magic:04X} != 0x{MAGIC:04X}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    return op, flags, req_id, payload_len


# ── name + f32-vec helpers (shared by vector ops) ────────────────────────────


def _put_name(name: str) -> bytes:
    """`[u16 len][utf8 bytes]`."""
    b = name.encode("utf-8")
    if len(b) > 0xFFFF:
        raise ValueError("vector name too long (>65535 bytes)")
    return struct.pack("<H", len(b)) + b


def _put_f32_vec(vector: list[float]) -> bytes:
    """`[u32 dim][f32 LE * dim]`."""
    if len(vector) > 0xFFFF_FFFF:
        raise ValueError("vector too large")
    return struct.pack("<I", len(vector)) + struct.pack(
        f"<{len(vector)}f", *vector,
    )


# ── KV encoders ──────────────────────────────────────────────────────────────


def encode_get_payload(key: bytes) -> bytes:
    if len(key) > 0xFFFF:
        raise ValueError("key too large (>65535 bytes)")
    return struct.pack("<H", len(key)) + key


def encode_set_payload(key: bytes, value: bytes) -> bytes:
    if len(key) > 0xFFFF:
        raise ValueError("key too large (>65535 bytes)")
    return struct.pack("<HI", len(key), len(value)) + key + value


def encode_del_payload(key: bytes) -> bytes:
    return encode_get_payload(key)  # same shape


def encode_mget_payload(keys: list[bytes]) -> bytes:
    if len(keys) > 0xFFFF_FFFF:
        raise ValueError("too many keys for one MGET")
    out = struct.pack("<I", len(keys))
    for k in keys:
        if len(k) > 0xFFFF:
            raise ValueError("key too large in MGET")
        out += struct.pack("<H", len(k)) + k
    return out


# ── KV response decoders ─────────────────────────────────────────────────────


def decode_value_response(payload: bytes) -> bytes:
    """Decode `[value_len u32][value]`. Used for GET / VGET Ok bodies."""
    if len(payload) < 4:
        raise ValueError("value response too short")
    (vlen,) = struct.unpack_from("<I", payload, 0)
    if len(payload) < 4 + vlen:
        raise ValueError("value response truncated")
    return payload[4:4 + vlen]


def decode_bool_response(payload: bytes) -> bool:
    """Decode the 1-byte bool used by DEL / VDEL."""
    if not payload:
        return False
    return payload[0] != 0


def decode_mget_response(payload: bytes) -> list[bytes | None]:
    """Decode `[n u32]` then n records of `[status u8] [if found: vlen u32, value]`.

    `status = 0` means found, `1` means missing (mirrors the server's
    encode_ok_mget convention).
    """
    if len(payload) < 4:
        raise ValueError("MGET response too short")
    (n,) = struct.unpack_from("<I", payload, 0)
    off = 4
    out: list[bytes | None] = []
    for _ in range(n):
        if off >= len(payload):
            raise ValueError("MGET response truncated")
        status = payload[off]
        off += 1
        if status == 0:
            if off + 4 > len(payload):
                raise ValueError("MGET response truncated mid-len")
            (vlen,) = struct.unpack_from("<I", payload, off)
            off += 4
            if off + vlen > len(payload):
                raise ValueError("MGET response truncated mid-value")
            out.append(payload[off:off + vlen])
            off += vlen
        else:
            out.append(None)
    return out


# ── Vector encoders ──────────────────────────────────────────────────────────


def encode_vindex_create_payload(name: str, dim: int, kind: int, backend: int
                                  ) -> bytes:
    """`[nlen u16][name][dim u32][kind u8][backend u8]`."""
    return (
        _put_name(name)
        + struct.pack("<I", dim)
        + struct.pack("<B", kind)
        + struct.pack("<B", backend)
    )


def encode_vindex_drop_payload(name: str) -> bytes:
    return _put_name(name)


def encode_vset_payload(name: str, vec_id: int, vector: list[float]) -> bytes:
    """`[nlen u16][name][vec_id u64][dim u32][f32 LE * dim]`."""
    return _put_name(name) + struct.pack("<Q", vec_id) + _put_f32_vec(vector)


def encode_vget_payload(name: str, vec_id: int) -> bytes:
    """`[nlen u16][name][vec_id u64]`."""
    return _put_name(name) + struct.pack("<Q", vec_id)


def encode_vdel_payload(name: str, vec_id: int) -> bytes:
    return encode_vget_payload(name, vec_id)


def encode_vsearch_payload(name: str, query: list[float], k: int,
                            l_search: int = 0) -> bytes:
    """`[nlen u16][name][k u32][dim u32][f32 LE * dim][optional u32 l_search]`.

    `l_search = 0` is "use the index default". The trailing field is
    optional - old servers ignore it - but we always include it for
    consistent frames.
    """
    out = _put_name(name) + struct.pack("<I", k) + _put_f32_vec(query)
    out += struct.pack("<I", l_search)
    return out


# ── Vector response decoders ─────────────────────────────────────────────────


def decode_vget_response(payload: bytes) -> list[float]:
    """Decode a VGET Ok body: `[vlen u32][f32 LE * (vlen / 4)]`."""
    raw = decode_value_response(payload)
    if len(raw) % 4 != 0:
        raise ValueError(f"VGET response body is {len(raw)} bytes (not /4)")
    dim = len(raw) // 4
    return list(struct.unpack_from(f"<{dim}f", raw, 0))


def decode_vindex_list_response(payload: bytes) -> list[dict]:
    """Decode `Op::VindexList` Ok body.

    Layout: `[u32 n][per row: [u16 nlen][name UTF-8][u32 dim][u8 kind]
    [u8 backend][u64 n_vectors]]`.

    Returns a list of dicts so the caller can choose a structured
    representation without needing a separate dataclass import path.
    """
    if len(payload) < 4:
        raise ValueError("vindex_list response too short")
    (n,) = struct.unpack_from("<I", payload, 0)
    off = 4
    out: list[dict] = []
    for _ in range(n):
        if off + 2 > len(payload):
            raise ValueError("vindex_list truncated mid-name-len")
        (nlen,) = struct.unpack_from("<H", payload, off)
        off += 2
        if off + nlen + 4 + 1 + 1 + 8 > len(payload):
            raise ValueError("vindex_list truncated mid-row")
        name = payload[off:off + nlen].decode("utf-8")
        off += nlen
        (dim,) = struct.unpack_from("<I", payload, off)
        off += 4
        kind = payload[off]
        off += 1
        backend = payload[off]
        off += 1
        (n_vectors,) = struct.unpack_from("<Q", payload, off)
        off += 8
        out.append({
            "name": name,
            "dim": dim,
            "kind": kind,
            "backend": backend,
            "n_vectors": n_vectors,
        })
    return out


def decode_shards_response(payload: bytes) -> list[dict]:
    """Decode `Op::Shards` Ok body. Layout: `[u32 n][per shard: [u32 id]
    [u64 cache_bytes][u64 evictions][u64 n_keys][u64 budget]]`."""
    if len(payload) < 4:
        raise ValueError("shards response too short")
    (n,) = struct.unpack_from("<I", payload, 0)
    out: list[dict] = []
    pos = 4
    row = struct.Struct("<IQQQQ")
    for _ in range(n):
        if pos + row.size > len(payload):
            raise ValueError("shards response truncated")
        sid, cb, evict, nk, bud = row.unpack_from(payload, pos)
        out.append({
            "shard_id": sid,
            "cache_bytes": cb,
            "cache_evictions": evict,
            "n_keys": nk,
            "cache_budget": bud,
        })
        pos += row.size
    return out


def decode_vsearch_response(payload: bytes) -> list[tuple[int, float]]:
    """Decode a VSEARCH Ok body: `[n u32][(id u64, score f32 LE) * n]`."""
    if len(payload) < 4:
        raise ValueError("VSEARCH response too short")
    (n,) = struct.unpack_from("<I", payload, 0)
    off = 4
    record = struct.Struct("<Qf")
    if len(payload) < 4 + n * record.size:
        raise ValueError("VSEARCH response truncated")
    out: list[tuple[int, float]] = []
    for _ in range(n):
        vec_id, score = record.unpack_from(payload, off)
        out.append((vec_id, score))
        off += record.size
    return out


# ── Err response ─────────────────────────────────────────────────────────────


def decode_err_response(payload: bytes) -> tuple[int, str]:
    """Decode `[code u8][msg_len u8][msg UTF-8]`.

    Returns `(code, message)`. Empty payload yields `(0, "")`.
    """
    if not payload:
        return 0, ""
    code = payload[0]
    if len(payload) < 2:
        return code, ""
    msg_len = payload[1]
    end = min(len(payload), 2 + msg_len)
    msg = payload[2:end].decode("utf-8", errors="replace")
    return code, msg
