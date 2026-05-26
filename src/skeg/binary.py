"""Binary-protocol client (port 7379 by default).

A thread-safe `BinaryClient` that speaks skeg-proto over TCP. Every
request waits for its response on the same socket (no pipelining for
now - pipelining is straightforward to layer on top of this, see
`pipeline()` in a future iteration).

Designed for compatibility with the PyO3-backed `_native.BinaryClient`:
both expose the same public methods so callers can swap backends
transparently via `skeg.client(prefer_native=...)`.
"""
from __future__ import annotations

import enum
import socket
import struct
import threading
from dataclasses import dataclass
from types import TracebackType
from typing import Iterable

from . import _wire as wire
from .errors import NotConnected, ProtocolError, ServerError


class VectorKind(enum.IntEnum):
    F32 = wire.KIND_F32
    INT8 = wire.KIND_INT8
    BINARY = wire.KIND_BINARY
    PQ = wire.KIND_PQ

    @classmethod
    def from_string(cls, s: str) -> "VectorKind":
        m = {
            "f32": cls.F32, "float32": cls.F32,
            "int8": cls.INT8,
            "binary": cls.BINARY, "bin": cls.BINARY,
            "pq": cls.PQ,
        }
        try:
            return m[s.lower()]
        except KeyError as e:
            raise ValueError(f"unknown vector kind {s!r}") from e


class VectorBackend(enum.IntEnum):
    FLAT = wire.BACKEND_FLAT
    DISK_VAMANA = wire.BACKEND_DISK_VAMANA

    @classmethod
    def from_string(cls, s: str) -> "VectorBackend":
        m = {
            "flat": cls.FLAT,
            "disk": cls.DISK_VAMANA, "disk_vamana": cls.DISK_VAMANA,
            "vamana": cls.DISK_VAMANA,
        }
        try:
            return m[s.lower()]
        except KeyError as e:
            raise ValueError(f"unknown backend {s!r}") from e


@dataclass(frozen=True)
class Hit:
    """One VSEARCH result.

    `score` semantics depend on the index's metric. For the default
    cosine index, higher = closer (1.0 = identical, 0.0 = orthogonal,
    -1.0 = anti-parallel). For hamming/binary tiers, score is the
    integer hamming distance cast to float (lower = closer).
    """
    id: int
    score: float


class BinaryClient:
    """Synchronous TCP client for skeg's binary protocol.

    Thread-safe at the request granularity: one client object may be
    used from multiple threads, with each call serialised on a lock.
    For pipelining or async use, instantiate one client per thread.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._req_id = 0
        self._lock = threading.Lock()
        self._closed = False

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def connect(cls, host: str = "127.0.0.1", port: int = 7379,
                *, timeout: float = 5.0) -> "BinaryClient":
        """Open a TCP connection to a skeg-server (binary protocol).

        `timeout` applies to the connect call and to every request
        afterwards (set to None to disable per-request timeout).
        """
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(sock)

    def __enter__(self) -> "BinaryClient":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying socket. Idempotent; safe to call from
        any thread, even concurrently with an in-flight request: the
        in-flight call will surface the socket shutdown as
        `NotConnected` rather than crash with a half-closed read."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._sock.close()
            except OSError:
                pass

    # ── KV ───────────────────────────────────────────────────────────

    def ping(self) -> None:
        """Send a PING. Raises ServerError on protocol failure."""
        self._call(wire.OP_PING, b"")

    def get(self, key: bytes) -> bytes | None:
        """Return the value for `key`, or None if it does not exist.

        The server signals "missing" via `Err{code=NotFound}`. We translate
        that to None here so callers see a clean Optional shape.
        """
        try:
            body = self._call(wire.OP_GET, wire.encode_get_payload(key))
        except ServerError as e:
            if e.code == wire.ERR_NOT_FOUND:
                return None
            raise
        return wire.decode_value_response(body)

    def set(self, key: bytes, value: bytes, *, no_reply: bool = False) -> None:
        """Store `value` under `key`. With `no_reply=True` the client
        does not wait for an ack; useful for streaming inserts."""
        flags = wire.FLAG_NO_REPLY if no_reply else 0
        if no_reply:
            # Fire-and-forget: no response read, no lock contention with
            # other in-flight reads.
            self._send_only(wire.OP_SET, wire.encode_set_payload(key, value),
                            flags=flags)
            return
        self._call(wire.OP_SET, wire.encode_set_payload(key, value))

    def delete(self, key: bytes) -> bool:
        """Delete `key`. Returns True if it existed, False otherwise.

        Server signals "key never existed" via Err{NotFound}, which we
        also translate to False so the result type stays a clean bool.
        """
        try:
            body = self._call(wire.OP_DEL, wire.encode_del_payload(key))
        except ServerError as e:
            if e.code == wire.ERR_NOT_FOUND:
                return False
            raise
        return wire.decode_bool_response(body)

    def mget(self, keys: Iterable[bytes]) -> list[bytes | None]:
        """Batch GET. Missing keys come back as None, order preserved.
        An empty input returns an empty list without a round-trip."""
        ks = list(keys)
        if not ks:
            return []
        body = self._call(wire.OP_MGET, wire.encode_mget_payload(ks))
        return wire.decode_mget_response(body)

    # ── Vector ───────────────────────────────────────────────────────

    def vindex_create(self, name: str, dim: int,
                       kind: VectorKind | str = VectorKind.INT8,
                       backend: VectorBackend | str = VectorBackend.FLAT,
                       ) -> None:
        """Create a vector index named `name` with `dim` dimensions.

        `kind` chooses the tier-1 quantisation; `backend` picks between
        the in-RAM flat layout and the on-disk Vamana graph.
        """
        if isinstance(kind, str):
            kind = VectorKind.from_string(kind)
        if isinstance(backend, str):
            backend = VectorBackend.from_string(backend)
        payload = wire.encode_vindex_create_payload(
            name, dim, int(kind), int(backend)
        )
        self._call(wire.OP_VINDEX_CREATE, payload)

    def vindex_drop(self, name: str) -> None:
        self._call(wire.OP_VINDEX_DROP, wire.encode_vindex_drop_payload(name))

    def vindex_list(self) -> list[dict]:
        """List every VINDEX known to the server.

        Returns a list of dicts: `{name, dim, kind, backend, n_vectors}`
        where `kind` and `backend` are the wire bytes (0=f32 / 1=int8 /
        2=binary; 0=flat / 1=disk-vamana). Stable alphabetical order by
        name so callers can diff between polls.
        """
        body = self._call(wire.OP_VINDEX_LIST, b"")
        return wire.decode_vindex_list_response(body)

    def shards(self) -> list[dict]:
        """Per-shard stats breakdown. Use for live dashboards / TUIs.

        Returns one dict per shard with `shard_id`, `cache_bytes`,
        `cache_evictions`, `n_keys`, `cache_budget`.
        """
        body = self._call(wire.OP_SHARDS, b"")
        return wire.decode_shards_response(body)

    def vset(self, name: str, vec_id: int,
              vector: Iterable[float]) -> None:
        """Insert or replace a vector by integer id.

        `vector` accepts any iterable of floats: `list`, `tuple`,
        `numpy.ndarray`, generator, etc. The adapter materialises it
        into a `list[float]` at the call boundary."""
        self._call(
            wire.OP_VSET,
            wire.encode_vset_payload(name, vec_id, _to_float_list(vector)),
        )

    def vget(self, name: str, vec_id: int) -> list[float] | None:
        """Return the vector for `vec_id`, or None if it does not exist."""
        try:
            body = self._call(wire.OP_VGET,
                              wire.encode_vget_payload(name, vec_id))
        except ServerError as e:
            if e.code == wire.ERR_NOT_FOUND:
                return None
            raise
        return wire.decode_vget_response(body)

    def vdel(self, name: str, vec_id: int) -> bool:
        try:
            body = self._call(wire.OP_VDEL,
                              wire.encode_vdel_payload(name, vec_id))
        except ServerError as e:
            if e.code == wire.ERR_NOT_FOUND:
                return False
            raise
        return wire.decode_bool_response(body)

    def vsearch(self, name: str, query: Iterable[float],
                 k: int = 10, l_search: int = 0) -> list[Hit]:
        """Approximate top-k nearest-neighbour search.

        `query` accepts any iterable of floats (list, tuple, numpy
        array, generator). `l_search` overrides the index's default
        search-list size when non-zero; larger values trade latency
        for recall.
        """
        body = self._call(
            wire.OP_VSEARCH,
            wire.encode_vsearch_payload(name, _to_float_list(query), k, l_search),
        )
        return [Hit(i, s) for i, s in wire.decode_vsearch_response(body)]

    # ── transport ────────────────────────────────────────────────────

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _send_only(self, op: int, payload: bytes, flags: int = 0) -> None:
        if self._closed:
            raise NotConnected("client is closed")
        with self._lock:
            req_id = self._next_req_id()
            header = wire.encode_header(op, req_id, len(payload), flags=flags)
            self._sock.sendall(header + payload)

    def _call(self, op: int, payload: bytes, flags: int = 0) -> bytes:
        if self._closed:
            raise NotConnected("client is closed")
        with self._lock:
            req_id = self._next_req_id()
            header = wire.encode_header(op, req_id, len(payload), flags=flags)
            self._sock.sendall(header + payload)

            resp_head = self._recv_exact(wire.HEADER_LEN)
            try:
                resp_op, _flags, resp_req, resp_len = wire.decode_header(resp_head)
            except ValueError as e:
                raise ProtocolError(str(e)) from e
            if resp_req != req_id:
                raise ProtocolError(
                    f"req_id mismatch: sent {req_id}, got {resp_req}"
                )
            body = self._recv_exact(resp_len) if resp_len else b""
            if resp_op == wire.OP_OK:
                return body
            if resp_op == wire.OP_ERR:
                code, message = wire.decode_err_response(body)
                raise ServerError(message, code=code)
            raise ProtocolError(f"unexpected response op 0x{resp_op:02X}")

    def _recv_exact(self, n: int) -> bytes:
        if n == 0:
            return b""
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            try:
                chunk = self._sock.recv_into(view[got:], n - got)
            except (ConnectionError, OSError) as e:
                # Connection-level failure: the socket is gone (peer
                # RST, FIN, kernel-level error, or a concurrent
                # close()). Surface as NotConnected so callers can
                # distinguish a transport drop from a protocol bug.
                raise NotConnected(f"socket error: {e}") from e
            if not chunk:
                raise NotConnected("peer closed mid-frame")
            got += chunk
        return bytes(buf)


def _to_float_list(values: Iterable[float]) -> list[float]:
    """Materialise any float iterable into a `list[float]`.

    Accepts `list`, `tuple`, `numpy.ndarray` (any dtype), generators,
    and other iterables. NumPy arrays are converted via `.tolist()`
    when present (cheaper than a Python-level loop) and otherwise
    fall back to the generic `list(...)` path.
    """
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        return list(tolist())
    return list(values)
