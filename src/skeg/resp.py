"""RESP2/3 client (port 6379 by default, Redis-compat wire).

Speaks the subset of RESP that skeg-resp3 implements: HELLO, PING, ECHO,
GET, SET, DEL, EXISTS, MGET, MSET, INCR/DECR, plus the SKEG.* namespace
(STATS, WHOAMI, AUTH placeholder).

Designed as a drop-in for any redis-py user who wants to point at skeg
without changing their client code. We intentionally do NOT depend on
redis-py: the wire protocol is small enough to keep inline, the install
footprint stays at zero deps.

Usage:

    from skeg import RespClient
    with RespClient.connect("127.0.0.1", 6379) as c:
        c.hello(3)                          # upgrade to RESP3
        c.set(b"hello", b"world")
        print(c.get(b"hello"))              # b"world"

Multi-tenant:

    c.hello(3, auth=("alice", "hunter2"))   # HELLO 3 AUTH ...
    c.skeg_whoami()                          # report bound tenant
"""
from __future__ import annotations

import socket
import threading
from types import TracebackType
from typing import Iterable

from .errors import NotConnected, ProtocolError, ServerError


class RespClient:
    """Synchronous RESP2/3 client.

    Connections start in RESP2 mode and are upgraded with `hello(3)`.
    Wire commands accept and return `bytes` (caller decides text vs binary).
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""
        self._lock = threading.Lock()
        self._closed = False
        self.version = 2

    @classmethod
    def connect(cls, host: str = "127.0.0.1", port: int = 6379,
                *, timeout: float = 5.0) -> "RespClient":
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(sock)

    def __enter__(self) -> "RespClient":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._sock.close()
        except OSError:
            pass

    # ── high-level commands ─────────────────────────────────────────

    def hello(self, version: int = 3, *,
              auth: tuple[str, str] | None = None,
              client_name: str | None = None) -> object:
        """`HELLO [version [AUTH user pass] [SETNAME name]]`.

        Returns the server's map/array of fields as a raw object
        (bytes/int leaves, list/dict containers).
        """
        parts = [b"HELLO", str(version).encode()]
        if auth is not None:
            user, password = auth
            parts += [b"AUTH", user.encode(), password.encode()]
        if client_name is not None:
            parts += [b"SETNAME", client_name.encode()]
        reply = self._cmd(parts)
        # Track the negotiated version so future calls know how to
        # interpret replies that differ between RESP2/3 (e.g. Map vs
        # flat-Array). The client behaves identically in this minimal
        # implementation; the field is exposed for diagnostics.
        self.version = version
        return reply

    def ping(self, message: bytes | None = None) -> bytes:
        """PING [message]. Returns +PONG or the echoed message."""
        if message is None:
            return _as_bytes(self._cmd([b"PING"]))
        return _as_bytes(self._cmd([b"PING", message]))

    def echo(self, message: bytes) -> bytes:
        return _as_bytes(self._cmd([b"ECHO", message]))

    def get(self, key: bytes) -> bytes | None:
        reply = self._cmd([b"GET", key])
        if reply is None:
            return None
        return _as_bytes(reply)

    def set(self, key: bytes, value: bytes) -> None:
        reply = self._cmd([b"SET", key, value])
        if _as_bytes(reply) != b"OK":
            raise ProtocolError(f"unexpected SET reply: {reply!r}")

    def delete(self, *keys: bytes) -> int:
        """DEL key [key ...]. Returns the number of keys actually deleted."""
        if not keys:
            raise ValueError("delete requires at least one key")
        parts: list[bytes] = [b"DEL"]
        parts.extend(keys)
        return int(self._cmd(parts))  # type: ignore[arg-type]

    def exists(self, *keys: bytes) -> int:
        if not keys:
            raise ValueError("exists requires at least one key")
        parts: list[bytes] = [b"EXISTS"]
        parts.extend(keys)
        return int(self._cmd(parts))  # type: ignore[arg-type]

    def mget(self, keys: Iterable[bytes]) -> list[bytes | None]:
        parts: list[bytes] = [b"MGET"]
        parts.extend(keys)
        reply = self._cmd(parts)
        if not isinstance(reply, list):
            raise ProtocolError(f"MGET reply not an array: {reply!r}")
        out: list[bytes | None] = []
        for item in reply:
            if item is None:
                out.append(None)
            else:
                out.append(_as_bytes(item))
        return out

    def mset(self, pairs: dict[bytes, bytes]) -> None:
        parts: list[bytes] = [b"MSET"]
        for k, v in pairs.items():
            parts.append(k)
            parts.append(v)
        reply = self._cmd(parts)
        if _as_bytes(reply) != b"OK":
            raise ProtocolError(f"unexpected MSET reply: {reply!r}")

    def incr(self, key: bytes) -> int:
        return int(self._cmd([b"INCR", key]))  # type: ignore[arg-type]

    def decr(self, key: bytes) -> int:
        return int(self._cmd([b"DECR", key]))  # type: ignore[arg-type]

    def incrby(self, key: bytes, delta: int) -> int:
        return int(self._cmd([b"INCRBY", key, str(delta).encode()]))  # type: ignore[arg-type]

    # ── SKEG.* namespace ────────────────────────────────────────────

    def skeg_stats(self) -> str:
        """Return the server's cache/stats summary as a UTF-8 string."""
        return _as_bytes(self._cmd([b"SKEG.STATS"])).decode("utf-8")

    def skeg_whoami(self) -> str:
        """Return `tenant=<hex> mode=<single-tenant|tenant-aware>`."""
        return _as_bytes(self._cmd([b"SKEG.WHOAMI"])).decode("utf-8")

    # ── transport ───────────────────────────────────────────────────

    def _cmd(self, parts: list[bytes]) -> object:
        if self._closed:
            raise NotConnected("client is closed")
        with self._lock:
            msg = self._encode(parts)
            try:
                self._sock.sendall(msg)
            except (OSError, ConnectionError) as e:
                raise ProtocolError(f"send failed: {e}") from e
            return self._read_reply()

    @staticmethod
    def _encode(parts: list[bytes]) -> bytes:
        # Inline array encoding to avoid building a bytes object per part.
        head = f"*{len(parts)}\r\n".encode()
        body = bytearray()
        for p in parts:
            body += f"${len(p)}\r\n".encode()
            body += p
            body += b"\r\n"
        return head + bytes(body)

    def _read_reply(self) -> object:
        line = self._readline()
        if not line:
            raise ProtocolError("empty reply line")
        kind, body = line[:1], line[1:]
        if kind == b"+":              # Simple string
            return body
        if kind == b":":              # Integer
            return int(body)
        if kind == b",":              # RESP3 Double
            return float(body)
        if kind == b"#":              # RESP3 Bool
            return body == b"t"
        if kind == b"-":              # Error
            raise ServerError(body.decode("utf-8", errors="replace"))
        if kind == b"$" or kind == b"=":   # Bulk / verbatim
            n = int(body)
            if n < 0:
                return None
            data = self._readexact(n)
            self._readexact(2)        # \r\n trailer
            return data
        if kind == b"_":              # RESP3 Null
            return None
        if kind == b"*":              # Array
            n = int(body)
            if n < 0:
                return None
            return [self._read_reply() for _ in range(n)]
        if kind == b"%":              # RESP3 Map - flatten to dict
            n = int(body)
            out: dict[bytes, object] = {}
            for _ in range(n):
                k = self._read_reply()
                v = self._read_reply()
                if isinstance(k, (bytes, bytearray)):
                    out[bytes(k)] = v
                else:
                    out[str(k).encode()] = v  # type: ignore[arg-type]
            return out
        if kind == b"~":              # RESP3 Set - return as list
            n = int(body)
            return [self._read_reply() for _ in range(n)]
        if kind == b">":              # RESP3 Push - read + return as list
            n = int(body)
            return [self._read_reply() for _ in range(n)]
        raise ProtocolError(f"unknown RESP prefix: {kind!r}")

    def _readline(self) -> bytes:
        while b"\r\n" not in self._buf:
            try:
                chunk = self._sock.recv(4096)
            except (OSError, ConnectionError) as e:
                raise ProtocolError(f"recv failed: {e}") from e
            if not chunk:
                raise ProtocolError("peer closed")
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\r\n")
        self._buf = rest
        return line

    def _readexact(self, n: int) -> bytes:
        while len(self._buf) < n:
            try:
                chunk = self._sock.recv(max(4096, n - len(self._buf)))
            except (OSError, ConnectionError) as e:
                raise ProtocolError(f"recv failed: {e}") from e
            if not chunk:
                raise ProtocolError("peer closed")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out


def _as_bytes(obj: object) -> bytes:
    """Coerce a RESP reply leaf to bytes. Integers/floats become decimal."""
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, (int, float)):
        return str(obj).encode()
    if obj is None:
        return b""
    raise TypeError(f"cannot coerce {type(obj).__name__} to bytes")
