"""skeg-py: Python client for skeg.

Three transports, one API:

    from skeg import BinaryClient    # native binary protocol (port 7379)
    from skeg import RespClient      # RESP2/3 wire (port 6379, Redis-compat)
    from skeg import client          # auto-pick: PyO3 backend if installed,
                                     # else pure-Python on binary protocol

Quick start:

    from skeg import BinaryClient
    with BinaryClient.connect("127.0.0.1", 7379) as c:
        c.set(b"hello", b"world")
        print(c.get(b"hello"))  # b"world"
        c.vindex_create("notes", dim=1024, kind="int8")
        c.vset("notes", 1, [0.1, 0.2, ...])
        for hit in c.vsearch("notes", [0.1, 0.2, ...], k=10):
            print(hit.id, hit.distance)

Status: alpha, pre-release. Wire formats are stable; the Python surface
may still change. Not yet on PyPI. See README.md for the install path
(pure-Python vs PyO3-backed).
"""
from __future__ import annotations

from .binary import BinaryClient, Hit, VectorKind, VectorBackend
from .resp import RespClient
from .errors import SkegError, ProtocolError, NotConnected, ServerError

# Backend selection: PyO3 if available (and not explicitly disabled),
# else pure-Python. Importing skeg.fast forces the native backend;
# importing skeg.pure forces the pure-Python backend.
try:
    from ._native import BinaryClient as _NativeBinary  # type: ignore
    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def client(addr: str = "127.0.0.1", port: int = 7379, *,
           prefer_native: bool = True) -> "BinaryClient":
    """Open a binary-protocol connection with the best available backend.

    `prefer_native=False` forces the pure-Python implementation even when
    the PyO3 backend is available (useful for benchmarks and to compare
    the two paths under the same workload).
    """
    if prefer_native and _HAS_NATIVE:
        return _NativeBinary(addr, port)  # type: ignore[no-any-return]
    return BinaryClient.connect(addr, port)


__all__ = [
    "BinaryClient",
    "RespClient",
    "Hit",
    "VectorKind",
    "VectorBackend",
    "SkegError",
    "ProtocolError",
    "NotConnected",
    "ServerError",
    "client",
]

__version__ = "0.1.0a0"
