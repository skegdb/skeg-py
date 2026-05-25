"""Error hierarchy shared across binary + RESP backends."""
from __future__ import annotations


class SkegError(Exception):
    """Base for every error raised by this package."""


class NotConnected(SkegError):
    """Operation attempted on a client that is closed or not connected."""


class ProtocolError(SkegError):
    """Server returned a frame that violates the protocol contract.

    Triggered for malformed headers, unexpected response op codes, and
    truncated payloads. The connection is unusable after this and must
    be closed by the caller.
    """


class ServerError(SkegError):
    """Server returned a typed error response.

    `code` is the protocol-level error byte (see skeg-proto Op::Err
    payload); `message` is the human-readable body the server sent.
    """

    def __init__(self, message: str, code: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
