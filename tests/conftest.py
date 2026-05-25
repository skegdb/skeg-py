"""Pytest fixtures.

Tests need a live skeg-server. We spawn one per test session (binary
protocol) and one for the RESP3 tests (`skeg-resp3` binary).

`SKEG_BIN` and `SKEG_RESP3_BIN` environment variables override the
default paths (../../../target/release/skeg, ../../../target/release/skeg-resp3).
If neither path nor env var is set, the tests are skipped with an
informative reason.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent  # adapters/python/tests -> skeg/
TARGET = REPO_ROOT / "target" / "release"
DEFAULT_SKEG = TARGET / "skeg"
DEFAULT_SKEG_RESP3 = TARGET / "skeg-resp3"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_tcp(port: int, timeout: float = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _binary_path(env_var: str, default: Path) -> Path | None:
    explicit = os.environ.get(env_var)
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    return default if default.exists() else None


@pytest.fixture(scope="session")
def skeg_binary() -> Path:
    """Absolute path to the skeg binary protocol server."""
    p = _binary_path("SKEG_BIN", DEFAULT_SKEG)
    if p is None:
        pytest.skip(
            f"skeg binary not found at {DEFAULT_SKEG}; set SKEG_BIN or run "
            f"`cargo build --release -p skeg-server` from the skeg repo"
        )
    return p


@pytest.fixture(scope="session")
def skeg_resp3_binary() -> Path:
    p = _binary_path("SKEG_RESP3_BIN", DEFAULT_SKEG_RESP3)
    if p is None:
        pytest.skip(
            f"skeg-resp3 binary not found at {DEFAULT_SKEG_RESP3}; set "
            f"SKEG_RESP3_BIN or run `cargo build --release`"
        )
    return p


@pytest.fixture(scope="session")
def binary_server(skeg_binary: Path) -> dict[str, object]:
    """Spawn a skeg-server on a free port. Cleanup on teardown."""
    data_dir = Path(tempfile.mkdtemp(prefix="skeg-py-bin-"))
    port = _free_port()
    proc = subprocess.Popen(
        [str(skeg_binary), "--data-dir", str(data_dir),
         "--addr", f"127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not _wait_tcp(port):
        proc.terminate()
        pytest.fail("skeg-server (binary) did not start")
    yield {"port": port, "host": "127.0.0.1", "data_dir": data_dir, "proc": proc}
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def resp_server(skeg_resp3_binary: Path) -> dict[str, object]:
    data_dir = Path(tempfile.mkdtemp(prefix="skeg-py-resp-"))
    port = _free_port()
    proc = subprocess.Popen(
        [str(skeg_resp3_binary), "--data-dir", str(data_dir),
         "--addr", f"127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not _wait_tcp(port):
        proc.terminate()
        pytest.fail("skeg-resp3 did not start")
    yield {"port": port, "host": "127.0.0.1", "data_dir": data_dir, "proc": proc}
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    import shutil
    shutil.rmtree(data_dir, ignore_errors=True)
