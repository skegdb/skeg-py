# skeg-py

Python client for [skeg](https://github.com/skegdb/skeg), an
SSD-primary KV+vector store designed for Personal AI Inference machines.

**Status: alpha, pre-release.** Wire protocols are stable; the Python
API surface may still change. Not yet published to PyPI.

## What's in the package

Two synchronous clients sharing one error hierarchy:

| Module | Wire | Server binary | Default port | Use case |
| --- | --- | --- | --- | --- |
| `skeg.BinaryClient` | skeg-proto (native) | `skeg` | 7379 | Max throughput, full feature surface (KV + vector) |
| `skeg.RespClient` | RESP2/3 (Redis) | `skeg-resp3` | 6379 | Drop-in compat with redis-cli / redis-py / Redis tooling |

Three backends behind one selector (`skeg.client(...)`):

- **Pure-Python** (default, zero deps) - just `pip install skeg`
- **PyO3 / Rust** (optional, faster framing) - `pip install skeg[fast]`
  (requires Rust toolchain at install time, or a pre-built wheel)

The PyO3 path mirrors `BinaryClient`'s public API exactly, so you can
swap with no code changes:

```python
import skeg
c = skeg.client(prefer_native=True)   # uses PyO3 if available
c = skeg.client(prefer_native=False)  # forces pure-Python
```

## Install

### From source (this monorepo)

```sh
# Pure-Python only (no Rust toolchain needed)
SKEG_PY_PURE=1 pip install -e .

# With PyO3 backend (needs Rust toolchain installed)
pip install -e .
```

### From PyPI (when published)

```sh
pip install skeg              # pure-Python
pip install 'skeg[fast]'      # PyO3-backed (binary wheels)
```

## Quick start

### KV

```python
from skeg import BinaryClient

with BinaryClient.connect("127.0.0.1", 7379) as c:
    c.ping()
    c.set(b"hello", b"world")
    print(c.get(b"hello"))               # b"world"
    print(c.mget([b"hello", b"nope"]))   # [b"world", None]
    c.delete(b"hello")
```

### Vectors (in-RAM flat)

```python
from skeg import BinaryClient

with BinaryClient.connect("127.0.0.1", 7379) as c:
    c.vindex_create("notes", dim=1024, kind="int8", backend="flat")
    c.vset("notes", 1, my_embedding_1024d)
    c.vset("notes", 2, another_embedding)
    for hit in c.vsearch("notes", query_embedding, k=10):
        print(hit.id, hit.distance)
```

### Vectors (on-disk Vamana)

Build the index offline (one-shot), then point the client at the served
copy:

```sh
skeg-tool build --input embeddings.npy --output ./data --name notes
skeg-server --mode serve --data-dir ./data --tier pq:128:256
```

```python
# Same client code as above; the server handles the disk-backed index.
hits = c.vsearch("notes", query, k=10)
```

### Redis-compat (RESP3)

```python
from skeg import RespClient

with RespClient.connect("127.0.0.1", 6379) as c:
    c.hello(3)                      # upgrade to RESP3
    c.set(b"foo", b"bar")
    print(c.get(b"foo"))            # b"bar"
    c.mset({b"a": b"1", b"b": b"2"})
    print(c.mget([b"a", b"b"]))     # [b"1", b"2"]
    print(c.incr(b"counter"))       # 1
    print(c.skeg_stats())           # "cache_bytes=... n_keys=..."
```

Multi-tenant via HELLO AUTH:

```python
c.hello(3, auth=("alice", "hunter2"))
print(c.skeg_whoami())  # "tenant=<hex> mode=tenant-aware"
# All subsequent GET/SET are auto-scoped to alice's namespace.
```

## Testing

```sh
# Build the server binaries first.
cd ../..
cargo build --release -p skeg-server

cd adapters/python
pip install -e '.[test]'
pytest
```

The test fixture spawns one server per session and tears it down at
the end. Set `SKEG_BIN` or `SKEG_RESP3_BIN` to point at custom binary
paths.

### Test-suite safety vs your data

**The pytest suite is designed for a server it owns.** It writes keys
under names like `doc:N`, `counter:N`, `sk-N`, and creates/drops
VINDEX entries prefixed with `pytst-`. If you ever override the
default fixture to point the suite at a server that already holds
real data:

- KV: keys with the names above will be overwritten or deleted.
- VINDEX: only entries that match the `pytst-` prefix are touched
  by the cleanup loop. Unrelated VINDEX are left alone.

In short: never point `pytest` at a server you can't afford to lose
KV state on. The pytest-spawned fixture is the safe default.

## Compatibility

- Python 3.10+ (uses `from __future__ import annotations` and structural
  types only; no 3.11+ features).
- macOS, Linux. Windows untested.
- Server protocol version 1 (the only version that exists). Wire format
  is stable.

## Status and roadmap

This is alpha software. Not yet published to PyPI. Wire-level changes
are unlikely (the binary header and RESP3 subset are frozen for v0.1).
Python API may evolve before 1.0.

- [x] Pure-Python binary client (KV + vector)
- [x] Pure-Python RESP3 client (KV + SKEG.*)
- [x] Tests against live server (binary + RESP)
- [ ] PyO3 backend (in progress, see `rust/`)
- [ ] Async client (asyncio) - design pending
- [ ] Pipelining helper - currently one request at a time per connection
- [ ] Streaming insert helper for high-throughput VSET batches

## License

Apache-2.0. See [LICENSE](LICENSE).
