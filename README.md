# skeg-py

Python client for [skeg](https://github.com/skegdb/skeg), an
SSD-primary KV+vector store designed for Personal AI Inference machines.

```sh
pip install skeg              # pure-Python
pip install 'skeg[fast]'      # PyO3-backed (binary wheels for macOS arm64, Linux x86_64, Linux aarch64)
```

Wire formats (skeg binary protocol v1 and the RESP3 subset) are frozen.
The PyO3 backend mirrors the pure-Python `BinaryClient` API exactly.

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

```sh
pip install skeg              # pure-Python, zero dependencies
pip install 'skeg[fast]'      # PyO3-backed; binary wheels for macOS arm64, Linux x86_64, Linux aarch64
```

To build from source (e.g. on Windows or another arch), `pip install`
will compile the PyO3 backend; set `SKEG_PY_PURE=1` to skip it.

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
skeg-cli build --input embeddings.npy --output ./data --name notes
skeg --mode serve --data-dir ./data --tier pq:128:256
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
brew tap skegdb/tap
brew install skeg
git clone https://github.com/skegdb/skeg-py
cd skeg-py
pip install -e '.[test]'
SKEG_BIN=$(which skeg) SKEG_RESP3_BIN=$(which skeg-resp3) pytest
```

The test fixture spawns one server per session and tears it down at
the end. `SKEG_BIN` / `SKEG_RESP3_BIN` may be omitted if `skeg` and
`skeg-resp3` are on `$PATH`.

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

- Python 3.10+ (one abi3 wheel covers 3.10, 3.11, 3.12, 3.13).
- macOS arm64, Linux x86_64, Linux aarch64 (wheels). Other targets
  build from sdist; set `SKEG_PY_PURE=1` to skip the PyO3 backend.
- Server protocol version 1. Wire format is stable.

## License

Apache-2.0. See [LICENSE](LICENSE).
