//! PyO3 backend for skeg-py.
//!
//! Wraps the `skeg-client` Rust crate so Python callers get framing and
//! socket I/O in native code. The Python wrapper in `skeg/__init__.py`
//! transparently routes to this module when it is importable.
//!
//! Design:
//! - One persistent tokio current-thread runtime per process, lazily
//!   built on the first call (`Lazy<Runtime>`). Each Python method
//!   `block_on`s the async client method on this runtime.
//! - `BinaryClient` is `!Sync` in the underlying Rust crate (TCP stream
//!   borrows are exclusive); we wrap it in `Mutex` on the Python side
//!   so multi-thread call sites get serialised correctly.
//! - All errors are mapped to a single `SkegError` Python exception so
//!   the surface matches `skeg.errors.SkegError`.

#![deny(unsafe_code)]
#![allow(clippy::needless_pass_by_value)]

use std::sync::OnceLock;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use tokio::runtime::Runtime;

use skeg_client::{ClientError, SkegClient, VectorBackend as RsBackend, VectorKind as RsKind};

create_exception!(_native, SkegError, PyException);

fn runtime() -> &'static Runtime {
    // Lazily build a single-thread current-thread runtime. We choose
    // current-thread because each Python call already holds the GIL on
    // entry; spawning a multi-thread reactor would add scheduler
    // overhead without throughput benefit for the sync client.
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime build")
    })
}

fn map_err(e: ClientError) -> PyErr {
    SkegError::new_err(e.to_string())
}

fn kind_from_str(s: &str) -> PyResult<RsKind> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "float32" => Ok(RsKind::F32),
        "int8" => Ok(RsKind::Int8),
        "binary" | "bin" => Ok(RsKind::Binary),
        other => Err(SkegError::new_err(format!("unknown vector kind {other:?}"))),
    }
}

fn backend_from_str(s: &str) -> PyResult<RsBackend> {
    match s.to_ascii_lowercase().as_str() {
        "flat" => Ok(RsBackend::Flat),
        "disk" | "disk_vamana" | "vamana" => Ok(RsBackend::DiskVamana),
        other => Err(SkegError::new_err(format!("unknown backend {other:?}"))),
    }
}

#[pyclass(name = "Hit", frozen)]
struct Hit {
    #[pyo3(get)]
    id: u64,
    #[pyo3(get)]
    score: f32,
}

#[pymethods]
impl Hit {
    fn __repr__(&self) -> String {
        format!("Hit(id={}, score={})", self.id, self.score)
    }
}

#[pyclass(name = "BinaryClient")]
struct PyBinaryClient {
    inner: std::sync::Mutex<Option<SkegClient>>,
}

#[pymethods]
impl PyBinaryClient {
    #[new]
    #[pyo3(signature = (host = "127.0.0.1", port = 7379))]
    fn new(host: &str, port: u16) -> PyResult<Self> {
        let addr = format!("{host}:{port}");
        let client = runtime()
            .block_on(SkegClient::connect(addr.as_str()))
            .map_err(|e| SkegError::new_err(format!("connect failed: {e}")))?;
        Ok(Self {
            inner: std::sync::Mutex::new(Some(client)),
        })
    }

    fn close(&self) {
        let mut guard = self.inner.lock().expect("client mutex");
        guard.take();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc=None, _tb=None))]
    fn __exit__(
        &self,
        _exc_type: Option<PyObject>,
        _exc: Option<PyObject>,
        _tb: Option<PyObject>,
    ) {
        self.close();
    }

    fn ping(&self) -> PyResult<()> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime().block_on(client.ping()).map_err(map_err)
    }

    fn get<'py>(&self, py: Python<'py>, key: &[u8]) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        let value = runtime().block_on(client.get(key)).map_err(map_err)?;
        Ok(value.map(|b| PyBytes::new_bound(py, &b)))
    }

    #[pyo3(signature = (key, value, no_reply = false))]
    fn set(&self, key: &[u8], value: &[u8], no_reply: bool) -> PyResult<()> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        if no_reply {
            runtime()
                .block_on(client.set_no_reply(key, value))
                .map_err(map_err)
        } else {
            runtime().block_on(client.set(key, value)).map_err(map_err)
        }
    }

    fn delete(&self, key: &[u8]) -> PyResult<bool> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime().block_on(client.del(key)).map_err(map_err)
    }

    fn mget<'py>(
        &self,
        py: Python<'py>,
        keys: Vec<Vec<u8>>,
    ) -> PyResult<Vec<Option<Bound<'py, PyBytes>>>> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        let key_refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
        let values = runtime().block_on(client.mget(&key_refs)).map_err(map_err)?;
        Ok(values
            .into_iter()
            .map(|opt| opt.map(|b| PyBytes::new_bound(py, &b)))
            .collect())
    }

    #[pyo3(signature = (name, dim, kind = "int8", backend = "flat"))]
    fn vindex_create(
        &self,
        name: &str,
        dim: u32,
        kind: &str,
        backend: &str,
    ) -> PyResult<()> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime()
            .block_on(client.vindex_create(name, dim, kind_from_str(kind)?,
                                            backend_from_str(backend)?))
            .map_err(map_err)
    }

    fn vindex_drop(&self, name: &str) -> PyResult<()> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime().block_on(client.vindex_drop(name)).map_err(map_err)
    }

    fn vindex_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        let rows = runtime().block_on(client.vindex_list()).map_err(map_err)?;
        let list = PyList::empty_bound(py);
        for row in rows {
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("name", row.name)?;
            d.set_item("dim", row.dim)?;
            d.set_item("kind", row.kind)?;
            d.set_item("backend", row.backend)?;
            d.set_item("n_vectors", row.n_vectors)?;
            list.append(d)?;
        }
        Ok(list)
    }

    fn shards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        let rows = runtime().block_on(client.shards()).map_err(map_err)?;
        let list = PyList::empty_bound(py);
        for row in rows {
            let d = pyo3::types::PyDict::new_bound(py);
            d.set_item("shard_id", row.shard_id)?;
            d.set_item("cache_bytes", row.cache_bytes)?;
            d.set_item("cache_evictions", row.cache_evictions)?;
            d.set_item("n_keys", row.n_keys)?;
            d.set_item("cache_budget", row.cache_budget)?;
            list.append(d)?;
        }
        Ok(list)
    }

    fn vset(&self, name: &str, vec_id: u64, vector: Vec<f32>) -> PyResult<()> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime()
            .block_on(client.vset(name, vec_id, &vector))
            .map_err(map_err)
    }

    fn vget(&self, name: &str, vec_id: u64) -> PyResult<Option<Vec<f32>>> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime().block_on(client.vget(name, vec_id)).map_err(map_err)
    }

    fn vdel(&self, name: &str, vec_id: u64) -> PyResult<bool> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        runtime().block_on(client.vdel(name, vec_id)).map_err(map_err)
    }

    #[pyo3(signature = (name, query, k = 10, _l_search = 0))]
    fn vsearch<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        query: Vec<f32>,
        k: u32,
        _l_search: u32,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut guard = self.inner.lock().expect("client mutex");
        let client = guard.as_mut().ok_or_else(|| SkegError::new_err("client closed"))?;
        let hits = runtime()
            .block_on(client.vsearch(name, &query, k))
            .map_err(map_err)?;
        let list = PyList::empty_bound(py);
        for (id, score) in hits {
            let h = Py::new(py, Hit { id, score })?;
            list.append(h)?;
        }
        Ok(list)
    }
}

#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SkegError", py.get_type_bound::<SkegError>())?;
    m.add_class::<Hit>()?;
    m.add_class::<PyBinaryClient>()?;
    Ok(())
}
