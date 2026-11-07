pub mod h5;
pub mod parquet;
pub mod simd;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) struct ThreadZstd {
    pub cctx: zstd_safe::CCtx<'static>,
    pub dctx: zstd_safe::DCtx<'static>,
    pub scratch: Vec<u8>,
    pub comp_scratch: Vec<u8>,
}

impl ThreadZstd {
    fn new() -> Self {
        Self {
            cctx: zstd_safe::CCtx::create(),
            dctx: zstd_safe::DCtx::create(),
            scratch: Vec::new(),
            comp_scratch: Vec::new(),
        }
    }

    #[inline]
    pub fn ensure_scratch(&mut self, n: usize) {
        if self.scratch.len() < n {
            self.scratch.resize(n, 0);
        }
    }

    #[inline]
    pub fn ensure_comp_scratch(&mut self, n: usize) {
        if self.comp_scratch.len() < n {
            self.comp_scratch.resize(n, 0);
        }
    }
}

thread_local! {
    pub(crate) static THREAD_ZSTD: RefCell<ThreadZstd> = RefCell::new(ThreadZstd::new());
}

pub(crate) fn get_pool(n_threads: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mtx = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = mtx.lock().unwrap();
    let n = n_threads.max(1);
    map.entry(n)
        .or_insert_with(|| {
            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .expect("failed to build rayon ThreadPool"),
            )
        })
        .clone()
}

#[inline]
pub(crate) fn view_pybuffer_u8(buf: &PyBuffer<u8>) -> PyResult<&[u8]> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("buffer must be C-contiguous"));
    }
    let slice =
        unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, buf.len_bytes()) };
    Ok(slice)
}

#[pymodule]
fn emcfile_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    h5::register(m)?;
    parquet::register(m)?;
    let py = m.py();
    let h5_mod = PyModule::new(py, "h5")?;
    h5::register(&h5_mod)?;
    m.add_submodule(&h5_mod)?;
    let pq_mod = PyModule::new(py, "parquet")?;
    parquet::register(&pq_mod)?;
    m.add_submodule(&pq_mod)?;
    Ok(())
}
