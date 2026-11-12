#[cfg(target_endian = "big")]
compile_error!("emcfile_fast requires a little-endian target");

pub mod detector;
pub mod emc;
pub mod h5;
pub mod parquet;
pub mod simd;

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, OnceLock};

use numpy::{
    Element, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2, PyUntypedArrayMethods,
};
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

fn max_pool_threads() -> usize {
    static MAX_THREADS: OnceLock<usize> = OnceLock::new();
    *MAX_THREADS.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
            .clamp(4, 64)
    })
}

/// Return a cached Rayon `ThreadPool` keyed by `(pid, n_threads)`.
///
/// Keying by `std::process::id()` ensures that after `os.fork()` on Unix, child
/// processes never attempt to dispatch work onto dead parent worker threads (and
/// never drop parent `ThreadPool` instances whose `Drop` would join non-existent
/// threads).
type PoolMap = HashMap<(u32, usize), Arc<rayon::ThreadPool>>;

pub(crate) fn get_pool(n_threads: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<PoolMap>> = OnceLock::new();
    let mtx = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = mtx.lock().unwrap_or_else(|e| e.into_inner());
    let pid = std::process::id();
    let n = n_threads.clamp(1, max_pool_threads());
    map.entry((pid, n))
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
    let n = buf.len_bytes();
    if n == 0 {
        return Ok(&[]);
    }
    let ptr = buf.buf_ptr() as *const u8;
    if ptr.is_null() {
        return Err(PyValueError::new_err("buffer pointer is null"));
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    Ok(slice)
}

#[inline]
pub(crate) fn view_pybuffer_u8_detachable<'a>(
    py: Python<'_>,
    buf: &'a PyBuffer<u8>,
) -> PyResult<Cow<'a, [u8]>> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("buffer must be C-contiguous"));
    }
    if buf.readonly() {
        Ok(Cow::Borrowed(view_pybuffer_u8(buf)?))
    } else {
        Ok(Cow::Owned(buf.to_vec(py)?))
    }
}

#[inline]
pub(crate) fn as_c_slice_2d<'a, T: Element>(
    arr: &'a PyReadonlyArray2<'_, T>,
    cols: usize,
    name: &str,
) -> PyResult<&'a [T]> {
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != cols {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {cols})"
        )));
    }
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

#[inline]
pub(crate) fn as_c_slice_mut_2d<'a, T: Element>(
    arr: &'a mut PyReadwriteArray2<'_, T>,
    cols: usize,
    name: &str,
) -> PyResult<&'a mut [T]> {
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != cols {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {cols})"
        )));
    }
    arr.as_slice_mut()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

#[inline]
pub(crate) fn as_c_slice_3d<'a, T: Element>(
    arr: &'a PyReadonlyArray3<'_, T>,
    d1: usize,
    d2: usize,
    name: &str,
) -> PyResult<&'a [T]> {
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    let shape = arr.shape();
    if shape.len() != 3 || shape[1] != d1 || shape[2] != d2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {d1}, {d2})"
        )));
    }
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be C-contiguous")))
}

/// Cross-platform borrowed OS file handle/descriptor extracted while holding the GIL.
///
/// The caller must keep the originating Python `&Bound<'_, PyAny>` alive for the
/// entire duration that `RawFileTarget::write_all` may be invoked.
#[derive(Clone, Copy)]
pub(crate) struct RawFileTarget {
    #[cfg(unix)]
    fd: std::os::fd::RawFd,
    #[cfg(windows)]
    handle: usize,
}

impl RawFileTarget {
    pub fn from_py(py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<Self> {
        #[cfg(unix)]
        {
            let _ = py;
            let fd = if let Ok(raw_fd) = target.extract::<i32>() {
                raw_fd
            } else {
                target.call_method0("fileno")?.extract::<i32>()?
            };
            if fd < 0 {
                return Err(PyValueError::new_err("invalid negative file descriptor"));
            }
            Ok(Self { fd })
        }
        #[cfg(windows)]
        {
            let crt_fd = if let Ok(raw_fd) = target.extract::<i64>() {
                raw_fd
            } else {
                target.call_method0("fileno")?.extract::<i64>()?
            };
            if crt_fd < 0 {
                return Err(PyValueError::new_err("invalid negative file descriptor"));
            }
            let msvcrt = py.import("msvcrt")?;
            let osf = msvcrt
                .call_method1("get_osfhandle", (crt_fd,))?
                .extract::<isize>()?;
            if osf == -1 || osf == 0 {
                return Err(PyValueError::new_err("invalid Windows OS file handle"));
            }
            Ok(Self {
                handle: osf as usize,
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (py, target);
            Err(PyValueError::new_err(
                "direct OS file writing is only supported on Unix and Windows",
            ))
        }
    }

    #[inline]
    pub fn write_all(self, buf: &[u8]) -> bool {
        if buf.is_empty() {
            return true;
        }
        #[cfg(unix)]
        {
            use std::os::fd::FromRawFd;
            let mut f = ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(self.fd) });
            f.write_all(buf).is_ok()
        }
        #[cfg(windows)]
        {
            use std::os::windows::io::{FromRawHandle, RawHandle};
            let mut f = ManuallyDrop::new(unsafe {
                std::fs::File::from_raw_handle(self.handle as RawHandle)
            });
            f.write_all(buf).is_ok()
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = buf;
            false
        }
    }
}

#[pymodule]
fn emcfile_fast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    h5::register(m)?;
    parquet::register(m)?;
    detector::register(m)?;
    emc::register(m)?;
    let py = m.py();
    let h5_mod = PyModule::new(py, "h5")?;
    h5::register(&h5_mod)?;
    m.add_submodule(&h5_mod)?;
    let pq_mod = PyModule::new(py, "parquet")?;
    parquet::register(&pq_mod)?;
    m.add_submodule(&pq_mod)?;
    let det_mod = PyModule::new(py, "detector")?;
    detector::register(&det_mod)?;
    m.add_submodule(&det_mod)?;
    let emc_mod = PyModule::new(py, "emc")?;
    emc::register(&emc_mod)?;
    m.add_submodule(&emc_mod)?;
    Ok(())
}
