use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

type CreateCCtxFn = unsafe extern "C" fn() -> *mut c_void;
type FreeCCtxFn = unsafe extern "C" fn(*mut c_void) -> usize;
type CompressCCtxFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *const c_void, usize, c_int) -> usize;
type CompressBoundFn = unsafe extern "C" fn(usize) -> usize;
type CreateDCtxFn = unsafe extern "C" fn() -> *mut c_void;
type FreeDCtxFn = unsafe extern "C" fn(*mut c_void) -> usize;
type DecompressDCtxFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, usize, *const c_void, usize) -> usize;
type IsErrorFn = unsafe extern "C" fn(usize) -> u32;

pub struct ZstdApi {
    pub create_cctx: CreateCCtxFn,
    pub free_cctx: FreeCCtxFn,
    pub compress_cctx: CompressCCtxFn,
    pub compress_bound: CompressBoundFn,
    pub create_dctx: CreateDCtxFn,
    pub free_dctx: FreeDCtxFn,
    pub decompress_dctx: DecompressDCtxFn,
    pub is_error: IsErrorFn,
}

unsafe impl Send for ZstdApi {}
unsafe impl Sync for ZstdApi {}

static ZSTD_API: OnceLock<Option<ZstdApi>> = OnceLock::new();

pub fn get_zstd() -> Option<&'static ZstdApi> {
    ZSTD_API
        .get_or_init(|| {
            let candidates = [
                "libzstd.so.1",
                "libzstd.so",
                "libzstd.1.dylib",
                "libzstd.dylib",
            ];
            for cand in candidates {
                let c_cand = CString::new(cand).ok()?;
                unsafe {
                    let handle = dlopen(c_cand.as_ptr(), RTLD_NOW);
                    if handle.is_null() {
                        continue;
                    }
                    macro_rules! sym {
                        ($name:literal, $ty:ty) => {{
                            let cname = CString::new($name).ok()?;
                            let ptr = dlsym(handle, cname.as_ptr());
                            if ptr.is_null() {
                                continue;
                            }
                            std::mem::transmute::<*mut c_void, $ty>(ptr)
                        }};
                    }
                    return Some(ZstdApi {
                        create_cctx: sym!("ZSTD_createCCtx", CreateCCtxFn),
                        free_cctx: sym!("ZSTD_freeCCtx", FreeCCtxFn),
                        compress_cctx: sym!("ZSTD_compressCCtx", CompressCCtxFn),
                        compress_bound: sym!("ZSTD_compressBound", CompressBoundFn),
                        create_dctx: sym!("ZSTD_createDCtx", CreateDCtxFn),
                        free_dctx: sym!("ZSTD_freeDCtx", FreeDCtxFn),
                        decompress_dctx: sym!("ZSTD_decompressDCtx", DecompressDCtxFn),
                        is_error: sym!("ZSTD_isError", IsErrorFn),
                    });
                }
            }
            None
        })
        .as_ref()
}

pub struct ThreadZstdCtx {
    pub cctx: *mut c_void,
    pub dctx: *mut c_void,
    pub scratch: Vec<u8>,
    pub comp_scratch: Vec<u8>,
}

impl ThreadZstdCtx {
    fn new() -> Self {
        let (cctx, dctx) = if let Some(api) = get_zstd() {
            unsafe { ((api.create_cctx)(), (api.create_dctx)()) }
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        Self {
            cctx,
            dctx,
            scratch: Vec::with_capacity(256 * 1024),
            comp_scratch: Vec::with_capacity(256 * 1024),
        }
    }

    #[inline]
    pub fn ensure_scratch(&mut self, len: usize) -> &mut [u8] {
        if self.scratch.len() < len {
            self.scratch.resize(len.max(256 * 1024), 0);
        }
        &mut self.scratch[..len]
    }

    #[inline]
    pub fn ensure_comp_scratch(&mut self, len: usize) -> &mut [u8] {
        if self.comp_scratch.len() < len {
            self.comp_scratch.resize(len.max(256 * 1024), 0);
        }
        &mut self.comp_scratch[..len]
    }
}

impl Drop for ThreadZstdCtx {
    fn drop(&mut self) {
        if let Some(api) = get_zstd() {
            unsafe {
                if !self.cctx.is_null() {
                    (api.free_cctx)(self.cctx);
                }
                if !self.dctx.is_null() {
                    (api.free_dctx)(self.dctx);
                }
            }
        }
    }
}

thread_local! {
    pub static THREAD_ZSTD: RefCell<ThreadZstdCtx> = RefCell::new(ThreadZstdCtx::new());
}
