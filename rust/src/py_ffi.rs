#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_longlong, c_void};

pub type Py_ssize_t = isize;

#[repr(C)]
pub struct PyObject {
    pub ob_refcnt: Py_ssize_t,
    pub ob_type: *mut c_void,
}

#[repr(C)]
pub struct PyModuleDef_Base {
    pub ob_base: PyObject,
    pub m_init: Option<unsafe extern "C" fn() -> *mut PyObject>,
    pub m_index: Py_ssize_t,
    pub m_copy: *mut PyObject,
}

#[repr(C)]
pub struct PyMethodDef {
    pub ml_name: *const c_char,
    pub ml_meth: Option<
        unsafe extern "C" fn(*mut PyObject, *const *mut PyObject, Py_ssize_t) -> *mut PyObject,
    >,
    pub ml_flags: c_int,
    pub ml_doc: *const c_char,
}

unsafe impl Sync for PyMethodDef {}

#[repr(C)]
pub struct PyModuleDef {
    pub m_base: PyModuleDef_Base,
    pub m_name: *const c_char,
    pub m_doc: *const c_char,
    pub m_size: Py_ssize_t,
    pub m_methods: *const PyMethodDef,
    pub m_slots: *const c_void,
    pub m_traverse: *const c_void,
    pub m_clear: *const c_void,
    pub m_free: *const c_void,
}

unsafe impl Sync for PyModuleDef {}

#[repr(C)]
pub struct Py_buffer {
    pub buf: *mut c_void,
    pub obj: *mut PyObject,
    pub len: Py_ssize_t,
    pub itemsize: Py_ssize_t,
    pub readonly: c_int,
    pub ndim: c_int,
    pub format: *mut c_char,
    pub shape: *mut Py_ssize_t,
    pub strides: *mut Py_ssize_t,
    pub suboffsets: *mut Py_ssize_t,
    pub internal: *mut c_void,
}

pub const METH_FASTCALL: c_int = 0x0080;
pub const PY_BUF_SIMPLE: c_int = 0;
pub const PY_BUF_WRITABLE: c_int = 0x0001;
pub const PYTHON_ABI_VERSION: c_int = 1013;

unsafe extern "C" {
    pub fn PyModule_Create2(def: *mut PyModuleDef, apiver: c_int) -> *mut PyObject;
    pub fn PyObject_GetBuffer(obj: *mut PyObject, view: *mut Py_buffer, flags: c_int) -> c_int;
    pub fn PyBuffer_Release(view: *mut Py_buffer);
    pub fn PyLong_AsLongLong(obj: *mut PyObject) -> c_longlong;
    pub fn PyLong_FromLongLong(v: c_longlong) -> *mut PyObject;
    pub fn PyObject_IsTrue(obj: *mut PyObject) -> c_int;
    pub fn PyEval_SaveThread() -> *mut c_void;
    pub fn PyEval_RestoreThread(tstate: *mut c_void);
    pub fn PyErr_SetString(exception: *mut PyObject, message: *const c_char);
    pub static mut PyExc_ValueError: *mut PyObject;
    pub static mut PyExc_RuntimeError: *mut PyObject;
}

pub struct AllowThreads {
    tstate: *mut c_void,
}

impl AllowThreads {
    #[inline]
    pub fn new() -> Self {
        Self {
            tstate: unsafe { PyEval_SaveThread() },
        }
    }
}

impl Drop for AllowThreads {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            PyEval_RestoreThread(self.tstate);
        }
    }
}

pub struct PyBuf {
    raw: Py_buffer,
}

impl PyBuf {
    #[inline]
    pub fn get_readonly(obj: *mut PyObject) -> Option<Self> {
        let mut raw = unsafe { std::mem::zeroed::<Py_buffer>() };
        if unsafe { PyObject_GetBuffer(obj, &mut raw, PY_BUF_SIMPLE) } != 0 {
            return None;
        }
        Some(Self { raw })
    }

    #[inline]
    pub fn get_writable(obj: *mut PyObject) -> Option<Self> {
        let mut raw = unsafe { std::mem::zeroed::<Py_buffer>() };
        if unsafe { PyObject_GetBuffer(obj, &mut raw, PY_BUF_WRITABLE) } != 0 {
            return None;
        }
        Some(Self { raw })
    }

    #[inline]
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        let elem_size = std::mem::size_of::<T>();
        let n = (self.raw.len as usize) / elem_size;
        if n == 0 || self.raw.buf.is_null() {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.raw.buf as *const T, n) }
    }

    #[inline]
    pub fn as_mut_slice<T: Copy>(&mut self) -> &mut [T] {
        let elem_size = std::mem::size_of::<T>();
        let n = (self.raw.len as usize) / elem_size;
        if n == 0 || self.raw.buf.is_null() {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.raw.buf as *mut T, n) }
    }
}

impl Drop for PyBuf {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            PyBuffer_Release(&mut self.raw);
        }
    }
}

#[inline]
pub fn arg_i64(args: *const *mut PyObject, idx: usize) -> i64 {
    unsafe { PyLong_AsLongLong(*args.add(idx)) as i64 }
}

#[inline]
pub fn arg_usize(args: *const *mut PyObject, idx: usize) -> usize {
    arg_i64(args, idx) as usize
}

#[inline]
pub fn arg_bool(args: *const *mut PyObject, idx: usize) -> bool {
    unsafe { PyObject_IsTrue(*args.add(idx)) != 0 }
}

#[inline]
pub fn raise_value_error(msg_cstr: &'static [u8]) -> *mut PyObject {
    unsafe {
        PyErr_SetString(PyExc_ValueError, msg_cstr.as_ptr() as *const c_char);
    }
    std::ptr::null_mut()
}

#[inline]
pub fn raise_runtime_error(msg_cstr: &'static [u8]) -> *mut PyObject {
    unsafe {
        PyErr_SetString(PyExc_RuntimeError, msg_cstr.as_ptr() as *const c_char);
    }
    std::ptr::null_mut()
}
