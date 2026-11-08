use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Mutex, OnceLock};

use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
    PyReadwriteArray2,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

use crate::simd::{
    cumsum_u32_inplace, delta_and_shuffle_segment_u32, shuffle_chunk_u32, unshuffle_span_u32,
    unshuffle_span_u32_opt_cumsum, zero_pad_chunk_planes,
};
use crate::{get_pool, view_pybuffer_u8, THREAD_ZSTD};

struct CachedChunk {
    key: (usize, i64, i64, usize),
    buf: Vec<u8>,
}

impl CachedChunk {
    fn new() -> Self {
        Self {
            key: (0, -1, -1, 0),
            buf: Vec::new(),
        }
    }
}

fn get_single_cache() -> &'static [Mutex<CachedChunk>; 3] {
    static CACHE: OnceLock<[Mutex<CachedChunk>; 3]> = OnceLock::new();
    CACHE.get_or_init(|| {
        [
            Mutex::new(CachedChunk::new()),
            Mutex::new(CachedChunk::new()),
            Mutex::new(CachedChunk::new()),
        ]
    })
}

#[inline]
unsafe fn copy_unaligned_u32(raw_u8: &[u8], start: usize, dst: &mut [u32]) {
    let count = dst.len();
    if count > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                raw_u8.as_ptr().add(start * 4),
                dst.as_mut_ptr() as *mut u8,
                count * 4,
            );
        }
    }
}

#[inline]
fn upper_bound_u64(arr: &[u64], target: u64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if arr[mid] <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

fn encode_single_chunk_into(
    values: &[u32],
    offsets: &[u64],
    c_idx: usize,
    chunk_len: usize,
    level: i32,
    is_delta: bool,
    check_sorted: bool,
    cctx: &mut zstd_safe::CCtx<'static>,
    shuffled: &mut [u8],
    comp_dst: &mut [u8],
) -> Result<usize, i32> {
    let total_len = values.len();
    let start = c_idx * chunk_len;
    let stop = (start + chunk_len).min(total_len);
    let count = stop - start;

    if !is_delta {
        shuffle_chunk_u32(&values[start..stop], chunk_len, shuffled);
    } else {
        let n_pats = offsets.len().saturating_sub(1);
        let mut p = if n_pats > 0 {
            let idx = upper_bound_u64(offsets, start as u64);
            idx.saturating_sub(1).min(n_pats - 1)
        } else {
            0
        };

        while p < n_pats {
            let ps = offsets[p] as usize;
            if ps >= stop {
                break;
            }
            let pe = offsets[p + 1] as usize;
            if pe > ps {
                let seg_s = ps.max(start);
                let seg_e = pe.min(stop);
                if seg_s < seg_e {
                    let init_prev = if seg_s == ps || seg_s == 0 {
                        0u32
                    } else {
                        values[seg_s - 1]
                    };
                    let rc = delta_and_shuffle_segment_u32(
                        &values[seg_s..seg_e],
                        init_prev,
                        check_sorted,
                        chunk_len,
                        seg_s - start,
                        shuffled,
                    );
                    if rc != 0 {
                        return Err(rc);
                    }
                }
            }
            p += 1;
        }
        zero_pad_chunk_planes(count, chunk_len, shuffled);
    }

    let u_bytes = chunk_len * 4;
    let c_len = cctx
        .compress(comp_dst, &shuffled[..u_bytes], level)
        .map_err(|_| -3i32)?;
    Ok(c_len)
}

#[pyfunction]
pub fn compress_raw_zstd<'py>(
    py: Python<'py>,
    b_data: &[u8],
    level: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let c_bound = zstd_safe::compress_bound(b_data.len()) + 64;
    let (buf, c_len) = THREAD_ZSTD
        .with(|cell| {
            let mut ctx = cell.borrow_mut();
            let mut out = vec![0u8; c_bound];
            match ctx.cctx.compress(&mut out[..], b_data, level) {
                Ok(n) => Ok((out, n)),
                Err(_) => Err(-3i32),
            }
        })
        .map_err(|_| PyRuntimeError::new_err("Zstd compression failed"))?;
    Ok(PyBytes::new(py, &buf[..c_len]))
}

#[pyfunction]
pub fn decode_raw_chunk<'py>(
    py: Python<'py>,
    mask: i64,
    b_payload: PyBuffer<u8>,
    elements: usize,
    sel_start: usize,
    sel_stop: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    if (mask & !3) != 0 {
        return Err(PyValueError::new_err("Unexpected HDF5 v2 chunk filter mask"));
    }
    let payload = view_pybuffer_u8(&b_payload)?;
    let u_bytes = elements * 4;
    let start = sel_start.min(elements);
    let stop = sel_stop.min(elements).max(start);
    let count = stop - start;
    let out_arr = unsafe { PyArray1::<u32>::new(py, [count], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let rc = THREAD_ZSTD.with(|cell| {
        let mut ctx = cell.borrow_mut();
        let zctx = &mut *ctx;

        let raw_slice: &[u8] = if (mask & 2) != 0 {
            if payload.len() != u_bytes {
                return -1;
            }
            payload
        } else {
            zctx.ensure_scratch(u_bytes);
            match zctx
                .dctx
                .decompress(&mut zctx.scratch[..u_bytes], payload)
            {
                Ok(n) if n == u_bytes => &zctx.scratch[..u_bytes],
                _ => return -1,
            }
        };

        if count > 0 {
            if (mask & 1) != 0 {
                unsafe {
                    copy_unaligned_u32(raw_slice, start, out_slice);
                }
            } else {
                unshuffle_span_u32(raw_slice, elements, start, out_slice);
            }
        }
        0
    });

    if rc != 0 {
        return Err(PyValueError::new_err(
            "Unexpected HDF5 v2 decoded chunk length",
        ));
    }
    Ok(out_arr)
}

#[pyfunction]
pub fn compress_chunk<'py>(
    py: Python<'py>,
    b_values: PyReadonlyArray1<'py, u32>,
    chunk_len: usize,
    level: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let values = b_values.as_slice()?;
    let u_bytes = chunk_len * 4;
    let c_bound = zstd_safe::compress_bound(u_bytes) + 64;

    let (buf, c_len) = py
        .detach(|| {
            THREAD_ZSTD.with(|cell| {
                let mut ctx = cell.borrow_mut();
                ctx.ensure_scratch(u_bytes);
                let zctx = &mut *ctx;
                shuffle_chunk_u32(values, chunk_len, &mut zctx.scratch[..u_bytes]);
                let mut out = vec![0u8; c_bound];
                let res = zctx
                    .cctx
                    .compress(&mut out[..], &zctx.scratch[..u_bytes], level);
                match res {
                    Ok(n) => Ok((out, n)),
                    Err(_) => Err(-3i32),
                }
            })
        })
        .map_err(|_| PyRuntimeError::new_err("Zstd compression failed"))?;

    Ok(PyBytes::new(py, &buf[..c_len]))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn encode_chunks_batch<'py>(
    py: Python<'py>,
    b_values: PyReadonlyArray1<'py, u32>,
    b_offsets: PyReadonlyArray1<'py, u64>,
    c_first: usize,
    c_last: usize,
    chunk_len: usize,
    level: i32,
    is_delta: bool,
    check_sorted: bool,
    n_workers: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyArray2<i64>>)> {
    let values = b_values.as_slice()?;
    let offsets = b_offsets.as_slice()?;
    let n_batch = c_last.saturating_sub(c_first);
    if n_batch == 0 {
        let empty_b = PyBytes::new(py, &[]);
        let empty_m = PyArray2::<i64>::zeros(py, [0, 3], false);
        return Ok((empty_b, empty_m));
    }

    let u_bytes = chunk_len * 4;
    let c_bound = zstd_safe::compress_bound(u_bytes) + 64;
    let err_flag = AtomicI32::new(0);

    let results: Vec<(usize, Vec<u8>)> = py.detach(|| {
        let encode_one = |idx: usize| -> (usize, Vec<u8>) {
            if err_flag.load(Ordering::Relaxed) != 0 {
                return (0, Vec::new());
            }
            let c_idx = c_first + idx;
            THREAD_ZSTD.with(|cell| {
                let mut ctx = cell.borrow_mut();
                ctx.ensure_scratch(u_bytes);
                ctx.ensure_comp_scratch(c_bound);
                let zctx = &mut *ctx;
                match encode_single_chunk_into(
                    values,
                    offsets,
                    c_idx,
                    chunk_len,
                    level,
                    is_delta,
                    check_sorted,
                    &mut zctx.cctx,
                    &mut zctx.scratch[..u_bytes],
                    &mut zctx.comp_scratch[..c_bound],
                ) {
                    Ok(c_len) => (c_idx * chunk_len, zctx.comp_scratch[..c_len].to_vec()),
                    Err(rc) => {
                        err_flag.store(rc, Ordering::Relaxed);
                        (0, Vec::new())
                    }
                }
            })
        };

        if n_workers <= 1 || n_batch <= 1 {
            (0..n_batch).map(encode_one).collect()
        } else {
            get_pool(n_workers.min(n_batch))
                .install(|| (0..n_batch).into_par_iter().map(encode_one).collect())
        }
    });

    let rc = err_flag.load(Ordering::Relaxed);
    if rc == -2 {
        return Err(PyValueError::new_err(
            "position values must be sorted within every pattern",
        ));
    }
    if rc != 0 {
        return Err(PyRuntimeError::new_err("Zstd compression failed in _h5_rs"));
    }

    let total_comp: usize = results.iter().map(|(_, v)| v.len()).sum();
    let mut meta_vec = Vec::with_capacity(n_batch * 3);
    let packed = PyBytes::new_with(py, total_comp, |dst: &mut [u8]| {
        let mut pos = 0usize;
        for (start_elem, vec) in &results {
            let l = vec.len();
            dst[pos..pos + l].copy_from_slice(vec);
            meta_vec.push(*start_elem as i64);
            meta_vec.push(pos as i64);
            meta_vec.push(l as i64);
            pos += l;
        }
        Ok(())
    })?;

    let meta_arr = PyArray1::from_vec(py, meta_vec).reshape([n_batch, 3])?;
    Ok((packed, meta_arr))
}

#[pyfunction]
pub fn compress_bound_u32(chunk_len: usize) -> usize {
    zstd_safe::compress_bound(chunk_len * 4) + 64
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn encode_chunks_into<'py>(
    py: Python<'py>,
    b_values: PyReadonlyArray1<'py, u32>,
    b_offsets: PyReadonlyArray1<'py, u64>,
    c_first: usize,
    c_last: usize,
    chunk_len: usize,
    level: i32,
    is_delta: bool,
    check_sorted: bool,
    n_workers: usize,
    mut b_out: PyReadwriteArray1<'py, u8>,
    mut b_meta: PyReadwriteArray2<'py, i64>,
) -> PyResult<usize> {
    let values = b_values.as_slice()?;
    let offsets = b_offsets.as_slice()?;
    let n_batch = c_last.saturating_sub(c_first);
    if n_batch == 0 {
        return Ok(0);
    }

    let u_bytes = chunk_len * 4;
    let c_bound = zstd_safe::compress_bound(u_bytes) + 64;
    let out_slice = b_out.as_slice_mut()?;
    let meta_slice = b_meta.as_slice_mut()?;
    if out_slice.len() < n_batch * c_bound || meta_slice.len() < n_batch * 3 {
        return Err(PyValueError::new_err("output buffers too small for batch"));
    }

    let out_addr = out_slice.as_mut_ptr() as usize;
    let meta_addr = meta_slice.as_mut_ptr() as usize;
    let err_flag = AtomicI32::new(0);

    py.detach(|| {
        let encode_one = |idx: usize| {
            if err_flag.load(Ordering::Relaxed) != 0 {
                return;
            }
            let c_idx = c_first + idx;
            let dst_comp = unsafe {
                std::slice::from_raw_parts_mut(
                    (out_addr as *mut u8).add(idx * c_bound),
                    c_bound,
                )
            };
            let dst_meta = unsafe {
                std::slice::from_raw_parts_mut((meta_addr as *mut i64).add(idx * 3), 3)
            };
            THREAD_ZSTD.with(|cell| {
                let mut ctx = cell.borrow_mut();
                ctx.ensure_scratch(u_bytes);
                let zctx = &mut *ctx;
                match encode_single_chunk_into(
                    values,
                    offsets,
                    c_idx,
                    chunk_len,
                    level,
                    is_delta,
                    check_sorted,
                    &mut zctx.cctx,
                    &mut zctx.scratch[..u_bytes],
                    dst_comp,
                ) {
                    Ok(c_len) => {
                        dst_meta[0] = (c_idx * chunk_len) as i64;
                        dst_meta[1] = (idx * c_bound) as i64;
                        dst_meta[2] = c_len as i64;
                    }
                    Err(rc) => {
                        err_flag.store(rc, Ordering::Relaxed);
                    }
                }
            });
        };

        if n_workers <= 1 || n_batch <= 1 {
            for idx in 0..n_batch {
                encode_one(idx);
            }
        } else {
            get_pool(n_workers.min(n_batch))
                .install(|| (0..n_batch).into_par_iter().for_each(encode_one));
        }
    });

    let rc = err_flag.load(Ordering::Relaxed);
    if rc == -2 {
        return Err(PyValueError::new_err(
            "position values must be sorted within every pattern",
        ));
    }
    if rc != 0 {
        return Err(PyRuntimeError::new_err("Zstd compression failed in _h5_rs"));
    }
    Ok(n_batch)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_dataset_mmap<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_table: PyReadonlyArray2<'py, i64>,
    chunk_len: usize,
    mut b_out: PyReadwriteArray1<'py, u32>,
    b_offsets: PyReadonlyArray1<'py, u64>,
    is_delta: bool,
    n_workers: usize,
) -> PyResult<i64> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let table = b_table.as_slice()?;
    let offsets = b_offsets.as_slice()?;
    let out_slice = b_out.as_slice_mut()?;

    let total_len = out_slice.len();
    let n_chunks = table.len() / 3;
    if total_len == 0 || n_chunks == 0 {
        return Ok(0);
    }

    let out_addr = out_slice.as_mut_ptr() as usize;
    let u_bytes = chunk_len * 4;
    let n_pats = offsets.len().saturating_sub(1);
    let n_workers = n_workers.max(1).min(n_chunks);

    let decode_chunk = |c_idx: usize| -> Result<(), i32> {
        let b_off = table[c_idx * 3] as usize;
        let c_sz = table[c_idx * 3 + 1] as usize;
        let mask = table[c_idx * 3 + 2];
        if (mask & !3) != 0 {
            return Err(-2);
        }
        let start = c_idx * chunk_len;
        let stop = (start + chunk_len).min(total_len);
        let out_u32 = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, total_len) };

        THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;

            let raw_slice: &[u8] = if (mask & 2) != 0 {
                if c_sz != u_bytes || b_off.saturating_add(u_bytes) > mm_u8.len() {
                    return Err(-1);
                }
                &mm_u8[b_off..b_off + u_bytes]
            } else {
                if b_off.saturating_add(c_sz) > mm_u8.len() {
                    return Err(-1);
                }
                zctx.ensure_scratch(u_bytes);
                match zctx
                    .dctx
                    .decompress(&mut zctx.scratch[..u_bytes], &mm_u8[b_off..b_off + c_sz])
                {
                    Ok(n) if n == u_bytes => &zctx.scratch[..u_bytes],
                    _ => return Err(-1),
                }
            };

            if !is_delta {
                if (mask & 1) != 0 {
                    unsafe {
                        copy_unaligned_u32(raw_slice, 0, &mut out_u32[start..stop]);
                    }
                } else {
                    unshuffle_span_u32(raw_slice, chunk_len, 0, &mut out_u32[start..stop]);
                }
            } else {
                let mut p = {
                    let idx = upper_bound_u64(offsets, start as u64);
                    idx.saturating_sub(1).min(n_pats.saturating_sub(1))
                };
                while p < n_pats {
                    let ps = offsets[p] as usize;
                    let pe = offsets[p + 1] as usize;
                    if ps >= stop {
                        break;
                    }
                    if pe > ps {
                        let seg_s = ps.max(start);
                        let seg_e = pe.min(stop);
                        if seg_s < seg_e {
                            let local_s = seg_s - start;
                            let full_pat = ps >= start && pe <= stop;
                            if (mask & 1) != 0 {
                                unsafe {
                                    copy_unaligned_u32(
                                        raw_slice,
                                        local_s,
                                        &mut out_u32[seg_s..seg_e],
                                    );
                                }
                                if full_pat {
                                    cumsum_u32_inplace(&mut out_u32[ps..pe]);
                                }
                            } else {
                                unshuffle_span_u32_opt_cumsum(
                                    raw_slice,
                                    chunk_len,
                                    local_s,
                                    &mut out_u32[seg_s..seg_e],
                                    full_pat,
                                );
                            }
                        }
                    }
                    p += 1;
                }
            }
            Ok(())
        })
    };

    let res = py.detach(|| {
        let r = if n_workers <= 1 || n_chunks <= 1 {
            (0..n_chunks).try_for_each(decode_chunk)
        } else {
            get_pool(n_workers).install(|| (0..n_chunks).into_par_iter().try_for_each(decode_chunk))
        };
        if r.is_ok() && is_delta && n_pats > 0 {
            let mut straddle_pats: Vec<(usize, usize)> = Vec::new();
            let mut last_straddle_pat = usize::MAX;
            for c_idx in 1..n_chunks {
                let boundary = (c_idx * chunk_len) as u64;
                let idx = upper_bound_u64(offsets, boundary);
                if idx > 0 && idx <= n_pats {
                    let p = idx - 1;
                    if p != last_straddle_pat {
                        let ps = offsets[p] as usize;
                        let pe = offsets[p + 1] as usize;
                        if ps < (boundary as usize) && (boundary as usize) < pe {
                            straddle_pats.push((ps, pe));
                            last_straddle_pat = p;
                        }
                    }
                }
            }
            if n_workers <= 1 || straddle_pats.len() < 64 {
                let out_u32 =
                    unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, total_len) };
                for (ps, pe) in straddle_pats {
                    cumsum_u32_inplace(&mut out_u32[ps..pe]);
                }
            } else {
                get_pool(n_workers).install(|| {
                    straddle_pats.par_iter().for_each(|&(ps, pe)| {
                        let seg = unsafe {
                            std::slice::from_raw_parts_mut(
                                (out_addr as *mut u32).add(ps),
                                pe - ps,
                            )
                        };
                        cumsum_u32_inplace(seg);
                    });
                });
            }
        }
        r
    });

    match res {
        Ok(()) => Ok(0),
        Err(-2) => Err(PyValueError::new_err("Unexpected HDF5 v2 chunk filter mask")),
        Err(_) => Err(PyValueError::new_err(
            "Unexpected HDF5 v2 decoded chunk length",
        )),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_indexed_mmap<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_table: PyReadonlyArray2<'py, i64>,
    chunk_len: usize,
    b_ranges: PyReadonlyArray2<'py, i64>,
    b_offsets: PyReadonlyArray1<'py, u64>,
    is_delta: bool,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let table = b_table.as_slice()?;
    let ranges = b_ranges.as_slice()?;
    let offsets = b_offsets.as_slice()?;
    let n_ranges = ranges.len() / 2;

    let mut total_elems = 0usize;
    let mut n_sel_pats = 0usize;
    for r in 0..n_ranges {
        let first = ranges[r * 2] as usize;
        let last = ranges[r * 2 + 1] as usize;
        if last > first {
            total_elems += (offsets[last] - offsets[first]) as usize;
            n_sel_pats += last - first;
        }
    }

    let out_arr = unsafe { PyArray1::<u32>::new(py, [total_elems], false) };
    if total_elems == 0 {
        return Ok(out_arr);
    }
    let out_slice = unsafe { out_arr.as_slice_mut()? };
    let out_addr = out_slice.as_mut_ptr() as usize;

    let res = py.detach(|| -> Result<(), i32> {
        let mut chunk_map: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        let mut boundaries: Vec<usize> = if is_delta {
            Vec::with_capacity(n_sel_pats + 1)
        } else {
            Vec::new()
        };
        if is_delta {
            boundaries.push(0);
        }

        let mut dst = 0usize;
        for r in 0..n_ranges {
            let first = ranges[r * 2] as usize;
            let last = ranges[r * 2 + 1] as usize;
            if last <= first {
                continue;
            }
            let mut begin = offsets[first] as usize;
            let end = offsets[last] as usize;
            if is_delta {
                let base = offsets[first];
                for p in (first + 1)..=last {
                    boundaries.push(dst + (offsets[p] - base) as usize);
                }
            }
            while begin < end {
                let c_idx = begin / chunk_len;
                let local = begin % chunk_len;
                let len = (end - begin).min(chunk_len - local);
                chunk_map.entry(c_idx).or_default().push((dst, local, len));
                begin += len;
                dst += len;
            }
        }

        let mut chunk_items: Vec<(usize, Vec<(usize, usize, usize)>)> =
            chunk_map.into_iter().collect();
        chunk_items.sort_unstable_by_key(|(c_idx, _)| *c_idx);

        let u_bytes = chunk_len * 4;
        let decode_item =
            |(c_idx, spans): &(usize, Vec<(usize, usize, usize)>)| -> Result<(), i32> {
                let b_off = table[*c_idx * 3] as usize;
                let c_sz = table[*c_idx * 3 + 1] as usize;
                let mask = table[*c_idx * 3 + 2];
                if (mask & !3) != 0 {
                    return Err(-2);
                }
                let out_u32 =
                    unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, total_elems) };

                THREAD_ZSTD.with(|cell| {
                    let mut ctx = cell.borrow_mut();
                    let zctx = &mut *ctx;

                    let raw_slice: &[u8] = if (mask & 2) != 0 {
                        if c_sz != u_bytes || b_off.saturating_add(u_bytes) > mm_u8.len() {
                            return Err(-1);
                        }
                        &mm_u8[b_off..b_off + u_bytes]
                    } else {
                        if b_off.saturating_add(c_sz) > mm_u8.len() {
                            return Err(-1);
                        }
                        zctx.ensure_scratch(u_bytes);
                        match zctx
                            .dctx
                            .decompress(&mut zctx.scratch[..u_bytes], &mm_u8[b_off..b_off + c_sz])
                        {
                            Ok(n) if n == u_bytes => &zctx.scratch[..u_bytes],
                            _ => return Err(-1),
                        }
                    };

                    for &(d_off, l_off, s_len) in spans {
                        if (mask & 1) != 0 {
                            unsafe {
                                copy_unaligned_u32(
                                    raw_slice,
                                    l_off,
                                    &mut out_u32[d_off..d_off + s_len],
                                );
                            }
                        } else {
                            unshuffle_span_u32(
                                raw_slice,
                                chunk_len,
                                l_off,
                                &mut out_u32[d_off..d_off + s_len],
                            );
                        }
                    }
                    Ok(())
                })
            };

        let n_items = chunk_items.len();
        let w = n_workers.max(1).min(n_items.max(1));
        if w <= 1 || n_items <= 1 {
            chunk_items.iter().try_for_each(decode_item)?;
        } else {
            get_pool(w).install(|| chunk_items.par_iter().try_for_each(decode_item))?;
        }

        if is_delta && boundaries.len() > 1 {
            let n_b = boundaries.len() - 1;
            let out_u32 =
                unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, total_elems) };
            if n_workers <= 1 || total_elems < 262_144 {
                for i in 0..n_b {
                    let s = boundaries[i];
                    let e = boundaries[i + 1];
                    if e > s + 1 {
                        cumsum_u32_inplace(&mut out_u32[s..e]);
                    }
                }
            } else {
                let w_d = n_workers.min(n_b);
                let step = (n_b + w_d - 1) / w_d;
                get_pool(w_d).install(|| {
                    (0..w_d).into_par_iter().for_each(|tid| {
                        let b0 = tid * step;
                        let b1 = ((tid + 1) * step).min(n_b);
                        let slice = unsafe {
                            std::slice::from_raw_parts_mut(out_addr as *mut u32, total_elems)
                        };
                        for i in b0..b1 {
                            let s = boundaries[i];
                            let e = boundaries[i + 1];
                            if e > s + 1 {
                                cumsum_u32_inplace(&mut slice[s..e]);
                            }
                        }
                    });
                });
            }
        }
        Ok(())
    });

    match res {
        Ok(()) => Ok(out_arr),
        Err(-2) => Err(PyValueError::new_err("Unexpected HDF5 v2 chunk filter mask")),
        Err(_) => Err(PyValueError::new_err(
            "Unexpected HDF5 v2 decoded chunk length",
        )),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_chunk_span_cached<'py>(
    py: Python<'py>,
    slot: usize,
    file_id: usize,
    chunk_start: i64,
    mask: i64,
    b_payload: PyBuffer<u8>,
    chunk_len: usize,
    sel_start: usize,
    sel_stop: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    if (mask & !3) != 0 {
        return Err(PyValueError::new_err("Unexpected HDF5 v2 chunk filter mask"));
    }
    let payload = view_pybuffer_u8(&b_payload)?;
    let u_bytes = chunk_len * 4;
    let count = sel_stop.saturating_sub(sel_start);
    let out_arr = unsafe { PyArray1::<u32>::new(py, [count], false) };
    let out_slice = unsafe { out_arr.as_slice_mut()? };

    let rc = THREAD_ZSTD.with(|cell| {
        let mut ctx = cell.borrow_mut();
        let zctx = &mut *ctx;
        let slot_idx = slot.min(2);

        if (mask & 2) != 0 {
            if payload.len() != u_bytes {
                return -1;
            }
            if count > 0 {
                if (mask & 1) != 0 {
                    unsafe {
                        copy_unaligned_u32(payload, sel_start, out_slice);
                    }
                } else {
                    unshuffle_span_u32(payload, chunk_len, sel_start, out_slice);
                }
            }
            return 0;
        }

        let key = (file_id, chunk_start, payload.len() as i64, chunk_len);
        let mut cache_entry = get_single_cache()[slot_idx].lock().unwrap();
        if cache_entry.key != key || cache_entry.buf.len() < u_bytes {
            if cache_entry.buf.len() < u_bytes {
                cache_entry.buf.resize(u_bytes, 0);
            }
            match zctx
                .dctx
                .decompress(&mut cache_entry.buf[..u_bytes], payload)
            {
                Ok(n) if n == u_bytes => {
                    cache_entry.key = key;
                }
                _ => {
                    cache_entry.key = (0, -1, -1, 0);
                    return -1;
                }
            }
        }

        if count > 0 {
            let raw = &cache_entry.buf[..u_bytes];
            if (mask & 1) != 0 {
                unsafe {
                    copy_unaligned_u32(raw, sel_start, out_slice);
                }
            } else {
                unshuffle_span_u32(raw, chunk_len, sel_start, out_slice);
            }
        }
        0
    });

    if rc != 0 {
        return Err(PyValueError::new_err(
            "Unexpected HDF5 v2 decoded chunk length",
        ));
    }
    Ok(out_arr)
}

#[allow(clippy::too_many_arguments)]
fn decode_single_col_from_mmap(
    slot_idx: usize,
    file_id: usize,
    mm_u8: &[u8],
    table: &[i64],
    chunk_len: usize,
    mut begin: usize,
    end: usize,
    out_slice: &mut [u32],
    is_delta: bool,
) -> i32 {
    if begin >= end {
        return 0;
    }
    let u_bytes = chunk_len * 4;
    let mut dst = 0usize;
    let total_len = out_slice.len();
    let single_chunk = total_len > 0 && (begin / chunk_len) == ((end - 1) / chunk_len);
    let mut fused_cumsum_done = false;
    let rc = THREAD_ZSTD.with(|cell| {
        let mut ctx = cell.borrow_mut();
        let zctx = &mut *ctx;
        while begin < end {
            let c_idx = begin / chunk_len;
            let start = begin % chunk_len;
            let len = (end - begin).min(chunk_len - start);
            let b_off = table[c_idx * 3] as usize;
            let c_sz = table[c_idx * 3 + 1] as usize;
            let mask = table[c_idx * 3 + 2];
            if (mask & !3) != 0 {
                return -2;
            }
            let do_fused = is_delta && single_chunk && (mask & 1) == 0;
            if (mask & 2) != 0 {
                if c_sz != u_bytes || b_off.saturating_add(u_bytes) > mm_u8.len() {
                    return -1;
                }
                let payload = &mm_u8[b_off..b_off + u_bytes];
                if (mask & 1) != 0 {
                    unsafe {
                        copy_unaligned_u32(payload, start, &mut out_slice[dst..dst + len]);
                    }
                } else {
                    unshuffle_span_u32_opt_cumsum(
                        payload,
                        chunk_len,
                        start,
                        &mut out_slice[dst..dst + len],
                        do_fused,
                    );
                    if do_fused {
                        fused_cumsum_done = true;
                    }
                }
            } else {
                if b_off.saturating_add(c_sz) > mm_u8.len() {
                    return -1;
                }
                let key = (file_id, (c_idx * chunk_len) as i64, c_sz as i64, chunk_len);
                let mut cache_entry = get_single_cache()[slot_idx].lock().unwrap();
                if cache_entry.key != key || cache_entry.buf.len() < u_bytes {
                    if cache_entry.buf.len() < u_bytes {
                        cache_entry.buf.resize(u_bytes, 0);
                    }
                    match zctx
                        .dctx
                        .decompress(&mut cache_entry.buf[..u_bytes], &mm_u8[b_off..b_off + c_sz])
                    {
                        Ok(n) if n == u_bytes => {
                            cache_entry.key = key;
                        }
                        _ => {
                            cache_entry.key = (0, -1, -1, 0);
                            return -1;
                        }
                    }
                }
                let raw = &cache_entry.buf[..u_bytes];
                if (mask & 1) != 0 {
                    unsafe {
                        copy_unaligned_u32(raw, start, &mut out_slice[dst..dst + len]);
                    }
                } else {
                    unshuffle_span_u32_opt_cumsum(
                        raw,
                        chunk_len,
                        start,
                        &mut out_slice[dst..dst + len],
                        do_fused,
                    );
                    if do_fused {
                        fused_cumsum_done = true;
                    }
                }
            }
            begin += len;
            dst += len;
        }
        0
    });
    if rc == 0 && is_delta && !fused_cumsum_done && out_slice.len() > 1 {
        cumsum_u32_inplace(out_slice);
    }
    rc
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_single_frame_mmap<'py>(
    py: Python<'py>,
    file_id: usize,
    b_mm: PyBuffer<u8>,
    b_table_po: PyReadonlyArray2<'py, i64>,
    chunk_len_po: usize,
    begin_o: usize,
    end_o: usize,
    b_table_pm: PyReadonlyArray2<'py, i64>,
    chunk_len_pm: usize,
    begin_m: usize,
    end_m: usize,
    b_table_cm: PyReadonlyArray2<'py, i64>,
    chunk_len_cm: usize,
    is_delta: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let t_po = b_table_po.as_slice()?;
    let t_pm = b_table_pm.as_slice()?;
    let t_cm = b_table_cm.as_slice()?;

    let len_o = end_o.saturating_sub(begin_o);
    let len_m = end_m.saturating_sub(begin_m);

    let arr_po = unsafe { PyArray1::<u32>::new(py, [len_o], false) };
    let arr_pm = unsafe { PyArray1::<u32>::new(py, [len_m], false) };
    let arr_cm = unsafe { PyArray1::<u32>::new(py, [len_m], false) };

    let s_po = unsafe { arr_po.as_slice_mut()? };
    let s_pm = unsafe { arr_pm.as_slice_mut()? };
    let s_cm = unsafe { arr_cm.as_slice_mut()? };

    let po_addr = s_po.as_mut_ptr() as usize;
    let pm_addr = s_pm.as_mut_ptr() as usize;
    let cm_addr = s_cm.as_mut_ptr() as usize;

    let rc = py.detach(|| {
        let large_chunks = chunk_len_po >= 262_144 && len_o > 0 && len_m > 0;
        if large_chunks {
            let sc = get_single_cache();
            let c0 = begin_o / chunk_len_po;
            let k0 = (
                file_id,
                (c0 * chunk_len_po) as i64,
                t_po[c0 * 3 + 1],
                chunk_len_po,
            );
            let c1 = begin_m / chunk_len_pm;
            let k1 = (
                file_id,
                (c1 * chunk_len_pm) as i64,
                t_pm[c1 * 3 + 1],
                chunk_len_pm,
            );
            let c2 = begin_m / chunk_len_cm;
            let k2 = (
                file_id,
                (c2 * chunk_len_cm) as i64,
                t_cm[c2 * 3 + 1],
                chunk_len_cm,
            );
            let misses = (sc[0].lock().unwrap().key != k0) as usize
                + (sc[1].lock().unwrap().key != k1) as usize
                + (sc[2].lock().unwrap().key != k2) as usize;
            if misses >= 2 {
                let err = AtomicI32::new(0);
                get_pool(3).install(|| {
                    rayon::scope(|s| {
                        s.spawn(|_| {
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(po_addr as *mut u32, len_o)
                            };
                            let r = decode_single_col_from_mmap(
                                0,
                                file_id,
                                mm_u8,
                                t_po,
                                chunk_len_po,
                                begin_o,
                                end_o,
                                out,
                                is_delta,
                            );
                            if r != 0 {
                                err.store(r, Ordering::Relaxed);
                            }
                        });
                        s.spawn(|_| {
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(pm_addr as *mut u32, len_m)
                            };
                            let r = decode_single_col_from_mmap(
                                1,
                                file_id,
                                mm_u8,
                                t_pm,
                                chunk_len_pm,
                                begin_m,
                                end_m,
                                out,
                                is_delta,
                            );
                            if r != 0 {
                                err.store(r, Ordering::Relaxed);
                            }
                        });
                        s.spawn(|_| {
                            let out = unsafe {
                                std::slice::from_raw_parts_mut(cm_addr as *mut u32, len_m)
                            };
                            let r = decode_single_col_from_mmap(
                                2,
                                file_id,
                                mm_u8,
                                t_cm,
                                chunk_len_cm,
                                begin_m,
                                end_m,
                                out,
                                false,
                            );
                            if r != 0 {
                                err.store(r, Ordering::Relaxed);
                            }
                        });
                    });
                });
                return err.load(Ordering::Relaxed);
            }
        }

        let out_po = unsafe { std::slice::from_raw_parts_mut(po_addr as *mut u32, len_o) };
        let r0 = decode_single_col_from_mmap(
            0,
            file_id,
            mm_u8,
            t_po,
            chunk_len_po,
            begin_o,
            end_o,
            out_po,
            is_delta,
        );
        if r0 != 0 {
            return r0;
        }
        let out_pm = unsafe { std::slice::from_raw_parts_mut(pm_addr as *mut u32, len_m) };
        let r1 = decode_single_col_from_mmap(
            1,
            file_id,
            mm_u8,
            t_pm,
            chunk_len_pm,
            begin_m,
            end_m,
            out_pm,
            is_delta,
        );
        if r1 != 0 {
            return r1;
        }
        let out_cm = unsafe { std::slice::from_raw_parts_mut(cm_addr as *mut u32, len_m) };
        decode_single_col_from_mmap(
            2,
            file_id,
            mm_u8,
            t_cm,
            chunk_len_cm,
            begin_m,
            end_m,
            out_cm,
            false,
        )
    });

    match rc {
        0 => Ok((arr_po, arr_pm, arr_cm)),
        -2 => Err(PyValueError::new_err("Unexpected HDF5 v2 chunk filter mask")),
        _ => Err(PyValueError::new_err(
            "Unexpected HDF5 v2 decoded chunk length",
        )),
    }
}

#[pyfunction]
pub fn cumsum_inplace_u32<'py>(mut b_arr: PyReadwriteArray1<'py, u32>) -> PyResult<()> {
    let slice = b_arr.as_slice_mut()?;
    cumsum_u32_inplace(slice);
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_raw_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(decode_raw_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(compress_chunk, m)?)?;
    m.add_function(wrap_pyfunction!(compress_bound_u32, m)?)?;
    m.add_function(wrap_pyfunction!(encode_chunks_batch, m)?)?;
    m.add_function(wrap_pyfunction!(encode_chunks_into, m)?)?;
    m.add_function(wrap_pyfunction!(decode_dataset_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(decode_indexed_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(decode_chunk_span_cached, m)?)?;
    m.add_function(wrap_pyfunction!(decode_single_frame_mmap, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_inplace_u32, m)?)?;
    Ok(())
}
