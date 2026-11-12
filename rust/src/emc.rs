use std::sync::atomic::{AtomicBool, Ordering};

use numpy::{
    Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{get_pool, view_pybuffer_u8_detachable, RawFileTarget};

const HEADER_BYTES: usize = 1024;
const STREAM_WRITE_BUF_CAP: usize = 8 << 20;

type EmcCountsOffsets<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
);

type SparseEvents<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<i32>>,
);

type FullPatternArrays<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<i32>>,
);

type SizedFullPatternArrays<'py> = (usize, FullPatternArrays<'py>);
type CooEvents = (Vec<u32>, Vec<u32>, Vec<i32>);

#[derive(Clone, Copy)]
struct EmcFileLayout {
    po_byte_off: usize,
    pm_byte_off: usize,
    cm_byte_off: usize,
    total_ones: usize,
    total_multi: usize,
}

#[inline]
fn validate_emc_buffer(
    buf_len: usize,
    num_data: usize,
    total_ones: usize,
    total_multi: usize,
) -> PyResult<EmcFileLayout> {
    let counts_bytes = num_data
        .checked_mul(8)
        .ok_or_else(|| PyValueError::new_err("EMC count byte size overflows usize"))?;
    let po_byte_off = HEADER_BYTES
        .checked_add(counts_bytes)
        .ok_or_else(|| PyValueError::new_err("EMC offset overflows usize"))?;
    let po_bytes = total_ones
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("EMC place_ones byte size overflows usize"))?;
    let pm_byte_off = po_byte_off
        .checked_add(po_bytes)
        .ok_or_else(|| PyValueError::new_err("EMC place_multi offset overflows usize"))?;
    let pm_bytes = total_multi
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("EMC place_multi byte size overflows usize"))?;
    let cm_byte_off = pm_byte_off
        .checked_add(pm_bytes)
        .ok_or_else(|| PyValueError::new_err("EMC count_multi offset overflows usize"))?;
    let expected_end = cm_byte_off
        .checked_add(pm_bytes)
        .ok_or_else(|| PyValueError::new_err("EMC file byte size overflows usize"))?;

    if buf_len < expected_end {
        return Err(PyValueError::new_err("Truncated EMC pattern file"));
    }
    if buf_len > expected_end {
        return Err(PyValueError::new_err("Error when parsing"));
    }

    Ok(EmcFileLayout {
        po_byte_off,
        pm_byte_off,
        cm_byte_off,
        total_ones,
        total_multi,
    })
}

#[inline]
fn check_offsets_bounds(offsets: &[u64], num_data: usize) -> PyResult<usize> {
    if offsets.len() != num_data + 1 || offsets[0] != 0 {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    let last = offsets[num_data];
    usize::try_from(last).map_err(|_| PyValueError::new_err("Pattern offset overflows usize"))
}

#[inline]
fn copy_bytes_to_u32(src_u8: &[u8], byte_off: usize, elem_start: usize, dst: &mut [u32]) {
    let count = dst.len();
    if count > 0 {
        let b_start = byte_off + elem_start * 4;
        let b_len = count * 4;
        assert!(b_start + b_len <= src_u8.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_u8.as_ptr().add(b_start),
                dst.as_mut_ptr() as *mut u8,
                b_len,
            );
        }
    }
}

#[inline]
fn copy_bytes_to_i32(src_u8: &[u8], byte_off: usize, elem_start: usize, dst: &mut [i32]) {
    let count = dst.len();
    if count > 0 {
        let b_start = byte_off + elem_start * 4;
        let b_len = count * 4;
        assert!(b_start + b_len <= src_u8.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_u8.as_ptr().add(b_start),
                dst.as_mut_ptr() as *mut u8,
                b_len,
            );
        }
    }
}

#[inline]
fn strided_len(start: isize, stop: isize, step: isize) -> usize {
    if step > 0 {
        if start >= stop {
            0
        } else {
            ((stop - start - 1) as usize) / (step as usize) + 1
        }
    } else if step < 0 {
        if start <= stop {
            0
        } else {
            ((start - stop - 1) as usize) / ((-step) as usize) + 1
        }
    } else {
        0
    }
}

#[inline]
fn normalize_row_id(raw: i64, num_data: usize) -> Result<usize, ()> {
    let n_i128 = num_data as i128;
    let mut v = raw as i128;
    if v < 0 {
        v += n_i128;
    }
    if v < 0 || v >= n_i128 {
        Err(())
    } else {
        Ok(v as usize)
    }
}

// ============================================================================
// 1. Binary .emc buffer / mmap readers & writer
// ============================================================================

#[pyfunction]
pub fn emc_read_counts_and_offsets<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
) -> PyResult<EmcCountsOffsets<'py>> {
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    if buf.len() < HEADER_BYTES {
        return Err(PyValueError::new_err("Truncated EMC header"));
    }
    let counts_bytes = num_data
        .checked_mul(8)
        .ok_or_else(|| PyValueError::new_err("EMC count byte size overflows usize"))?;
    let need = HEADER_BYTES
        .checked_add(counts_bytes)
        .ok_or_else(|| PyValueError::new_err("EMC count byte size overflows usize"))?;
    if buf.len() < need {
        return Err(PyValueError::new_err("Truncated EMC pattern count arrays"));
    }

    let arr_ones = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_ones_idx = PyArray1::<u64>::zeros(py, [num_data + 1], false);
    let arr_multi_idx = PyArray1::<u64>::zeros(py, [num_data + 1], false);

    let ones_addr = unsafe { arr_ones.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { arr_multi.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_addr = unsafe { arr_ones_idx.as_slice_mut()? }.as_mut_ptr() as usize;
    let midx_addr = unsafe { arr_multi_idx.as_slice_mut()? }.as_mut_ptr() as usize;

    py.detach(|| {
        let ones = unsafe { std::slice::from_raw_parts_mut(ones_addr as *mut u32, num_data) };
        let multi = unsafe { std::slice::from_raw_parts_mut(multi_addr as *mut u32, num_data) };
        let oidx = unsafe { std::slice::from_raw_parts_mut(oidx_addr as *mut u64, num_data + 1) };
        let midx = unsafe { std::slice::from_raw_parts_mut(midx_addr as *mut u64, num_data + 1) };

        copy_bytes_to_u32(buf, HEADER_BYTES, 0, ones);
        copy_bytes_to_u32(buf, HEADER_BYTES + num_data * 4, 0, multi);

        oidx[0] = 0;
        midx[0] = 0;
        let mut acc_o = 0u64;
        let mut acc_m = 0u64;
        for i in 0..num_data {
            acc_o += ones[i] as u64;
            acc_m += multi[i] as u64;
            oidx[i + 1] = acc_o;
            midx[i + 1] = acc_m;
        }
    });

    Ok((arr_ones, arr_multi, arr_ones_idx, arr_multi_idx))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_read_dense_frame<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    row_idx: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if row_idx >= num_data {
        return Err(PyIndexError::new_err("Pattern index out of range"));
    }
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let o0 = oidx[row_idx] as usize;
    let o1 = oidx[row_idx + 1] as usize;
    let m0 = midx[row_idx] as usize;
    let m1 = midx[row_idx + 1] as usize;
    if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    let out_arr = PyArray1::<i32>::zeros(py, [num_pix], false);
    let out = unsafe { out_arr.as_slice_mut()? };

    let po_bytes = &buf[layout.po_byte_off + o0 * 4..layout.po_byte_off + o1 * 4];
    for &chunk in po_bytes.as_chunks::<4>().0 {
        let idx = u32::from_le_bytes(chunk) as usize;
        if idx >= num_pix {
            return Err(PyIndexError::new_err(
                "place_ones index out of bounds for num_pix",
            ));
        }
        out[idx] = 1;
    }

    let pm_bytes = &buf[layout.pm_byte_off + m0 * 4..layout.pm_byte_off + m1 * 4];
    let cm_bytes = &buf[layout.cm_byte_off + m0 * 4..layout.cm_byte_off + m1 * 4];
    for (&p_chk, &c_chk) in pm_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .zip(cm_bytes.as_chunks::<4>().0)
    {
        let idx = u32::from_le_bytes(p_chk) as usize;
        let val = i32::from_le_bytes(c_chk);
        if idx >= num_pix {
            return Err(PyIndexError::new_err(
                "place_multi index out of bounds for num_pix",
            ));
        }
        out[idx] = val;
    }

    Ok(out_arr)
}

#[pyfunction]
pub fn emc_read_sparse_frame<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    row_idx: usize,
) -> PyResult<SparseEvents<'py>> {
    if row_idx >= num_data {
        return Err(PyIndexError::new_err("Pattern index out of range"));
    }
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let o0 = oidx[row_idx] as usize;
    let o1 = oidx[row_idx + 1] as usize;
    let m0 = midx[row_idx] as usize;
    let m1 = midx[row_idx + 1] as usize;
    if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    let len_o = o1 - o0;
    let len_m = m1 - m0;
    let arr_po = PyArray1::<u32>::zeros(py, [len_o], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [len_m], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [len_m], false);

    copy_bytes_to_u32(buf, layout.po_byte_off, o0, unsafe {
        arr_po.as_slice_mut()?
    });
    copy_bytes_to_u32(buf, layout.pm_byte_off, m0, unsafe {
        arr_pm.as_slice_mut()?
    });
    copy_bytes_to_i32(buf, layout.cm_byte_off, m0, unsafe {
        arr_cm.as_slice_mut()?
    });

    Ok((arr_po, arr_pm, arr_cm))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_read_slice<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    start: usize,
    stop: usize,
    n_workers: usize,
) -> PyResult<SparseEvents<'py>> {
    let s = start.min(num_data);
    let e = stop.min(num_data).max(s);
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let o0 = oidx[s] as usize;
    let o1 = oidx[e] as usize;
    let m0 = midx[s] as usize;
    let m1 = midx[e] as usize;
    if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    let len_o = o1 - o0;
    let len_m = m1 - m0;
    let arr_po = PyArray1::<u32>::zeros(py, [len_o], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [len_m], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [len_m], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;

    py.detach(|| {
        let total_elems = len_o + len_m * 2;
        if n_workers > 1 && total_elems >= (4 << 20) {
            let w = n_workers.min(8);
            get_pool(w).install(|| {
                let po_dst =
                    unsafe { std::slice::from_raw_parts_mut(po_addr as *mut u8, len_o * 4) };
                let pm_dst =
                    unsafe { std::slice::from_raw_parts_mut(pm_addr as *mut u8, len_m * 4) };
                let cm_dst =
                    unsafe { std::slice::from_raw_parts_mut(cm_addr as *mut u8, len_m * 4) };
                let chunk_bytes = 2 << 20;
                rayon::scope(|sc| {
                    sc.spawn(|_| {
                        let src = &buf[layout.po_byte_off + o0 * 4..layout.po_byte_off + o1 * 4];
                        po_dst
                            .par_chunks_mut(chunk_bytes)
                            .enumerate()
                            .for_each(|(ci, dst_c)| {
                                let off = ci * chunk_bytes;
                                dst_c.copy_from_slice(&src[off..off + dst_c.len()]);
                            });
                    });
                    sc.spawn(|_| {
                        let src = &buf[layout.pm_byte_off + m0 * 4..layout.pm_byte_off + m1 * 4];
                        pm_dst
                            .par_chunks_mut(chunk_bytes)
                            .enumerate()
                            .for_each(|(ci, dst_c)| {
                                let off = ci * chunk_bytes;
                                dst_c.copy_from_slice(&src[off..off + dst_c.len()]);
                            });
                    });
                    sc.spawn(|_| {
                        let src = &buf[layout.cm_byte_off + m0 * 4..layout.cm_byte_off + m1 * 4];
                        cm_dst
                            .par_chunks_mut(chunk_bytes)
                            .enumerate()
                            .for_each(|(ci, dst_c)| {
                                let off = ci * chunk_bytes;
                                dst_c.copy_from_slice(&src[off..off + dst_c.len()]);
                            });
                    });
                });
            });
        } else {
            let po = unsafe { std::slice::from_raw_parts_mut(po_addr as *mut u32, len_o) };
            let pm = unsafe { std::slice::from_raw_parts_mut(pm_addr as *mut u32, len_m) };
            let cm = unsafe { std::slice::from_raw_parts_mut(cm_addr as *mut i32, len_m) };
            copy_bytes_to_u32(buf, layout.po_byte_off, o0, po);
            copy_bytes_to_u32(buf, layout.pm_byte_off, m0, pm);
            copy_bytes_to_i32(buf, layout.cm_byte_off, m0, cm);
        }
    });

    Ok((arr_po, arr_pm, arr_cm))
}

#[allow(clippy::too_many_arguments)]
fn emc_gather_from_resolved_ids<'py>(
    py: Python<'py>,
    buf: &[u8],
    layout: EmcFileLayout,
    ones: &[u32],
    multi: &[u32],
    oidx: &[u64],
    midx: &[u64],
    ids: &[usize],
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    let k = ids.len();
    let arr_ones = PyArray1::<u32>::zeros(py, [k], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [k], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [k + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [k + 1], false);

    let s_ones = unsafe { arr_ones.as_slice_mut()? };
    let s_multi = unsafe { arr_multi.as_slice_mut()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };

    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut acc_o = 0u64;
    let mut acc_m = 0u64;
    for (i, &r) in ids.iter().enumerate() {
        let co = ones[r];
        let cm = multi[r];
        s_ones[i] = co;
        s_multi[i] = cm;
        acc_o += co as u64;
        acc_m += cm as u64;
        s_oidx[i + 1] = acc_o;
        s_midx[i + 1] = acc_m;
    }

    let out_ones_len = usize::try_from(acc_o)
        .map_err(|_| PyValueError::new_err("Selected place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(acc_m)
        .map_err(|_| PyValueError::new_err("Selected place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_out_ref: &[u64] = s_oidx;
    let midx_out_ref: &[u64] = s_midx;

    let bad_offsets = AtomicBool::new(false);
    py.detach(|| {
        let copy_row = |i: usize| {
            let r = ids[i];
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > layout.total_ones || m1 < m0 || m1 > layout.total_multi {
                bad_offsets.store(true, Ordering::Relaxed);
                return;
            }
            let do0 = oidx_out_ref[i] as usize;
            let dm0 = midx_out_ref[i] as usize;
            let lo = o1 - o0;
            let lm = m1 - m0;
            if lo > 0 && lo == (oidx_out_ref[i + 1] - oidx_out_ref[i]) as usize {
                let dst =
                    unsafe { std::slice::from_raw_parts_mut((po_addr as *mut u32).add(do0), lo) };
                copy_bytes_to_u32(buf, layout.po_byte_off, o0, dst);
            } else if lo != (oidx_out_ref[i + 1] - oidx_out_ref[i]) as usize {
                bad_offsets.store(true, Ordering::Relaxed);
                return;
            }
            if lm > 0 && lm == (midx_out_ref[i + 1] - midx_out_ref[i]) as usize {
                let dst_pm =
                    unsafe { std::slice::from_raw_parts_mut((pm_addr as *mut u32).add(dm0), lm) };
                let dst_cm =
                    unsafe { std::slice::from_raw_parts_mut((cm_addr as *mut i32).add(dm0), lm) };
                copy_bytes_to_u32(buf, layout.pm_byte_off, m0, dst_pm);
                copy_bytes_to_i32(buf, layout.cm_byte_off, m0, dst_cm);
            } else if lm != (midx_out_ref[i + 1] - midx_out_ref[i]) as usize {
                bad_offsets.store(true, Ordering::Relaxed);
            }
        };

        let total_work = out_ones_len + out_multi_len * 2 + k;
        if n_workers > 1 && k >= 256 && total_work >= 262_144 {
            let w = n_workers.min((k / 64).max(1));
            get_pool(w).install(|| {
                (0..k).into_par_iter().for_each(copy_row);
            });
        } else {
            for i in 0..k {
                copy_row(i);
            }
        }
    });

    if bad_offsets.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Pattern counts do not match offsets"));
    }

    Ok((
        arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_read_strided<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    ones: PyReadonlyArray1<'py, u32>,
    multi: PyReadonlyArray1<'py, u32>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    start: isize,
    stop: isize,
    step: isize,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let s_ones = ones.as_slice()?;
    let s_multi = multi.as_slice()?;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    if s_ones.len() != num_data || s_multi.len() != num_data {
        return Err(PyValueError::new_err("Invalid pattern counts length"));
    }
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let k = strided_len(start, stop, step);
    let mut ids = Vec::with_capacity(k);
    let mut cur = start;
    for _ in 0..k {
        if cur < 0 || (cur as usize) >= num_data {
            return Err(PyIndexError::new_err("row index out of range"));
        }
        ids.push(cur as usize);
        cur += step;
    }

    emc_gather_from_resolved_ids(
        py, buf, layout, s_ones, s_multi, oidx, midx, &ids, n_workers,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_read_indexed<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    ones: PyReadonlyArray1<'py, u32>,
    multi: PyReadonlyArray1<'py, u32>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    row_ids: PyReadonlyArray1<'py, i64>,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let s_ones = ones.as_slice()?;
    let s_multi = multi.as_slice()?;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let raw_ids = row_ids.as_slice()?;
    if s_ones.len() != num_data || s_multi.len() != num_data {
        return Err(PyValueError::new_err("Invalid pattern counts length"));
    }
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let mut ids = Vec::with_capacity(raw_ids.len());
    for &raw in raw_ids {
        let r = normalize_row_id(raw, num_data)
            .map_err(|_| PyIndexError::new_err("row index out of range"))?;
        ids.push(r);
    }

    emc_gather_from_resolved_ids(
        py, buf, layout, s_ones, s_multi, oidx, midx, &ids, n_workers,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_read_mask<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    ones: PyReadonlyArray1<'py, u32>,
    multi: PyReadonlyArray1<'py, u32>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    mask: PyReadonlyArray1<'py, bool>,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    let s_mask = mask.as_slice()?;
    if s_mask.len() != num_data {
        return Err(PyIndexError::new_err(
            "Boolean row mask must match the number of patterns",
        ));
    }
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let s_ones = ones.as_slice()?;
    let s_multi = multi.as_slice()?;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    if s_ones.len() != num_data || s_multi.len() != num_data {
        return Err(PyValueError::new_err("Invalid pattern counts length"));
    }
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let mut ids = Vec::new();
    for (i, &keep) in s_mask.iter().enumerate() {
        if keep {
            ids.push(i);
        }
    }

    emc_gather_from_resolved_ids(
        py, buf, layout, s_ones, s_multi, oidx, midx, &ids, n_workers,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_extract_column<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    row_ids: Option<PyReadonlyArray1<'py, i64>>,
    col: usize,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if col >= num_pix {
        return Err(PyIndexError::new_err("column index out of bounds"));
    }
    let Ok(target_col) = u32::try_from(col) else {
        let out_len = row_ids.as_ref().map_or(num_data, |r| r.len());
        return Ok(PyArray1::<i32>::zeros(py, [out_len], false));
    };

    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let resolved_ids: Option<Vec<usize>> = if let Some(ref r_arr) = row_ids {
        let raw = r_arr.as_slice()?;
        let mut v = Vec::with_capacity(raw.len());
        for &id in raw {
            let norm = normalize_row_id(id, num_data)
                .map_err(|_| PyIndexError::new_err("row index out of range"))?;
            v.push(norm);
        }
        Some(v)
    } else {
        None
    };

    let out_len = resolved_ids.as_ref().map_or(num_data, Vec::len);
    let out_arr = PyArray1::<i32>::zeros(py, [out_len], false);
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;

    let bad = AtomicBool::new(false);
    py.detach(|| {
        let eval_row = |r: usize| -> i32 {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return 0;
            }
            let mut acc: i32 = 0;
            let po_b = &buf[layout.po_byte_off + o0 * 4..layout.po_byte_off + o1 * 4];
            for &chk in po_b.as_chunks::<4>().0 {
                if u32::from_le_bytes(chk) == target_col {
                    acc = acc.wrapping_add(1);
                }
            }
            let pm_b = &buf[layout.pm_byte_off + m0 * 4..layout.pm_byte_off + m1 * 4];
            let cm_b = &buf[layout.cm_byte_off + m0 * 4..layout.cm_byte_off + m1 * 4];
            for (&pc, &cc) in pm_b.as_chunks::<4>().0.iter().zip(cm_b.as_chunks::<4>().0) {
                if u32::from_le_bytes(pc) == target_col {
                    let val = i32::from_le_bytes(cc);
                    acc = acc.wrapping_add(val);
                }
            }
            acc
        };

        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i32, out_len) };
        if n_workers > 1 && out_len >= 4096 {
            let w = n_workers.min((out_len / 1024).max(1));
            get_pool(w).install(|| {
                if let Some(ref ids) = resolved_ids {
                    out.par_iter_mut()
                        .enumerate()
                        .for_each(|(i, slot)| *slot = eval_row(ids[i]));
                } else {
                    out.par_iter_mut()
                        .enumerate()
                        .for_each(|(r, slot)| *slot = eval_row(r));
                }
            });
        } else if let Some(ref ids) = resolved_ids {
            for (i, &r) in ids.iter().enumerate() {
                out[i] = eval_row(r);
            }
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = eval_row(r);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    Ok(out_arr)
}

#[pyfunction]
pub fn emc_sum_total(
    py: Python<'_>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    total_ones: usize,
    total_multi: usize,
    n_workers: usize,
) -> PyResult<i128> {
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;
    let cm_bytes = &buf[layout.cm_byte_off..layout.cm_byte_off + total_multi * 4];

    let multi_sum: i128 = py.detach(|| {
        if n_workers > 1 && total_multi >= 524_288 {
            let w = n_workers.min(8);
            let chunk_bytes = ((total_multi / w).max(65_536)) * 4;
            get_pool(w).install(|| {
                cm_bytes
                    .par_chunks(chunk_bytes)
                    .map(|part| {
                        let mut s: i64 = 0;
                        for &chk in part.as_chunks::<4>().0 {
                            s += i32::from_le_bytes(chk) as i64;
                        }
                        s as i128
                    })
                    .sum()
            })
        } else {
            cm_bytes
                .chunks(1 << 26)
                .map(|part| {
                    let mut s: i64 = 0;
                    for &chk in part.as_chunks::<4>().0 {
                        s += i32::from_le_bytes(chk) as i64;
                    }
                    s as i128
                })
                .sum()
        }
    });

    Ok((total_ones as i128) + multi_sum)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_sum_axis1<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    total_ones: usize,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let s_ones = ones.as_slice()?;
    let midx = multi_idx.as_slice()?;
    if s_ones.len() != num_data {
        return Err(PyValueError::new_err("Invalid ones count array length"));
    }
    let total_multi = check_offsets_bounds(midx, num_data)?;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let out_arr = PyArray1::<i64>::zeros(py, [num_data], false);
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i64, num_data) };
        if total_multi == 0 {
            for r in 0..num_data {
                out[r] = s_ones[r] as i64;
            }
            return;
        }
        let sum_row = |r: usize| -> i64 {
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return 0;
            }
            let mut acc = s_ones[r] as i64;
            let cm_b = &buf[layout.cm_byte_off + m0 * 4..layout.cm_byte_off + m1 * 4];
            for &chk in cm_b.as_chunks::<4>().0 {
                acc += i32::from_le_bytes(chk) as i64;
            }
            acc
        };

        if n_workers > 1 && (num_data >= 32_768 || total_multi >= 524_288) && num_data >= 64 {
            let w = n_workers.min((num_data / 64).max(1));
            get_pool(w).install(|| {
                out.par_iter_mut()
                    .enumerate()
                    .for_each(|(r, slot)| *slot = sum_row(r));
            });
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = sum_row(r);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    Ok(out_arr)
}

#[pyfunction]
pub fn emc_sum_axis0<'py>(
    py: Python<'py>,
    b_buf: PyBuffer<u8>,
    num_data: usize,
    num_pix: usize,
    total_ones: usize,
    total_multi: usize,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let buf_cow = view_pybuffer_u8_detachable(py, &b_buf)?;
    let buf: &[u8] = &buf_cow;
    let layout = validate_emc_buffer(buf.len(), num_data, total_ones, total_multi)?;

    let out_arr = PyArray1::<i64>::zeros(py, [num_pix], false);
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let oob = AtomicBool::new(false);

    py.detach(|| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i64, num_pix) };
        let po_bytes = &buf[layout.po_byte_off..layout.po_byte_off + total_ones * 4];
        let pm_bytes = &buf[layout.pm_byte_off..layout.pm_byte_off + total_multi * 4];
        let cm_bytes = &buf[layout.cm_byte_off..layout.cm_byte_off + total_multi * 4];

        let total_events = total_ones + total_multi;
        let use_par =
            n_workers > 1 && total_events >= 1_048_576 && num_pix > 0 && num_pix <= 2_097_152;

        if use_par {
            let w = n_workers.min(8);
            let partials: Vec<Vec<i64>> = get_pool(w).install(|| {
                (0..w)
                    .into_par_iter()
                    .map(|tid| {
                        let mut local = vec![0i64; num_pix];
                        let o_beg = total_ones * tid / w;
                        let o_end = total_ones * (tid + 1) / w;
                        for &chk in po_bytes[o_beg * 4..o_end * 4].as_chunks::<4>().0 {
                            let idx = u32::from_le_bytes(chk) as usize;
                            if idx >= num_pix {
                                oob.store(true, Ordering::Relaxed);
                                return local;
                            }
                            local[idx] += 1;
                        }
                        let m_beg = total_multi * tid / w;
                        let m_end = total_multi * (tid + 1) / w;
                        for (&pc, &cc) in pm_bytes[m_beg * 4..m_end * 4]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .zip(cm_bytes[m_beg * 4..m_end * 4].as_chunks::<4>().0)
                        {
                            let idx = u32::from_le_bytes(pc) as usize;
                            let val = i32::from_le_bytes(cc) as i64;
                            if idx >= num_pix {
                                oob.store(true, Ordering::Relaxed);
                                return local;
                            }
                            local[idx] += val;
                        }
                        local
                    })
                    .collect()
            });
            if !oob.load(Ordering::Relaxed) {
                for p in &partials {
                    for (dst, &src) in out.iter_mut().zip(p.iter()) {
                        *dst += src;
                    }
                }
            }
        } else {
            for &chk in po_bytes.as_chunks::<4>().0 {
                let idx = u32::from_le_bytes(chk) as usize;
                if idx >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                out[idx] += 1;
            }
            for (&pc, &cc) in pm_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .zip(cm_bytes.as_chunks::<4>().0)
            {
                let idx = u32::from_le_bytes(pc) as usize;
                let val = i32::from_le_bytes(cc) as i64;
                if idx >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                out[idx] += val;
            }
        }
    });

    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err(
            "pixel index out of bounds for num_pix",
        ));
    }

    Ok(out_arr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn emc_write_binary<'py>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    num_data: i32,
    num_pix: i32,
    ones_list: Vec<PyReadonlyArray1<'py, u32>>,
    multi_list: Vec<PyReadonlyArray1<'py, u32>>,
    place_ones_list: Vec<PyReadonlyArray1<'py, u32>>,
    place_multi_list: Vec<PyReadonlyArray1<'py, u32>>,
    count_multi_list: Vec<PyReadonlyArray1<'py, i32>>,
) -> PyResult<()> {
    let raw_target = RawFileTarget::from_py(py, target)?;

    let mut segments: Vec<(usize, usize)> = Vec::with_capacity(
        ones_list.len()
            + multi_list.len()
            + place_ones_list.len()
            + place_multi_list.len()
            + count_multi_list.len(),
    );
    let mut total_bytes: usize = HEADER_BYTES;

    let mut push_u32 = |arr: &PyReadonlyArray1<'py, u32>| -> PyResult<()> {
        let s = arr.as_slice()?;
        let b_len = s
            .len()
            .checked_mul(4)
            .ok_or_else(|| PyValueError::new_err("array byte length overflow"))?;
        total_bytes = total_bytes
            .checked_add(b_len)
            .ok_or_else(|| PyValueError::new_err("total EMC file byte size overflow"))?;
        if b_len > 0 {
            segments.push((s.as_ptr() as usize, b_len));
        }
        Ok(())
    };

    for a in &ones_list {
        push_u32(a)?;
    }
    for a in &multi_list {
        push_u32(a)?;
    }
    for a in &place_ones_list {
        push_u32(a)?;
    }
    for a in &place_multi_list {
        push_u32(a)?;
    }
    for a in &count_multi_list {
        let s = a.as_slice()?;
        let b_len = s
            .len()
            .checked_mul(4)
            .ok_or_else(|| PyValueError::new_err("array byte length overflow"))?;
        total_bytes = total_bytes
            .checked_add(b_len)
            .ok_or_else(|| PyValueError::new_err("total EMC file byte size overflow"))?;
        if b_len > 0 {
            segments.push((s.as_ptr() as usize, b_len));
        }
    }

    let ok = py.detach(|| {
        let cap = total_bytes.min(STREAM_WRITE_BUF_CAP);
        let mut buf = Vec::with_capacity(cap);
        buf.resize(HEADER_BYTES, 0u8);
        buf[0..4].copy_from_slice(&num_data.to_le_bytes());
        buf[4..8].copy_from_slice(&num_pix.to_le_bytes());

        for &(ptr, b_len) in &segments {
            let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, b_len) };
            if buf.len() + b_len <= cap {
                buf.extend_from_slice(bytes);
            } else {
                if !buf.is_empty() {
                    if !raw_target.write_all(&buf) {
                        return false;
                    }
                    buf.clear();
                }
                if b_len >= cap {
                    if !raw_target.write_all(bytes) {
                        return false;
                    }
                } else {
                    buf.extend_from_slice(bytes);
                }
            }
        }
        if !buf.is_empty() && !raw_target.write_all(&buf) {
            return false;
        }
        true
    });

    // Keep borrowed arrays and file object alive until py.detach completes.
    drop((
        ones_list,
        multi_list,
        place_ones_list,
        place_multi_list,
        count_multi_list,
    ));

    if !ok {
        return Err(PyOSError::new_err("Failed to write EMC binary data"));
    }
    Ok(())
}

// ============================================================================
// 2. In-memory reductions, todense, and fused SpMM / SpMV (@)
// ============================================================================

#[pyfunction]
pub fn sum_total_i128(
    py: Python<'_>,
    num_ones: usize,
    count_multi: PyReadonlyArray1<'_, i32>,
    n_workers: usize,
) -> PyResult<i128> {
    let cm = count_multi.as_slice()?;
    let cm_addr = cm.as_ptr() as usize;
    let cm_len = cm.len();

    let multi_sum: i128 = py.detach(|| {
        let slice = unsafe { std::slice::from_raw_parts(cm_addr as *const i32, cm_len) };
        if n_workers > 1 && cm_len >= 524_288 {
            let w = n_workers.min(8);
            let chunk = (cm_len / w).max(65_536);
            get_pool(w).install(|| {
                slice
                    .par_chunks(chunk)
                    .map(|part| part.iter().map(|&v| v as i64).sum::<i64>() as i128)
                    .sum()
            })
        } else {
            slice
                .chunks(1 << 26)
                .map(|part| part.iter().map(|&v| v as i64).sum::<i64>() as i128)
                .sum()
        }
    });

    Ok((num_ones as i128) + multi_sum)
}

#[pyfunction]
pub fn sum_axis1_i64<'py>(
    py: Python<'py>,
    ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    count_multi: PyReadonlyArray1<'py, i32>,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let s_ones = ones.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let cm = count_multi.as_slice()?;
    let num_data = s_ones.len();
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if total_multi != cm.len() {
        return Err(PyValueError::new_err(
            "multi_idx and count_multi lengths do not match",
        ));
    }

    let out_arr = PyArray1::<i64>::zeros(py, [num_data], false);
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i64, num_data) };
        if total_multi == 0 {
            for r in 0..num_data {
                out[r] = s_ones[r] as i64;
            }
            return;
        }
        let sum_row = |r: usize| -> i64 {
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return 0;
            }
            let mut acc = s_ones[r] as i64;
            for &v in &cm[m0..m1] {
                acc += v as i64;
            }
            acc
        };

        if n_workers > 1 && (num_data >= 32_768 || total_multi >= 524_288) && num_data >= 64 {
            let w = n_workers.min((num_data / 64).max(1));
            get_pool(w).install(|| {
                out.par_iter_mut()
                    .enumerate()
                    .for_each(|(r, slot)| *slot = sum_row(r));
            });
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = sum_row(r);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    Ok(out_arr)
}

#[pyfunction]
pub fn sum_axis0_i64<'py>(
    py: Python<'py>,
    num_pix: usize,
    place_ones: PyReadonlyArray1<'py, u32>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;
    if pm.len() != cm.len() {
        return Err(PyValueError::new_err(
            "place_multi and count_multi lengths must match",
        ));
    }

    let out_arr = PyArray1::<i64>::zeros(py, [num_pix], false);
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let oob = AtomicBool::new(false);

    py.detach(|| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i64, num_pix) };
        let total_events = po.len() + pm.len();
        let use_par =
            n_workers > 1 && total_events >= 1_048_576 && num_pix > 0 && num_pix <= 2_097_152;

        if use_par {
            let w = n_workers.min(8);
            let partials: Vec<Vec<i64>> = get_pool(w).install(|| {
                (0..w)
                    .into_par_iter()
                    .map(|tid| {
                        let mut local = vec![0i64; num_pix];
                        let o_beg = po.len() * tid / w;
                        let o_end = po.len() * (tid + 1) / w;
                        for &idx_u32 in &po[o_beg..o_end] {
                            let idx = idx_u32 as usize;
                            if idx >= num_pix {
                                oob.store(true, Ordering::Relaxed);
                                return local;
                            }
                            local[idx] += 1;
                        }
                        let m_beg = pm.len() * tid / w;
                        let m_end = pm.len() * (tid + 1) / w;
                        for (&idx_u32, &val) in pm[m_beg..m_end].iter().zip(cm[m_beg..m_end].iter())
                        {
                            let idx = idx_u32 as usize;
                            if idx >= num_pix {
                                oob.store(true, Ordering::Relaxed);
                                return local;
                            }
                            local[idx] += val as i64;
                        }
                        local
                    })
                    .collect()
            });
            if !oob.load(Ordering::Relaxed) {
                for p in &partials {
                    for (dst, &src) in out.iter_mut().zip(p.iter()) {
                        *dst += src;
                    }
                }
            }
        } else {
            for &idx_u32 in po {
                let idx = idx_u32 as usize;
                if idx >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                out[idx] += 1;
            }
            for (&idx_u32, &val) in pm.iter().zip(cm.iter()) {
                let idx = idx_u32 as usize;
                if idx >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                out[idx] += val as i64;
            }
        }
    });

    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err(
            "pixel index out of bounds for num_pix",
        ));
    }

    Ok(out_arr)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn todense_2d<'py>(
    py: Python<'py>,
    num_data: usize,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let out_arr = PyArray2::<i32>::zeros(py, [num_data, num_pix], false);
    if num_data == 0 || num_pix == 0 {
        if total_ones > 0 || total_multi > 0 {
            return Err(PyIndexError::new_err(
                "pixel index out of bounds for num_pix",
            ));
        }
        return Ok(out_arr);
    }

    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let total_cells = num_data.saturating_mul(num_pix);
    let oob = AtomicBool::new(false);
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let scatter_row = |r: usize, row_dst: &mut [i32]| {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return;
            }
            for &idx_u32 in &po[o0..o1] {
                let idx = idx_u32 as usize;
                if idx >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                row_dst[idx] = row_dst[idx].wrapping_add(1);
            }
            for (&idx_u32, &val) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                let idx = idx_u32 as usize;
                if idx >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                row_dst[idx] = row_dst[idx].wrapping_add(val);
            }
        };

        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i32, total_cells) };
        if n_workers > 1 && num_data >= 8 && total_cells >= 262_144 {
            let w = n_workers.min((num_data / 4).max(1));
            get_pool(w).install(|| {
                out.par_chunks_mut(num_pix)
                    .enumerate()
                    .for_each(|(r, row_dst)| scatter_row(r, row_dst));
            });
        } else {
            for (r, row_dst) in out.chunks_mut(num_pix).enumerate() {
                scatter_row(r, row_dst);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err(
            "pixel index out of bounds for num_pix",
        ));
    }

    Ok(out_arr)
}

trait MatmulAccum: Element + Copy + Default + Send + Sync {
    fn from_cnt(c: i32) -> Self;
    fn add_assign(&mut self, rhs: Self);
    fn mul_add_assign(&mut self, cnt: Self, rhs: Self);
}

impl MatmulAccum for f64 {
    #[inline(always)]
    fn from_cnt(c: i32) -> Self {
        c as f64
    }
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self += rhs;
    }
    #[inline(always)]
    fn mul_add_assign(&mut self, cnt: Self, rhs: Self) {
        *self += cnt * rhs;
    }
}

impl MatmulAccum for i64 {
    #[inline(always)]
    fn from_cnt(c: i32) -> Self {
        c as i64
    }
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.wrapping_add(rhs);
    }
    #[inline(always)]
    fn mul_add_assign(&mut self, cnt: Self, rhs: Self) {
        *self = self.wrapping_add(cnt.wrapping_mul(rhs));
    }
}

impl MatmulAccum for i32 {
    #[inline(always)]
    fn from_cnt(c: i32) -> Self {
        c
    }
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.wrapping_add(rhs);
    }
    #[inline(always)]
    fn mul_add_assign(&mut self, cnt: Self, rhs: Self) {
        *self = self.wrapping_add(cnt.wrapping_mul(rhs));
    }
}

trait IntoAccum<Out: MatmulAccum>: Element + Copy + Sync {
    fn into_acc(self) -> Out;
}

macro_rules! impl_into_accum {
    ($src:ty => $dst:ty : $expr:ident => $conv:expr) => {
        impl IntoAccum<$dst> for $src {
            #[inline(always)]
            fn into_acc(self) -> $dst {
                let $expr = self;
                $conv
            }
        }
    };
}

impl_into_accum!(f64 => f64 : x => x);
impl_into_accum!(f32 => f64 : x => x as f64);
impl_into_accum!(u64 => f64 : x => x as f64);
impl_into_accum!(i64 => i64 : x => x);
impl_into_accum!(u32 => i64 : x => x as i64);
impl_into_accum!(i32 => i32 : x => x);
impl_into_accum!(i16 => i32 : x => x as i32);
impl_into_accum!(u16 => i32 : x => x as i32);
impl_into_accum!(i8 => i32 : x => x as i32);
impl_into_accum!(u8 => i32 : x => x as i32);
impl_into_accum!(bool => i32 : x => i32::from(x));

#[allow(clippy::too_many_arguments)]
fn run_spmv_1d<'py, T, Out>(
    py: Python<'py>,
    num_data: usize,
    num_pix: usize,
    oidx: &[u64],
    po: &[u32],
    midx: &[u64],
    pm: &[u32],
    cm: &[i32],
    rhs_arr: PyReadonlyArray1<'py, T>,
    n_workers: usize,
) -> PyResult<Option<Py<PyAny>>>
where
    T: IntoAccum<Out>,
    Out: MatmulAccum,
{
    if !rhs_arr.is_c_contiguous() {
        return Ok(None);
    }
    let rhs = rhs_arr.as_slice()?;
    if rhs.len() != num_pix {
        return Ok(None);
    }

    let out_arr = PyArray1::<Out>::zeros(py, [num_data], false);
    if num_data == 0 || (po.is_empty() && pm.is_empty()) {
        return Ok(Some(out_arr.into_any().unbind()));
    }

    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let total_ones = po.len();
    let total_multi = pm.len();
    let oob = AtomicBool::new(false);
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let eval_row = |r: usize| -> Out {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return Out::default();
            }
            let mut acc = Out::default();
            for &col_u32 in &po[o0..o1] {
                let col = col_u32 as usize;
                if col >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return Out::default();
                }
                acc.add_assign(rhs[col].into_acc());
            }
            for (&col_u32, &cnt) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                let col = col_u32 as usize;
                if col >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return Out::default();
                }
                acc.mul_add_assign(Out::from_cnt(cnt), rhs[col].into_acc());
            }
            acc
        };

        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Out, num_data) };
        let work = total_ones + total_multi;
        if n_workers > 1 && num_data >= 64 && work >= 131_072 {
            let w = n_workers.min((num_data / 32).max(1));
            get_pool(w).install(|| {
                out.par_iter_mut()
                    .enumerate()
                    .for_each(|(r, slot)| *slot = eval_row(r));
            });
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = eval_row(r);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err(
            "pixel index out of bounds for num_pix",
        ));
    }

    Ok(Some(out_arr.into_any().unbind()))
}

#[allow(clippy::too_many_arguments)]
fn run_spmm_2d<'py, T, Out>(
    py: Python<'py>,
    num_data: usize,
    num_pix: usize,
    oidx: &[u64],
    po: &[u32],
    midx: &[u64],
    pm: &[u32],
    cm: &[i32],
    rhs_arr: PyReadonlyArray2<'py, T>,
    n_workers: usize,
) -> PyResult<Option<Py<PyAny>>>
where
    T: IntoAccum<Out>,
    Out: MatmulAccum,
{
    if !rhs_arr.is_c_contiguous() {
        return Ok(None);
    }
    let shape = rhs_arr.shape();
    if shape.len() != 2 || shape[0] != num_pix {
        return Ok(None);
    }
    let k_cols = shape[1];
    let rhs = rhs_arr.as_slice()?;

    let out_arr = PyArray2::<Out>::zeros(py, [num_data, k_cols], false);
    if num_data == 0 || k_cols == 0 || (po.is_empty() && pm.is_empty()) {
        if num_pix == 0 && (!po.is_empty() || !pm.is_empty()) {
            return Err(PyIndexError::new_err(
                "pixel index out of bounds for num_pix",
            ));
        }
        return Ok(Some(out_arr.into_any().unbind()));
    }

    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let total_ones = po.len();
    let total_multi = pm.len();
    let total_out = num_data.saturating_mul(k_cols);
    let oob = AtomicBool::new(false);
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let eval_row = |r: usize, dst_row: &mut [Out]| {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return;
            }
            for &col_u32 in &po[o0..o1] {
                let col = col_u32 as usize;
                if col >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                let src_row = &rhs[col * k_cols..(col + 1) * k_cols];
                for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                    d.add_assign(s.into_acc());
                }
            }
            for (&col_u32, &cnt) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                let col = col_u32 as usize;
                if col >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return;
                }
                let c_acc = Out::from_cnt(cnt);
                let src_row = &rhs[col * k_cols..(col + 1) * k_cols];
                for (d, &s) in dst_row.iter_mut().zip(src_row.iter()) {
                    d.mul_add_assign(c_acc, s.into_acc());
                }
            }
        };

        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Out, total_out) };
        let work = (total_ones + total_multi).saturating_mul(k_cols);
        if n_workers > 1 && num_data >= 16 && work >= 131_072 {
            let w = n_workers.min((num_data / 8).max(1));
            let chunk_rows = num_data.div_ceil(w);
            get_pool(w).install(|| {
                out.par_chunks_mut(chunk_rows * k_cols)
                    .enumerate()
                    .for_each(|(tid, block)| {
                        let base_r = tid * chunk_rows;
                        for (off, dst_row) in block.chunks_mut(k_cols).enumerate() {
                            eval_row(base_r + off, dst_row);
                        }
                    });
            });
        } else {
            for (r, dst_row) in out.chunks_mut(k_cols).enumerate() {
                eval_row(r, dst_row);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err(
            "pixel index out of bounds for num_pix",
        ));
    }

    Ok(Some(out_arr.into_any().unbind()))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn matmul_dense<'py>(
    py: Python<'py>,
    num_data: usize,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    rhs: &Bound<'py, PyAny>,
    n_workers: usize,
) -> PyResult<Option<Py<PyAny>>> {
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    macro_rules! try_dispatch {
        ($t:ty => $out:ty) => {
            if let Ok(a2) = rhs.extract::<PyReadonlyArray2<'py, $t>>() {
                return run_spmm_2d::<$t, $out>(
                    py, num_data, num_pix, oidx, po, midx, pm, cm, a2, n_workers,
                );
            }
            if let Ok(a1) = rhs.extract::<PyReadonlyArray1<'py, $t>>() {
                return run_spmv_1d::<$t, $out>(
                    py, num_data, num_pix, oidx, po, midx, pm, cm, a1, n_workers,
                );
            }
        };
    }

    try_dispatch!(f64 => f64);
    try_dispatch!(f32 => f64);
    try_dispatch!(i64 => i64);
    try_dispatch!(i32 => i32);
    try_dispatch!(bool => i32);
    try_dispatch!(u32 => i64);
    try_dispatch!(u64 => f64);
    try_dispatch!(i16 => i32);
    try_dispatch!(u16 => i32);
    try_dispatch!(i8 => i32);
    try_dispatch!(u8 => i32);

    Ok(None)
}

// ============================================================================
// 3. In-memory row gathering & column selection
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn gather_in_memory_from_ids<'py>(
    py: Python<'py>,
    ones: &[u32],
    multi: &[u32],
    oidx: &[u64],
    midx: &[u64],
    po: &[u32],
    pm: &[u32],
    cm: &[i32],
    ids: &[usize],
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    let k = ids.len();
    let arr_ones = PyArray1::<u32>::zeros(py, [k], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [k], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [k + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [k + 1], false);

    let s_ones = unsafe { arr_ones.as_slice_mut()? };
    let s_multi = unsafe { arr_multi.as_slice_mut()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };

    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut acc_o = 0u64;
    let mut acc_m = 0u64;
    for (i, &r) in ids.iter().enumerate() {
        let co = ones[r];
        let cm_cnt = multi[r];
        s_ones[i] = co;
        s_multi[i] = cm_cnt;
        acc_o += co as u64;
        acc_m += cm_cnt as u64;
        s_oidx[i + 1] = acc_o;
        s_midx[i + 1] = acc_m;
    }

    let out_ones_len = usize::try_from(acc_o)
        .map_err(|_| PyValueError::new_err("Selected place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(acc_m)
        .map_err(|_| PyValueError::new_err("Selected place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_out_ref: &[u64] = s_oidx;
    let midx_out_ref: &[u64] = s_midx;

    let total_ones = po.len();
    let total_multi = pm.len();
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let copy_row = |i: usize| {
            let r = ids[i];
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            let lo = o1.wrapping_sub(o0);
            let lm = m1.wrapping_sub(m0);
            let exp_lo = (oidx_out_ref[i + 1] - oidx_out_ref[i]) as usize;
            let exp_lm = (midx_out_ref[i + 1] - midx_out_ref[i]) as usize;
            if o1 < o0
                || o1 > total_ones
                || m1 < m0
                || m1 > total_multi
                || lo != exp_lo
                || lm != exp_lm
            {
                bad.store(true, Ordering::Relaxed);
                return;
            }
            if lo > 0 {
                let do0 = oidx_out_ref[i] as usize;
                let dst =
                    unsafe { std::slice::from_raw_parts_mut((po_addr as *mut u32).add(do0), lo) };
                dst.copy_from_slice(&po[o0..o1]);
            }
            if lm > 0 {
                let dm0 = midx_out_ref[i] as usize;
                let dst_pm =
                    unsafe { std::slice::from_raw_parts_mut((pm_addr as *mut u32).add(dm0), lm) };
                let dst_cm =
                    unsafe { std::slice::from_raw_parts_mut((cm_addr as *mut i32).add(dm0), lm) };
                dst_pm.copy_from_slice(&pm[m0..m1]);
                dst_cm.copy_from_slice(&cm[m0..m1]);
            }
        };

        let total_work = out_ones_len + out_multi_len * 2 + k;
        if n_workers > 1 && k >= 512 && total_work >= 262_144 {
            let w = n_workers.min((k / 128).max(1));
            get_pool(w).install(|| {
                (0..k).into_par_iter().for_each(copy_row);
            });
        } else {
            for i in 0..k {
                copy_row(i);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Pattern counts do not match offsets"));
    }

    Ok((
        arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gather_rows_ids<'py>(
    py: Python<'py>,
    ones: PyReadonlyArray1<'py, u32>,
    multi: PyReadonlyArray1<'py, u32>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    row_ids: PyReadonlyArray1<'py, i64>,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    let s_ones = ones.as_slice()?;
    let s_multi = multi.as_slice()?;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;
    let raw_ids = row_ids.as_slice()?;

    let num_data = s_ones.len();
    if s_multi.len() != num_data {
        return Err(PyValueError::new_err("ones and multi lengths must match"));
    }
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let mut ids = Vec::with_capacity(raw_ids.len());
    for &raw in raw_ids {
        let r = normalize_row_id(raw, num_data)
            .map_err(|_| PyIndexError::new_err("row index out of range"))?;
        ids.push(r);
    }

    gather_in_memory_from_ids(py, s_ones, s_multi, oidx, midx, po, pm, cm, &ids, n_workers)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gather_rows_strided<'py>(
    py: Python<'py>,
    ones: PyReadonlyArray1<'py, u32>,
    multi: PyReadonlyArray1<'py, u32>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    start: isize,
    stop: isize,
    step: isize,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    let s_ones = ones.as_slice()?;
    let s_multi = multi.as_slice()?;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    let num_data = s_ones.len();
    if s_multi.len() != num_data {
        return Err(PyValueError::new_err("ones and multi lengths must match"));
    }
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let k = strided_len(start, stop, step);
    let mut ids = Vec::with_capacity(k);
    let mut cur = start;
    for _ in 0..k {
        if cur < 0 || (cur as usize) >= num_data {
            return Err(PyIndexError::new_err("row index out of range"));
        }
        ids.push(cur as usize);
        cur += step;
    }

    gather_in_memory_from_ids(py, s_ones, s_multi, oidx, midx, po, pm, cm, &ids, n_workers)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gather_rows_mask<'py>(
    py: Python<'py>,
    ones: PyReadonlyArray1<'py, u32>,
    multi: PyReadonlyArray1<'py, u32>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    mask: PyReadonlyArray1<'py, bool>,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    let s_ones = ones.as_slice()?;
    let s_multi = multi.as_slice()?;
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;
    let s_mask = mask.as_slice()?;

    let num_data = s_ones.len();
    if s_mask.len() != num_data {
        return Err(PyIndexError::new_err(
            "Boolean row mask must match the number of patterns",
        ));
    }
    if s_multi.len() != num_data {
        return Err(PyValueError::new_err("ones and multi lengths must match"));
    }
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let mut ids = Vec::new();
    for (i, &keep) in s_mask.iter().enumerate() {
        if keep {
            ids.push(i);
        }
    }

    gather_in_memory_from_ids(py, s_ones, s_multi, oidx, midx, po, pm, cm, &ids, n_workers)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn extract_column_1d<'py>(
    py: Python<'py>,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    row_ids: Option<PyReadonlyArray1<'py, i64>>,
    col: usize,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    if col >= num_pix {
        return Err(PyIndexError::new_err("column index out of bounds"));
    }
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    if oidx.is_empty() || oidx.len() != midx.len() {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    let num_data = oidx.len() - 1;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let resolved_ids: Option<Vec<usize>> = if let Some(ref r_arr) = row_ids {
        let raw = r_arr.as_slice()?;
        let mut v = Vec::with_capacity(raw.len());
        for &id in raw {
            let norm = normalize_row_id(id, num_data)
                .map_err(|_| PyIndexError::new_err("row index out of range"))?;
            v.push(norm);
        }
        Some(v)
    } else {
        None
    };

    let out_len = resolved_ids.as_ref().map_or(num_data, Vec::len);
    let out_arr = PyArray1::<i32>::zeros(py, [out_len], false);
    let Ok(target_col) = u32::try_from(col) else {
        return Ok(out_arr);
    };

    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let bad = AtomicBool::new(false);

    py.detach(|| {
        let eval_row = |r: usize| -> i32 {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return 0;
            }
            let mut acc: i32 = 0;
            for &p in &po[o0..o1] {
                if p == target_col {
                    acc = acc.wrapping_add(1);
                }
            }
            for (&p, &val) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                if p == target_col {
                    acc = acc.wrapping_add(val);
                }
            }
            acc
        };

        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut i32, out_len) };
        if n_workers > 1 && out_len >= 4096 {
            let w = n_workers.min((out_len / 1024).max(1));
            get_pool(w).install(|| {
                if let Some(ref ids) = resolved_ids {
                    out.par_iter_mut()
                        .enumerate()
                        .for_each(|(i, slot)| *slot = eval_row(ids[i]));
                } else {
                    out.par_iter_mut()
                        .enumerate()
                        .for_each(|(r, slot)| *slot = eval_row(r));
                }
            });
        } else if let Some(ref ids) = resolved_ids {
            for (i, &r) in ids.iter().enumerate() {
                out[i] = eval_row(r);
            }
        } else {
            for (r, slot) in out.iter_mut().enumerate() {
                *slot = eval_row(r);
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }

    Ok(out_arr)
}

#[allow(clippy::too_many_arguments)]
fn filter_columns_unique<'py, F>(
    py: Python<'py>,
    num_pix: usize,
    oidx: &[u64],
    po: &[u32],
    midx: &[u64],
    pm: &[u32],
    cm: &[i32],
    col_count: usize,
    n_workers: usize,
    map_fn: F,
) -> PyResult<SizedFullPatternArrays<'py>>
where
    F: Fn(u32) -> (bool, u32) + Sync,
{
    let num_data = oidx.len() - 1;
    let total_ones = po.len();
    let total_multi = pm.len();

    let arr_ones = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [num_data + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [num_data + 1], false);

    if num_data == 0 || col_count == 0 {
        for &p in po {
            if (p as usize) >= num_pix {
                return Err(PyIndexError::new_err("pixel index out of bounds"));
            }
        }
        for &p in pm {
            if (p as usize) >= num_pix {
                return Err(PyIndexError::new_err("pixel index out of bounds"));
            }
        }
        let empty_po = PyArray1::<u32>::zeros(py, [0], false);
        let empty_pm = PyArray1::<u32>::zeros(py, [0], false);
        let empty_cm = PyArray1::<i32>::zeros(py, [0], false);
        return Ok((
            col_count,
            (
                arr_ones, arr_multi, arr_oidx, arr_midx, empty_po, empty_pm, empty_cm,
            ),
        ));
    }

    let ones_addr = unsafe { arr_ones.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { arr_multi.as_slice_mut()? }.as_mut_ptr() as usize;
    let oob = AtomicBool::new(false);
    let bad = AtomicBool::new(false);

    let total_in_events = total_ones + total_multi;
    let use_par = n_workers > 1 && num_data >= 128 && total_in_events >= 131_072;

    py.detach(|| {
        let count_chunk = |r_start: usize, r_end: usize| {
            let out_ones = unsafe {
                std::slice::from_raw_parts_mut(
                    (ones_addr as *mut u32).add(r_start),
                    r_end - r_start,
                )
            };
            let out_multi = unsafe {
                std::slice::from_raw_parts_mut(
                    (multi_addr as *mut u32).add(r_start),
                    r_end - r_start,
                )
            };
            for (local_i, r) in (r_start..r_end).enumerate() {
                let o0 = oidx[r] as usize;
                let o1 = oidx[r + 1] as usize;
                let m0 = midx[r] as usize;
                let m1 = midx[r + 1] as usize;
                if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                    bad.store(true, Ordering::Relaxed);
                    return;
                }
                let mut co = 0u32;
                for &p in &po[o0..o1] {
                    if (p as usize) >= num_pix {
                        oob.store(true, Ordering::Relaxed);
                        return;
                    }
                    let (keep, _) = map_fn(p);
                    co += u32::from(keep);
                }
                let mut cm_cnt = 0u32;
                for &p in &pm[m0..m1] {
                    if (p as usize) >= num_pix {
                        oob.store(true, Ordering::Relaxed);
                        return;
                    }
                    let (keep, _) = map_fn(p);
                    cm_cnt += u32::from(keep);
                }
                out_ones[local_i] = co;
                out_multi[local_i] = cm_cnt;
            }
        };

        if use_par {
            let w = n_workers.min((num_data / 32).max(1));
            let chunk_rows = num_data.div_ceil(w);
            get_pool(w).install(|| {
                (0..w).into_par_iter().for_each(|tid| {
                    let rs = (tid * chunk_rows).min(num_data);
                    let re = ((tid + 1) * chunk_rows).min(num_data);
                    if rs < re {
                        count_chunk(rs, re);
                    }
                });
            });
        } else {
            count_chunk(0, num_data);
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err("pixel index out of bounds"));
    }

    let s_ones = unsafe { arr_ones.as_slice()? };
    let s_multi = unsafe { arr_multi.as_slice()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };
    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut acc_o = 0u64;
    let mut acc_m = 0u64;
    for r in 0..num_data {
        acc_o += s_ones[r] as u64;
        acc_m += s_multi[r] as u64;
        s_oidx[r + 1] = acc_o;
        s_midx[r + 1] = acc_m;
    }

    let out_ones_len = usize::try_from(acc_o)
        .map_err(|_| PyValueError::new_err("Selected place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(acc_m)
        .map_err(|_| PyValueError::new_err("Selected place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_out_ref: &[u64] = s_oidx;
    let midx_out_ref: &[u64] = s_midx;

    py.detach(|| {
        let scatter_chunk = |r_start: usize, r_end: usize| {
            let mut row_po: Vec<u32> = Vec::new();
            let mut row_pm: Vec<u32> = Vec::new();
            let mut row_cm: Vec<i32> = Vec::new();

            for r in r_start..r_end {
                let o0 = oidx[r] as usize;
                let o1 = oidx[r + 1] as usize;
                let do_cur = oidx_out_ref[r] as usize;
                let do_end = oidx_out_ref[r + 1] as usize;
                let want_o = do_end - do_cur;
                if want_o > 0 {
                    if row_po.len() < want_o + 1 {
                        row_po.resize(want_o + 1, 0);
                    }
                    let mut k = 0usize;
                    for &p in &po[o0..o1] {
                        let (keep, np) = map_fn(p);
                        unsafe {
                            *row_po.get_unchecked_mut(k) = np;
                        }
                        k += usize::from(keep);
                    }
                    debug_assert_eq!(k, want_o);
                    let dst_po = unsafe {
                        std::slice::from_raw_parts_mut((po_addr as *mut u32).add(do_cur), want_o)
                    };
                    dst_po.copy_from_slice(&row_po[..want_o]);
                }

                let m0 = midx[r] as usize;
                let m1 = midx[r + 1] as usize;
                let dm_cur = midx_out_ref[r] as usize;
                let dm_end = midx_out_ref[r + 1] as usize;
                let want_m = dm_end - dm_cur;
                if want_m > 0 {
                    if row_pm.len() < want_m + 1 {
                        row_pm.resize(want_m + 1, 0);
                        row_cm.resize(want_m + 1, 0);
                    }
                    let mut k = 0usize;
                    for (&p, &val) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                        let (keep, np) = map_fn(p);
                        unsafe {
                            *row_pm.get_unchecked_mut(k) = np;
                            *row_cm.get_unchecked_mut(k) = val;
                        }
                        k += usize::from(keep);
                    }
                    debug_assert_eq!(k, want_m);
                    let dst_pm = unsafe {
                        std::slice::from_raw_parts_mut((pm_addr as *mut u32).add(dm_cur), want_m)
                    };
                    let dst_cm = unsafe {
                        std::slice::from_raw_parts_mut((cm_addr as *mut i32).add(dm_cur), want_m)
                    };
                    dst_pm.copy_from_slice(&row_pm[..want_m]);
                    dst_cm.copy_from_slice(&row_cm[..want_m]);
                }
            }
        };

        if use_par {
            let w = n_workers.min((num_data / 32).max(1));
            let chunk_rows = num_data.div_ceil(w);
            get_pool(w).install(|| {
                (0..w).into_par_iter().for_each(|tid| {
                    let rs = (tid * chunk_rows).min(num_data);
                    let re = ((tid + 1) * chunk_rows).min(num_data);
                    if rs < re {
                        scatter_chunk(rs, re);
                    }
                });
            });
        } else {
            scatter_chunk(0, num_data);
        }
    });

    Ok((
        col_count,
        (
            arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
        ),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn select_columns_slice<'py>(
    py: Python<'py>,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    start: isize,
    stop: isize,
    step: isize,
    n_workers: usize,
) -> PyResult<SizedFullPatternArrays<'py>> {
    if step == 0 {
        return Err(PyValueError::new_err("slice step cannot be zero"));
    }
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    if oidx.is_empty() || oidx.len() != midx.len() {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    let num_data = oidx.len() - 1;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let col_count = strided_len(start, stop, step);
    if col_count > (u32::MAX as usize) + 1 {
        return Err(PyValueError::new_err(
            "Selected column count exceeds uint32",
        ));
    }

    if step == 1 {
        let s_u = start.max(0) as usize;
        let e_u = stop.max(0) as usize;
        filter_columns_unique(
            py,
            num_pix,
            oidx,
            po,
            midx,
            pm,
            cm,
            col_count,
            n_workers,
            |p| {
                let pu = p as usize;
                let keep = pu >= s_u && pu < e_u;
                (keep, pu.wrapping_sub(s_u) as u32)
            },
        )
    } else if step > 1 {
        let s_u = start.max(0) as usize;
        let e_u = stop.max(0) as usize;
        let st_u = step as usize;
        filter_columns_unique(
            py,
            num_pix,
            oidx,
            po,
            midx,
            pm,
            cm,
            col_count,
            n_workers,
            |p| {
                let pu = p as usize;
                if pu >= s_u && pu < e_u {
                    let diff = pu - s_u;
                    if diff.is_multiple_of(st_u) {
                        (true, (diff / st_u) as u32)
                    } else {
                        (false, 0)
                    }
                } else {
                    (false, 0)
                }
            },
        )
    } else {
        // step < 0: selected pixels satisfy stop < p <= start and (start - p) % (-step) == 0
        let s_i = start;
        let e_i = stop;
        let neg_st = (-step) as usize;
        filter_columns_unique(
            py,
            num_pix,
            oidx,
            po,
            midx,
            pm,
            cm,
            col_count,
            n_workers,
            |p| {
                let pi = p as isize;
                if pi <= s_i && pi > e_i {
                    let diff = (s_i - pi) as usize;
                    if diff.is_multiple_of(neg_st) {
                        (true, (diff / neg_st) as u32)
                    } else {
                        (false, 0)
                    }
                } else {
                    (false, 0)
                }
            },
        )
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn select_columns_mask<'py>(
    py: Python<'py>,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    mask: PyReadonlyArray1<'py, bool>,
    n_workers: usize,
) -> PyResult<SizedFullPatternArrays<'py>> {
    let s_mask = mask.as_slice()?;
    if s_mask.len() != num_pix {
        return Err(PyIndexError::new_err(
            "Boolean column mask must match the number of pixels",
        ));
    }
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    if oidx.is_empty() || oidx.len() != midx.len() {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    let num_data = oidx.len() - 1;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    // Compact L2-resident 64-bit bitset + popcount rank directory (0.1875 bytes/pixel).
    let n_words = num_pix.div_ceil(64);
    let mut words = vec![0u64; n_words];
    let mut ranks = vec![0u32; n_words];
    let mut running: usize = 0;

    for (w_idx, chunk) in s_mask.chunks(64).enumerate() {
        ranks[w_idx] = running as u32;
        let mut w = 0u64;
        for (bit, &keep) in chunk.iter().enumerate() {
            if keep {
                w |= 1u64 << bit;
            }
        }
        words[w_idx] = w;
        running += w.count_ones() as usize;
    }

    if running > (u32::MAX as usize) + 1 {
        return Err(PyValueError::new_err(
            "Selected column count exceeds uint32",
        ));
    }

    filter_columns_unique(
        py,
        num_pix,
        oidx,
        po,
        midx,
        pm,
        cm,
        running,
        n_workers,
        |p| {
            let pu = p as usize;
            let w_idx = pu >> 6;
            let bit = 1u64 << (pu & 63);
            let w = unsafe { *words.get_unchecked(w_idx) };
            let r = unsafe { *ranks.get_unchecked(w_idx) };
            ((w & bit) != 0, r + (w & (bit - 1)).count_ones())
        },
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn select_columns_indices<'py>(
    py: Python<'py>,
    num_pix: usize,
    ones_idx: PyReadonlyArray1<'py, u64>,
    place_ones: PyReadonlyArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    place_multi: PyReadonlyArray1<'py, u32>,
    count_multi: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i64>,
    n_workers: usize,
) -> PyResult<SizedFullPatternArrays<'py>> {
    let raw_cols = cols.as_slice()?;
    let col_count = raw_cols.len();
    if col_count > (u32::MAX as usize) + 1 {
        return Err(PyValueError::new_err(
            "Selected column count exceeds uint32",
        ));
    }

    let mut norm_cols = Vec::with_capacity(col_count);
    for &c in raw_cols {
        let nc = normalize_row_id(c, num_pix)
            .map_err(|_| PyIndexError::new_err("column index out of range"))?;
        norm_cols.push(nc);
    }

    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let pm = place_multi.as_slice()?;
    let cm = count_multi.as_slice()?;

    if oidx.is_empty() || oidx.len() != midx.len() {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    let num_data = oidx.len() - 1;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    if col_count == 0 || num_data == 0 {
        return filter_columns_unique(
            py,
            num_pix,
            oidx,
            po,
            midx,
            pm,
            cm,
            col_count,
            n_workers,
            |_| (false, 0),
        );
    }

    // Fast path for normal detector sizes (<= 32M pixels): compact 32-bit direct table.
    // Tag encoding: u32::MAX = unselected; < 0x8000_0000 = single target col;
    // 0x8000_0000 | idx = duplicate targets slice dup_targets[dup_spans[idx]].
    if num_pix <= 33_554_432 && col_count < 0x8000_0000 {
        let mut remap = vec![u32::MAX; num_pix];
        let mut has_dups = false;
        for (dst_idx, &src_col) in norm_cols.iter().enumerate() {
            if remap[src_col] != u32::MAX {
                has_dups = true;
                break;
            }
            remap[src_col] = dst_idx as u32;
        }
        if !has_dups {
            return filter_columns_unique(
                py,
                num_pix,
                oidx,
                po,
                midx,
                pm,
                cm,
                col_count,
                n_workers,
                |p| {
                    let mapped = unsafe { *remap.get_unchecked(p as usize) };
                    (mapped != u32::MAX, mapped)
                },
            );
        }

        // Duplicate columns present: single L1/L2-resident remap[num_pix] + compact dup lists.
        remap.fill(u32::MAX);
        let mut dup_lists: Vec<Vec<u32>> = Vec::new();
        for (dst_idx, &src_col) in norm_cols.iter().enumerate() {
            let cur = remap[src_col];
            if cur == u32::MAX {
                remap[src_col] = dst_idx as u32;
            } else if (cur & 0x8000_0000) == 0 {
                let d_idx = dup_lists.len() as u32;
                dup_lists.push(vec![cur, dst_idx as u32]);
                remap[src_col] = 0x8000_0000 | d_idx;
            } else {
                dup_lists[(cur & 0x7FFF_FFFF) as usize].push(dst_idx as u32);
            }
        }
        let mut dup_targets: Vec<u32> = Vec::new();
        let mut dup_spans: Vec<(usize, usize)> = Vec::with_capacity(dup_lists.len());
        for lst in dup_lists {
            let b = dup_targets.len();
            dup_targets.extend_from_slice(&lst);
            dup_spans.push((b, dup_targets.len()));
        }

        return filter_columns_tagged_table(
            py,
            num_pix,
            oidx,
            po,
            midx,
            pm,
            cm,
            col_count,
            n_workers,
            &remap,
            &dup_spans,
            &dup_targets,
        );
    }

    // Wide pixel-space path (> 32M pixels): O(|cols|) sorted pairs + binary search.
    let mut pairs: Vec<(usize, u32)> = norm_cols
        .iter()
        .enumerate()
        .map(|(dst, &src)| (src, dst as u32))
        .collect();
    pairs.sort_by_key(|&(src, _)| src);
    let mut targets = Vec::with_capacity(col_count);
    let mut unique_src = Vec::new();
    let mut spans = Vec::new();
    let mut idx = 0usize;
    while idx < pairs.len() {
        let src = pairs[idx].0;
        let beg = targets.len();
        while idx < pairs.len() && pairs[idx].0 == src {
            targets.push(pairs[idx].1);
            idx += 1;
        }
        unique_src.push(src);
        spans.push((beg, targets.len()));
    }

    filter_columns_multi(
        py,
        num_pix,
        oidx,
        po,
        midx,
        pm,
        cm,
        col_count,
        n_workers,
        |p| {
            let pu = p as usize;
            if let Ok(pos) = unique_src.binary_search(&pu) {
                let (b, e) = spans[pos];
                &targets[b..e]
            } else {
                &[]
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn filter_columns_tagged_table<'py>(
    py: Python<'py>,
    num_pix: usize,
    oidx: &[u64],
    po: &[u32],
    midx: &[u64],
    pm: &[u32],
    cm: &[i32],
    col_count: usize,
    n_workers: usize,
    remap: &[u32],
    dup_spans: &[(usize, usize)],
    dup_targets: &[u32],
) -> PyResult<SizedFullPatternArrays<'py>> {
    let num_data = oidx.len() - 1;
    let total_ones = po.len();
    let total_multi = pm.len();

    let arr_ones = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [num_data + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [num_data + 1], false);

    let ones_addr = unsafe { arr_ones.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { arr_multi.as_slice_mut()? }.as_mut_ptr() as usize;
    let oob = AtomicBool::new(false);
    let bad = AtomicBool::new(false);
    let overflow = AtomicBool::new(false);

    let total_in_events = total_ones + total_multi;
    let use_par = n_workers > 1 && num_data >= 128 && total_in_events >= 131_072;

    py.detach(|| {
        let count_chunk = |r_start: usize, r_end: usize| {
            let out_ones = unsafe {
                std::slice::from_raw_parts_mut(
                    (ones_addr as *mut u32).add(r_start),
                    r_end - r_start,
                )
            };
            let out_multi = unsafe {
                std::slice::from_raw_parts_mut(
                    (multi_addr as *mut u32).add(r_start),
                    r_end - r_start,
                )
            };
            for (local_i, r) in (r_start..r_end).enumerate() {
                let o0 = oidx[r] as usize;
                let o1 = oidx[r + 1] as usize;
                let m0 = midx[r] as usize;
                let m1 = midx[r + 1] as usize;
                if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                    bad.store(true, Ordering::Relaxed);
                    return;
                }
                let mut co: u64 = 0;
                for &p in &po[o0..o1] {
                    let pu = p as usize;
                    if pu >= num_pix {
                        oob.store(true, Ordering::Relaxed);
                        return;
                    }
                    let tag = unsafe { *remap.get_unchecked(pu) };
                    if tag < 0x8000_0000 {
                        co += 1;
                    } else if tag != u32::MAX {
                        let (b, e) =
                            unsafe { *dup_spans.get_unchecked((tag & 0x7FFF_FFFF) as usize) };
                        co += (e - b) as u64;
                    }
                }
                let mut cm_cnt: u64 = 0;
                for &p in &pm[m0..m1] {
                    let pu = p as usize;
                    if pu >= num_pix {
                        oob.store(true, Ordering::Relaxed);
                        return;
                    }
                    let tag = unsafe { *remap.get_unchecked(pu) };
                    if tag < 0x8000_0000 {
                        cm_cnt += 1;
                    } else if tag != u32::MAX {
                        let (b, e) =
                            unsafe { *dup_spans.get_unchecked((tag & 0x7FFF_FFFF) as usize) };
                        cm_cnt += (e - b) as u64;
                    }
                }
                if co > u32::MAX as u64 || cm_cnt > u32::MAX as u64 {
                    overflow.store(true, Ordering::Relaxed);
                    return;
                }
                out_ones[local_i] = co as u32;
                out_multi[local_i] = cm_cnt as u32;
            }
        };

        if use_par {
            let w = n_workers.min((num_data / 32).max(1));
            let chunk_rows = num_data.div_ceil(w);
            get_pool(w).install(|| {
                (0..w).into_par_iter().for_each(|tid| {
                    let rs = (tid * chunk_rows).min(num_data);
                    let re = ((tid + 1) * chunk_rows).min(num_data);
                    if rs < re {
                        count_chunk(rs, re);
                    }
                });
            });
        } else {
            count_chunk(0, num_data);
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err("pixel index out of bounds"));
    }
    if overflow.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err(
            "Selected per-pattern count exceeds uint32",
        ));
    }

    let s_ones = unsafe { arr_ones.as_slice()? };
    let s_multi = unsafe { arr_multi.as_slice()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };
    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut acc_o = 0u64;
    let mut acc_m = 0u64;
    for r in 0..num_data {
        acc_o += s_ones[r] as u64;
        acc_m += s_multi[r] as u64;
        s_oidx[r + 1] = acc_o;
        s_midx[r + 1] = acc_m;
    }

    let out_ones_len = usize::try_from(acc_o)
        .map_err(|_| PyValueError::new_err("Selected place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(acc_m)
        .map_err(|_| PyValueError::new_err("Selected place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_out_ref: &[u64] = s_oidx;
    let midx_out_ref: &[u64] = s_midx;

    py.detach(|| {
        let scatter_chunk = |r_start: usize, r_end: usize| {
            for r in r_start..r_end {
                let o0 = oidx[r] as usize;
                let o1 = oidx[r + 1] as usize;
                let do_cur = oidx_out_ref[r] as usize;
                let do_end = oidx_out_ref[r + 1] as usize;
                if do_end > do_cur {
                    let dst_po = unsafe {
                        std::slice::from_raw_parts_mut(
                            (po_addr as *mut u32).add(do_cur),
                            do_end - do_cur,
                        )
                    };
                    let mut k = 0usize;
                    for &p in &po[o0..o1] {
                        let tag = unsafe { *remap.get_unchecked(p as usize) };
                        if tag < 0x8000_0000 {
                            dst_po[k] = tag;
                            k += 1;
                        } else if tag != u32::MAX {
                            let (b, e) =
                                unsafe { *dup_spans.get_unchecked((tag & 0x7FFF_FFFF) as usize) };
                            let ts = &dup_targets[b..e];
                            dst_po[k..k + ts.len()].copy_from_slice(ts);
                            k += ts.len();
                        }
                    }
                    debug_assert_eq!(k, dst_po.len());
                }

                let m0 = midx[r] as usize;
                let m1 = midx[r + 1] as usize;
                let dm_cur = midx_out_ref[r] as usize;
                let dm_end = midx_out_ref[r + 1] as usize;
                if dm_end > dm_cur {
                    let dst_pm = unsafe {
                        std::slice::from_raw_parts_mut(
                            (pm_addr as *mut u32).add(dm_cur),
                            dm_end - dm_cur,
                        )
                    };
                    let dst_cm = unsafe {
                        std::slice::from_raw_parts_mut(
                            (cm_addr as *mut i32).add(dm_cur),
                            dm_end - dm_cur,
                        )
                    };
                    let mut k = 0usize;
                    for (&p, &val) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                        let tag = unsafe { *remap.get_unchecked(p as usize) };
                        if tag < 0x8000_0000 {
                            dst_pm[k] = tag;
                            dst_cm[k] = val;
                            k += 1;
                        } else if tag != u32::MAX {
                            let (b, e) =
                                unsafe { *dup_spans.get_unchecked((tag & 0x7FFF_FFFF) as usize) };
                            let ts = &dup_targets[b..e];
                            dst_pm[k..k + ts.len()].copy_from_slice(ts);
                            dst_cm[k..k + ts.len()].fill(val);
                            k += ts.len();
                        }
                    }
                    debug_assert_eq!(k, dst_pm.len());
                }
            }
        };

        if use_par {
            let w = n_workers.min((num_data / 32).max(1));
            let chunk_rows = num_data.div_ceil(w);
            get_pool(w).install(|| {
                (0..w).into_par_iter().for_each(|tid| {
                    let rs = (tid * chunk_rows).min(num_data);
                    let re = ((tid + 1) * chunk_rows).min(num_data);
                    if rs < re {
                        scatter_chunk(rs, re);
                    }
                });
            });
        } else {
            scatter_chunk(0, num_data);
        }
    });

    Ok((
        col_count,
        (
            arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
        ),
    ))
}

#[allow(clippy::too_many_arguments)]
fn filter_columns_multi<'py, 't, F>(
    py: Python<'py>,
    num_pix: usize,
    oidx: &[u64],
    po: &[u32],
    midx: &[u64],
    pm: &[u32],
    cm: &[i32],
    col_count: usize,
    n_workers: usize,
    targets_fn: F,
) -> PyResult<SizedFullPatternArrays<'py>>
where
    F: Fn(u32) -> &'t [u32] + Sync,
{
    let num_data = oidx.len() - 1;
    let total_ones = po.len();
    let total_multi = pm.len();

    let arr_ones = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [num_data + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [num_data + 1], false);

    let ones_addr = unsafe { arr_ones.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { arr_multi.as_slice_mut()? }.as_mut_ptr() as usize;
    let oob = AtomicBool::new(false);
    let bad = AtomicBool::new(false);
    let overflow = AtomicBool::new(false);

    let total_in_events = total_ones + total_multi;
    let use_par = n_workers > 1 && num_data >= 256 && total_in_events >= 262_144;

    py.detach(|| {
        let out_ones = unsafe { std::slice::from_raw_parts_mut(ones_addr as *mut u32, num_data) };
        let out_multi = unsafe { std::slice::from_raw_parts_mut(multi_addr as *mut u32, num_data) };

        let count_row = |r: usize| -> (u32, u32) {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            if o1 < o0 || o1 > total_ones || m1 < m0 || m1 > total_multi {
                bad.store(true, Ordering::Relaxed);
                return (0, 0);
            }
            let mut co: u64 = 0;
            for &p in &po[o0..o1] {
                if (p as usize) >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return (0, 0);
                }
                co += targets_fn(p).len() as u64;
            }
            let mut cm_cnt: u64 = 0;
            for &p in &pm[m0..m1] {
                if (p as usize) >= num_pix {
                    oob.store(true, Ordering::Relaxed);
                    return (0, 0);
                }
                cm_cnt += targets_fn(p).len() as u64;
            }
            if co > u32::MAX as u64 || cm_cnt > u32::MAX as u64 {
                overflow.store(true, Ordering::Relaxed);
                return (0, 0);
            }
            (co as u32, cm_cnt as u32)
        };

        if use_par {
            let w = n_workers.min((num_data / 64).max(1));
            get_pool(w).install(|| {
                out_ones
                    .par_iter_mut()
                    .zip(out_multi.par_iter_mut())
                    .enumerate()
                    .for_each(|(r, (so, sm))| {
                        let (co, cm_cnt) = count_row(r);
                        *so = co;
                        *sm = cm_cnt;
                    });
            });
        } else {
            for r in 0..num_data {
                let (co, cm_cnt) = count_row(r);
                out_ones[r] = co;
                out_multi[r] = cm_cnt;
            }
        }
    });

    if bad.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    if oob.load(Ordering::Relaxed) {
        return Err(PyIndexError::new_err("pixel index out of bounds"));
    }
    if overflow.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err(
            "Selected per-pattern count exceeds uint32",
        ));
    }

    let s_ones = unsafe { arr_ones.as_slice()? };
    let s_multi = unsafe { arr_multi.as_slice()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };
    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut acc_o = 0u64;
    let mut acc_m = 0u64;
    for r in 0..num_data {
        acc_o += s_ones[r] as u64;
        acc_m += s_multi[r] as u64;
        s_oidx[r + 1] = acc_o;
        s_midx[r + 1] = acc_m;
    }

    let out_ones_len = usize::try_from(acc_o)
        .map_err(|_| PyValueError::new_err("Selected place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(acc_m)
        .map_err(|_| PyValueError::new_err("Selected place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_out_ref: &[u64] = s_oidx;
    let midx_out_ref: &[u64] = s_midx;

    py.detach(|| {
        let scatter_row = |r: usize| {
            let o0 = oidx[r] as usize;
            let o1 = oidx[r + 1] as usize;
            let do_cur = oidx_out_ref[r] as usize;
            let do_end = oidx_out_ref[r + 1] as usize;
            if do_end > do_cur {
                let dst_po = unsafe {
                    std::slice::from_raw_parts_mut(
                        (po_addr as *mut u32).add(do_cur),
                        do_end - do_cur,
                    )
                };
                let mut k = 0usize;
                for &p in &po[o0..o1] {
                    let ts = targets_fn(p);
                    dst_po[k..k + ts.len()].copy_from_slice(ts);
                    k += ts.len();
                }
                debug_assert_eq!(k, dst_po.len());
            }

            let m0 = midx[r] as usize;
            let m1 = midx[r + 1] as usize;
            let dm_cur = midx_out_ref[r] as usize;
            let dm_end = midx_out_ref[r + 1] as usize;
            if dm_end > dm_cur {
                let dst_pm = unsafe {
                    std::slice::from_raw_parts_mut(
                        (pm_addr as *mut u32).add(dm_cur),
                        dm_end - dm_cur,
                    )
                };
                let dst_cm = unsafe {
                    std::slice::from_raw_parts_mut(
                        (cm_addr as *mut i32).add(dm_cur),
                        dm_end - dm_cur,
                    )
                };
                let mut k = 0usize;
                for (&p, &val) in pm[m0..m1].iter().zip(cm[m0..m1].iter()) {
                    let ts = targets_fn(p);
                    if !ts.is_empty() {
                        dst_pm[k..k + ts.len()].copy_from_slice(ts);
                        dst_cm[k..k + ts.len()].fill(val);
                        k += ts.len();
                    }
                }
                debug_assert_eq!(k, dst_pm.len());
            }
        };

        if use_par {
            let w = n_workers.min((num_data / 64).max(1));
            get_pool(w).install(|| {
                (0..num_data).into_par_iter().for_each(scatter_row);
            });
        } else {
            for r in 0..num_data {
                scatter_row(r);
            }
        }
    });

    Ok((
        col_count,
        (
            arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
        ),
    ))
}

// ============================================================================
// 4. Dense / COO / CSR conversion & index sorting
// ============================================================================

trait DensePhoton: Element + Copy + Sync {
    /// Classify value: <= 0 -> 0, == 1 -> 1, > 1 -> Some(i32) or Err if > i32::MAX.
    fn classify(self) -> Result<i32, ()>;
    fn is_non_positive(self) -> bool;
}

macro_rules! impl_dense_signed {
    ($($t:ty),*) => {
        $(
            impl DensePhoton for $t {
                #[inline(always)]
                fn classify(self) -> Result<i32, ()> {
                    if self <= 0 {
                        Ok(0)
                    } else if self == 1 {
                        Ok(1)
                    } else {
                        i32::try_from(self).map_err(|_| ())
                    }
                }
                #[inline(always)]
                fn is_non_positive(self) -> bool {
                    self <= 0
                }
            }
        )*
    };
}

macro_rules! impl_dense_unsigned {
    ($($t:ty),*) => {
        $(
            impl DensePhoton for $t {
                #[inline(always)]
                fn classify(self) -> Result<i32, ()> {
                    if self == 0 {
                        Ok(0)
                    } else if self == 1 {
                        Ok(1)
                    } else {
                        i32::try_from(self).map_err(|_| ())
                    }
                }
                #[inline(always)]
                fn is_non_positive(self) -> bool {
                    self == 0
                }
            }
        )*
    };
}

impl_dense_signed!(i8, i16, i32, i64);
impl_dense_unsigned!(u8, u16, u32, u64);

fn convert_dense_2d_typed<'py, T: DensePhoton>(
    py: Python<'py>,
    arr: PyReadonlyArray2<'py, T>,
    n_workers: usize,
) -> PyResult<Option<SizedFullPatternArrays<'py>>> {
    if !arr.is_c_contiguous() {
        return Ok(None);
    }
    let shape = arr.shape();
    if shape.len() != 2 {
        return Ok(None);
    }
    let num_data = shape[0];
    let num_pix = shape[1];
    if num_pix > (u32::MAX as usize) + 1 {
        return Err(PyValueError::new_err(
            "place_ones values exceed uint32 maximum",
        ));
    }
    let slice = arr.as_slice()?;

    let arr_ones = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_data], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [num_data + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [num_data + 1], false);

    if num_data == 0 || num_pix == 0 {
        let empty_po = PyArray1::<u32>::zeros(py, [0], false);
        let empty_pm = PyArray1::<u32>::zeros(py, [0], false);
        let empty_cm = PyArray1::<i32>::zeros(py, [0], false);
        return Ok(Some((
            num_pix,
            (
                arr_ones, arr_multi, arr_oidx, arr_midx, empty_po, empty_pm, empty_cm,
            ),
        )));
    }

    let ones_addr = unsafe { arr_ones.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { arr_multi.as_slice_mut()? }.as_mut_ptr() as usize;
    let total_cells = num_data.saturating_mul(num_pix);
    let use_par = n_workers > 1 && num_data >= 4 && total_cells >= 131_072;
    let val_overflow = AtomicBool::new(false);

    py.detach(|| {
        let out_ones = unsafe { std::slice::from_raw_parts_mut(ones_addr as *mut u32, num_data) };
        let out_multi = unsafe { std::slice::from_raw_parts_mut(multi_addr as *mut u32, num_data) };

        let count_row = |row: &[T]| -> (u32, u32) {
            let mut co = 0u32;
            let mut cm = 0u32;
            for blk in row.chunks(64) {
                if blk.iter().all(|&x| x.is_non_positive()) {
                    continue;
                }
                for &x in blk {
                    match x.classify() {
                        Ok(0) => {}
                        Ok(1) => co += 1,
                        Ok(_) => cm += 1,
                        Err(()) => {
                            val_overflow.store(true, Ordering::Relaxed);
                            return (0, 0);
                        }
                    }
                }
            }
            (co, cm)
        };

        if use_par {
            let w = n_workers.min((num_data / 2).max(1));
            get_pool(w).install(|| {
                slice
                    .par_chunks(num_pix)
                    .zip(out_ones.par_iter_mut().zip(out_multi.par_iter_mut()))
                    .for_each(|(row, (so, sm))| {
                        let (co, cm) = count_row(row);
                        *so = co;
                        *sm = cm;
                    });
            });
        } else {
            for (r, row) in slice.chunks(num_pix).enumerate() {
                let (co, cm) = count_row(row);
                out_ones[r] = co;
                out_multi[r] = cm;
            }
        }
    });

    if val_overflow.load(Ordering::Relaxed) {
        return Err(PyValueError::new_err(
            "count_multi values exceed int32 maximum",
        ));
    }

    let s_ones = unsafe { arr_ones.as_slice()? };
    let s_multi = unsafe { arr_multi.as_slice()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };
    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut acc_o = 0u64;
    let mut acc_m = 0u64;
    for r in 0..num_data {
        acc_o += s_ones[r] as u64;
        acc_m += s_multi[r] as u64;
        s_oidx[r + 1] = acc_o;
        s_midx[r + 1] = acc_m;
    }

    let out_ones_len = usize::try_from(acc_o)
        .map_err(|_| PyValueError::new_err("place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(acc_m)
        .map_err(|_| PyValueError::new_err("place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    if out_ones_len == 0 && out_multi_len == 0 {
        return Ok(Some((
            num_pix,
            (
                arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
            ),
        )));
    }

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_ref: &[u64] = s_oidx;
    let midx_ref: &[u64] = s_midx;

    py.detach(|| {
        let scatter_row = |r: usize, row: &[T]| {
            let o0 = oidx_ref[r] as usize;
            let o1 = oidx_ref[r + 1] as usize;
            let m0 = midx_ref[r] as usize;
            let m1 = midx_ref[r + 1] as usize;
            if o0 == o1 && m0 == m1 {
                return;
            }
            let dst_po =
                unsafe { std::slice::from_raw_parts_mut((po_addr as *mut u32).add(o0), o1 - o0) };
            let dst_pm =
                unsafe { std::slice::from_raw_parts_mut((pm_addr as *mut u32).add(m0), m1 - m0) };
            let dst_cm =
                unsafe { std::slice::from_raw_parts_mut((cm_addr as *mut i32).add(m0), m1 - m0) };
            let mut ko = 0usize;
            let mut km = 0usize;
            for (blk_idx, blk) in row.chunks(64).enumerate() {
                if blk.iter().all(|&x| x.is_non_positive()) {
                    continue;
                }
                let base_col = (blk_idx * 64) as u32;
                for (off, &x) in blk.iter().enumerate() {
                    if let Ok(v) = x.classify() {
                        if v == 1 {
                            dst_po[ko] = base_col + (off as u32);
                            ko += 1;
                        } else if v > 1 {
                            dst_pm[km] = base_col + (off as u32);
                            dst_cm[km] = v;
                            km += 1;
                        }
                    }
                }
            }
            debug_assert_eq!(ko, dst_po.len());
            debug_assert_eq!(km, dst_pm.len());
        };

        if use_par {
            let w = n_workers.min((num_data / 2).max(1));
            get_pool(w).install(|| {
                slice
                    .par_chunks(num_pix)
                    .enumerate()
                    .for_each(|(r, row)| scatter_row(r, row));
            });
        } else {
            for (r, row) in slice.chunks(num_pix).enumerate() {
                scatter_row(r, row);
            }
        }
    });

    Ok(Some((
        num_pix,
        (
            arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
        ),
    )))
}

#[pyfunction]
pub fn from_dense_2d<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    n_workers: usize,
) -> PyResult<Option<SizedFullPatternArrays<'py>>> {
    macro_rules! try_dense {
        ($($t:ty),*) => {
            $(
                if let Ok(a2) = arr.extract::<PyReadonlyArray2<'py, $t>>() {
                    return convert_dense_2d_typed(py, a2, n_workers);
                }
            )*
        };
    }
    try_dense!(i32, i64, u32, u16, i16, u8, i8, u64);
    Ok(None)
}

#[pyfunction]
pub fn from_coo<'py>(
    py: Python<'py>,
    num_rows: usize,
    num_pix: usize,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, i64>,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    if num_rows > (u32::MAX as usize) + 1 || num_pix > (u32::MAX as usize) + 1 {
        return Err(PyValueError::new_err(
            "COO shape exceeds uint32 coordinate range",
        ));
    }
    let s_rows = rows.as_slice()?;
    let s_cols = cols.as_slice()?;
    let s_data = data.as_slice()?;
    let nnz = s_rows.len();
    if s_cols.len() != nnz || s_data.len() != nnz {
        return Err(PyValueError::new_err(
            "COO row, col, and data lengths must match",
        ));
    }

    let arr_ones = PyArray1::<u32>::zeros(py, [num_rows], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_rows], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [num_rows + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [num_rows + 1], false);

    if nnz == 0 || num_rows == 0 || num_pix == 0 {
        for &r in s_rows {
            if r < 0 || (r as usize) >= num_rows {
                return Err(PyValueError::new_err("COO row index out of bounds"));
            }
        }
        for &c in s_cols {
            if c < 0 || (c as usize) >= num_pix {
                return Err(PyValueError::new_err("COO col index out of bounds"));
            }
        }
        let empty_po = PyArray1::<u32>::zeros(py, [0], false);
        let empty_pm = PyArray1::<u32>::zeros(py, [0], false);
        let empty_cm = PyArray1::<i32>::zeros(py, [0], false);
        return Ok((
            arr_ones, arr_multi, arr_oidx, arr_midx, empty_po, empty_pm, empty_cm,
        ));
    }

    let ones_slice = unsafe { arr_ones.as_slice_mut()? };
    let multi_slice = unsafe { arr_multi.as_slice_mut()? };
    let oidx_slice = unsafe { arr_oidx.as_slice_mut()? };
    let midx_slice = unsafe { arr_midx.as_slice_mut()? };

    let res: Result<CooEvents, i32> = py.detach(|| {
        let mut entries: Vec<(u64, i64)> = Vec::with_capacity(nnz);
        let mut is_sorted = true;
        let mut prev_key = 0u64;
        for i in 0..nnz {
            let r = s_rows[i];
            let c = s_cols[i];
            if r < 0 || (r as usize) >= num_rows || c < 0 || (c as usize) >= num_pix {
                return Err(-1);
            }
            let key = ((r as u64) << 32) | (c as u64);
            if i > 0 && key < prev_key {
                is_sorted = false;
            }
            prev_key = key;
            entries.push((key, s_data[i]));
        }

        if !is_sorted {
            if n_workers > 1 && entries.len() >= 131_072 {
                let w = n_workers.min(8);
                get_pool(w).install(|| {
                    entries.par_sort_unstable_by_key(|&(k, _)| k);
                });
            } else {
                entries.sort_unstable_by_key(|&(k, _)| k);
            }
        }

        let mut po_vec: Vec<u32> = Vec::new();
        let mut pm_vec: Vec<u32> = Vec::new();
        let mut cm_vec: Vec<i32> = Vec::new();

        let mut i = 0usize;
        while i < entries.len() {
            let key = entries[i].0;
            let mut sum_val: i128 = entries[i].1 as i128;
            i += 1;
            while i < entries.len() && entries[i].0 == key {
                sum_val += entries[i].1 as i128;
                i += 1;
            }
            if sum_val <= 0 {
                continue;
            }
            if sum_val > i32::MAX as i128 {
                return Err(-2);
            }
            let r = (key >> 32) as usize;
            let c = (key & 0xFFFF_FFFF) as u32;
            if sum_val == 1 {
                ones_slice[r] += 1;
                po_vec.push(c);
            } else {
                multi_slice[r] += 1;
                pm_vec.push(c);
                cm_vec.push(sum_val as i32);
            }
        }

        oidx_slice[0] = 0;
        midx_slice[0] = 0;
        let mut ao = 0u64;
        let mut am = 0u64;
        for r in 0..num_rows {
            ao += ones_slice[r] as u64;
            am += multi_slice[r] as u64;
            oidx_slice[r + 1] = ao;
            midx_slice[r + 1] = am;
        }

        Ok((po_vec, pm_vec, cm_vec))
    });

    match res {
        Ok((po_vec, pm_vec, cm_vec)) => {
            let arr_po = PyArray1::from_vec(py, po_vec);
            let arr_pm = PyArray1::from_vec(py, pm_vec);
            let arr_cm = PyArray1::from_vec(py, cm_vec);
            Ok((
                arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
            ))
        }
        Err(-1) => Err(PyValueError::new_err("COO index out of bounds")),
        _ => Err(PyValueError::new_err(
            "count_multi values exceed int32 maximum",
        )),
    }
}

#[pyfunction]
pub fn from_canonical_csr<'py>(
    py: Python<'py>,
    num_rows: usize,
    num_pix: usize,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i64>,
    data: PyReadonlyArray1<'py, i64>,
    n_workers: usize,
) -> PyResult<FullPatternArrays<'py>> {
    if num_pix > (u32::MAX as usize) + 1 {
        return Err(PyValueError::new_err(
            "CSR column count exceeds uint32 maximum",
        ));
    }
    let s_indptr = indptr.as_slice()?;
    let s_indices = indices.as_slice()?;
    let s_data = data.as_slice()?;
    if s_indptr.len() != num_rows + 1 {
        return Err(PyValueError::new_err("Invalid CSR indptr length"));
    }
    if s_indices.len() != s_data.len() {
        return Err(PyValueError::new_err(
            "CSR indices and data lengths must match",
        ));
    }
    let nnz = s_data.len();

    let arr_ones = PyArray1::<u32>::zeros(py, [num_rows], false);
    let arr_multi = PyArray1::<u32>::zeros(py, [num_rows], false);
    let arr_oidx = PyArray1::<u64>::zeros(py, [num_rows + 1], false);
    let arr_midx = PyArray1::<u64>::zeros(py, [num_rows + 1], false);

    let ones_addr = unsafe { arr_ones.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { arr_multi.as_slice_mut()? }.as_mut_ptr() as usize;
    let err_code = std::sync::atomic::AtomicI32::new(0);
    let use_par = n_workers > 1 && num_rows >= 256 && nnz >= 262_144;

    py.detach(|| {
        let out_ones = unsafe { std::slice::from_raw_parts_mut(ones_addr as *mut u32, num_rows) };
        let out_multi = unsafe { std::slice::from_raw_parts_mut(multi_addr as *mut u32, num_rows) };

        let count_row = |r: usize| -> (u32, u32) {
            let b = s_indptr[r];
            let e = s_indptr[r + 1];
            if b < 0 || e < b || (e as usize) > nnz {
                err_code.store(-1, Ordering::Relaxed);
                return (0, 0);
            }
            let mut co = 0u32;
            let mut cm = 0u32;
            for idx in (b as usize)..(e as usize) {
                let col = s_indices[idx];
                if col < 0 || (col as usize) >= num_pix {
                    err_code.store(-1, Ordering::Relaxed);
                    return (0, 0);
                }
                let val = s_data[idx];
                if val == 1 {
                    co += 1;
                } else if val > 1 {
                    if val > i32::MAX as i64 {
                        err_code.store(-2, Ordering::Relaxed);
                        return (0, 0);
                    }
                    cm += 1;
                }
            }
            (co, cm)
        };

        if use_par {
            let w = n_workers.min((num_rows / 64).max(1));
            get_pool(w).install(|| {
                out_ones
                    .par_iter_mut()
                    .zip(out_multi.par_iter_mut())
                    .enumerate()
                    .for_each(|(r, (so, sm))| {
                        let (co, cm) = count_row(r);
                        *so = co;
                        *sm = cm;
                    });
            });
        } else {
            for r in 0..num_rows {
                let (co, cm) = count_row(r);
                out_ones[r] = co;
                out_multi[r] = cm;
            }
        }
    });

    match err_code.load(Ordering::Relaxed) {
        0 => {}
        -2 => {
            return Err(PyValueError::new_err(
                "count_multi values exceed int32 maximum",
            ))
        }
        _ => return Err(PyValueError::new_err("Invalid CSR structure or bounds")),
    }

    let s_ones = unsafe { arr_ones.as_slice()? };
    let s_multi = unsafe { arr_multi.as_slice()? };
    let s_oidx = unsafe { arr_oidx.as_slice_mut()? };
    let s_midx = unsafe { arr_midx.as_slice_mut()? };
    s_oidx[0] = 0;
    s_midx[0] = 0;
    let mut ao = 0u64;
    let mut am = 0u64;
    for r in 0..num_rows {
        ao += s_ones[r] as u64;
        am += s_multi[r] as u64;
        s_oidx[r + 1] = ao;
        s_midx[r + 1] = am;
    }

    let out_ones_len = usize::try_from(ao)
        .map_err(|_| PyValueError::new_err("place_ones size overflows usize"))?;
    let out_multi_len = usize::try_from(am)
        .map_err(|_| PyValueError::new_err("place_multi size overflows usize"))?;

    let arr_po = PyArray1::<u32>::zeros(py, [out_ones_len], false);
    let arr_pm = PyArray1::<u32>::zeros(py, [out_multi_len], false);
    let arr_cm = PyArray1::<i32>::zeros(py, [out_multi_len], false);

    let po_addr = unsafe { arr_po.as_slice_mut()? }.as_mut_ptr() as usize;
    let pm_addr = unsafe { arr_pm.as_slice_mut()? }.as_mut_ptr() as usize;
    let cm_addr = unsafe { arr_cm.as_slice_mut()? }.as_mut_ptr() as usize;
    let oidx_ref: &[u64] = s_oidx;
    let midx_ref: &[u64] = s_midx;

    py.detach(|| {
        let scatter_row = |r: usize| {
            let b = s_indptr[r] as usize;
            let e = s_indptr[r + 1] as usize;
            let o0 = oidx_ref[r] as usize;
            let o1 = oidx_ref[r + 1] as usize;
            let m0 = midx_ref[r] as usize;
            let m1 = midx_ref[r + 1] as usize;
            if o0 == o1 && m0 == m1 {
                return;
            }
            let dst_po =
                unsafe { std::slice::from_raw_parts_mut((po_addr as *mut u32).add(o0), o1 - o0) };
            let dst_pm =
                unsafe { std::slice::from_raw_parts_mut((pm_addr as *mut u32).add(m0), m1 - m0) };
            let dst_cm =
                unsafe { std::slice::from_raw_parts_mut((cm_addr as *mut i32).add(m0), m1 - m0) };
            let mut ko = 0usize;
            let mut km = 0usize;
            for idx in b..e {
                let val = s_data[idx];
                if val == 1 {
                    dst_po[ko] = s_indices[idx] as u32;
                    ko += 1;
                } else if val > 1 {
                    dst_pm[km] = s_indices[idx] as u32;
                    dst_cm[km] = val as i32;
                    km += 1;
                }
            }
            debug_assert_eq!(ko, dst_po.len());
            debug_assert_eq!(km, dst_pm.len());
        };

        if use_par {
            let w = n_workers.min((num_rows / 64).max(1));
            get_pool(w).install(|| {
                (0..num_rows).into_par_iter().for_each(scatter_row);
            });
        } else {
            for r in 0..num_rows {
                scatter_row(r);
            }
        }
    });

    Ok((
        arr_ones, arr_multi, arr_oidx, arr_midx, arr_po, arr_pm, arr_cm,
    ))
}

#[inline]
fn stream_strictly_increasing(
    values: &[u32],
    offsets: &[u64],
    n_workers: usize,
) -> Result<bool, ()> {
    if offsets.is_empty() || offsets[0] != 0 {
        return Err(());
    }
    let num_data = offsets.len() - 1;
    let total = values.len();
    if offsets[num_data] != total as u64 {
        return Err(());
    }
    if total <= 1 {
        return Ok(true);
    }

    let check_row = |r: usize| -> Result<bool, ()> {
        let s = offsets[r] as usize;
        let e = offsets[r + 1] as usize;
        if e < s || e > total {
            return Err(());
        }
        if e - s > 1 {
            for w in values[s..e].windows(2) {
                if w[1] <= w[0] {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    };

    if n_workers > 1 && num_data >= 1024 && total >= 524_288 {
        let bad = AtomicBool::new(false);
        let w = n_workers.min((num_data / 256).max(1));
        let sorted = get_pool(w).install(|| {
            (0..num_data).into_par_iter().all(|r| match check_row(r) {
                Ok(v) => v,
                Err(()) => {
                    bad.store(true, Ordering::Relaxed);
                    false
                }
            })
        });
        if bad.load(Ordering::Relaxed) {
            Err(())
        } else {
            Ok(sorted)
        }
    } else {
        for r in 0..num_data {
            if !check_row(r)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[pyfunction]
pub fn has_sorted_indices(
    py: Python<'_>,
    ones_idx: PyReadonlyArray1<'_, u64>,
    place_ones: PyReadonlyArray1<'_, u32>,
    multi_idx: PyReadonlyArray1<'_, u64>,
    place_multi: PyReadonlyArray1<'_, u32>,
    n_workers: usize,
) -> PyResult<bool> {
    let oidx = ones_idx.as_slice()?;
    let po = place_ones.as_slice()?;
    let midx = multi_idx.as_slice()?;
    let pm = place_multi.as_slice()?;

    let res = py.detach(|| -> Result<bool, ()> {
        if !stream_strictly_increasing(pm, midx, n_workers)? {
            return Ok(false);
        }
        stream_strictly_increasing(po, oidx, n_workers)
    });

    res.map_err(|()| PyValueError::new_err("Invalid pattern offsets array"))
}

#[pyfunction]
pub fn sort_indices_inplace<'py>(
    py: Python<'py>,
    ones_idx: PyReadonlyArray1<'py, u64>,
    mut place_ones: PyReadwriteArray1<'py, u32>,
    multi_idx: PyReadonlyArray1<'py, u64>,
    mut place_multi: PyReadwriteArray1<'py, u32>,
    mut count_multi: PyReadwriteArray1<'py, i32>,
    n_workers: usize,
) -> PyResult<()> {
    let oidx = ones_idx.as_slice()?;
    let midx = multi_idx.as_slice()?;
    if oidx.is_empty() || oidx.len() != midx.len() {
        return Err(PyValueError::new_err("Invalid pattern offsets array"));
    }
    let num_data = oidx.len() - 1;
    let total_ones = check_offsets_bounds(oidx, num_data)?;
    let total_multi = check_offsets_bounds(midx, num_data)?;

    // Validate monotonic non-decreasing offsets before creating disjoint row slices.
    for w in oidx.windows(2) {
        if w[1] < w[0] {
            return Err(PyValueError::new_err("ones_idx must be non-decreasing"));
        }
    }
    for w in midx.windows(2) {
        if w[1] < w[0] {
            return Err(PyValueError::new_err("multi_idx must be non-decreasing"));
        }
    }

    let po = place_ones.as_slice_mut()?;
    let pm = place_multi.as_slice_mut()?;
    let cm = count_multi.as_slice_mut()?;
    if po.len() != total_ones || pm.len() != total_multi || cm.len() != total_multi {
        return Err(PyValueError::new_err(
            "Event arrays do not match pattern offset totals",
        ));
    }

    let po_addr = po.as_mut_ptr() as usize;
    let pm_addr = pm.as_mut_ptr() as usize;
    let cm_addr = cm.as_mut_ptr() as usize;

    py.detach(|| {
        let sort_chunk = |r_start: usize, r_end: usize| {
            let mut scratch: Vec<(u32, i32)> = Vec::new();
            for r in r_start..r_end {
                let o0 = oidx[r] as usize;
                let o1 = oidx[r + 1] as usize;
                if o1 - o0 > 1 {
                    let s_po = unsafe {
                        std::slice::from_raw_parts_mut((po_addr as *mut u32).add(o0), o1 - o0)
                    };
                    if !s_po.windows(2).all(|w| w[0] <= w[1]) {
                        s_po.sort_unstable();
                    }
                }
                let m0 = midx[r] as usize;
                let m1 = midx[r + 1] as usize;
                let len_m = m1 - m0;
                if len_m > 1 {
                    let s_pm = unsafe {
                        std::slice::from_raw_parts_mut((pm_addr as *mut u32).add(m0), len_m)
                    };
                    if !s_pm.windows(2).all(|w| w[0] <= w[1]) {
                        let s_cm = unsafe {
                            std::slice::from_raw_parts_mut((cm_addr as *mut i32).add(m0), len_m)
                        };
                        scratch.clear();
                        scratch.extend(s_pm.iter().copied().zip(s_cm.iter().copied()));
                        scratch.sort_by_key(|&(p, _)| p);
                        for (k, &(p, c)) in scratch.iter().enumerate() {
                            s_pm[k] = p;
                            s_cm[k] = c;
                        }
                    }
                }
            }
        };

        let total_events = total_ones + total_multi;
        if n_workers > 1 && num_data >= 512 && total_events >= 65_536 {
            let w = n_workers.min((num_data / 128).max(1));
            let chunk_rows = num_data.div_ceil(w);
            get_pool(w).install(|| {
                (0..w).into_par_iter().for_each(|tid| {
                    let rs = (tid * chunk_rows).min(num_data);
                    let re = ((tid + 1) * chunk_rows).min(num_data);
                    if rs < re {
                        sort_chunk(rs, re);
                    }
                });
            });
        } else {
            sort_chunk(0, num_data);
        }
    });

    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emc_read_counts_and_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(emc_read_dense_frame, m)?)?;
    m.add_function(wrap_pyfunction!(emc_read_sparse_frame, m)?)?;
    m.add_function(wrap_pyfunction!(emc_read_slice, m)?)?;
    m.add_function(wrap_pyfunction!(emc_read_strided, m)?)?;
    m.add_function(wrap_pyfunction!(emc_read_indexed, m)?)?;
    m.add_function(wrap_pyfunction!(emc_read_mask, m)?)?;
    m.add_function(wrap_pyfunction!(emc_extract_column, m)?)?;
    m.add_function(wrap_pyfunction!(emc_sum_total, m)?)?;
    m.add_function(wrap_pyfunction!(emc_sum_axis1, m)?)?;
    m.add_function(wrap_pyfunction!(emc_sum_axis0, m)?)?;
    m.add_function(wrap_pyfunction!(emc_write_binary, m)?)?;

    m.add_function(wrap_pyfunction!(sum_total_i128, m)?)?;
    m.add_function(wrap_pyfunction!(sum_axis1_i64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_axis0_i64, m)?)?;
    m.add_function(wrap_pyfunction!(todense_2d, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_dense, m)?)?;

    m.add_function(wrap_pyfunction!(gather_rows_ids, m)?)?;
    m.add_function(wrap_pyfunction!(gather_rows_strided, m)?)?;
    m.add_function(wrap_pyfunction!(gather_rows_mask, m)?)?;
    m.add_function(wrap_pyfunction!(extract_column_1d, m)?)?;
    m.add_function(wrap_pyfunction!(select_columns_slice, m)?)?;
    m.add_function(wrap_pyfunction!(select_columns_mask, m)?)?;
    m.add_function(wrap_pyfunction!(select_columns_indices, m)?)?;

    m.add_function(wrap_pyfunction!(from_dense_2d, m)?)?;
    m.add_function(wrap_pyfunction!(from_coo, m)?)?;
    m.add_function(wrap_pyfunction!(from_canonical_csr, m)?)?;
    m.add_function(wrap_pyfunction!(has_sorted_indices, m)?)?;
    m.add_function(wrap_pyfunction!(sort_indices_inplace, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_len_positive_and_negative() {
        assert_eq!(strided_len(0, 10, 1), 10);
        assert_eq!(strided_len(0, 10, 2), 5);
        assert_eq!(strided_len(0, 9, 2), 5);
        assert_eq!(strided_len(5, 5, 1), 0);
        assert_eq!(strided_len(8, 2, 1), 0);
        assert_eq!(strided_len(9, -1, -1), 10);
        assert_eq!(strided_len(9, 0, -2), 5);
        assert_eq!(strided_len(2, 8, -1), 0);
    }

    #[test]
    fn test_validate_emc_buffer_detects_truncation_and_trailing_bytes() {
        // 2 patterns, total_ones = 3, total_multi = 1 -> 1024 + 16 + 12 + 4 + 4 = 1060 bytes
        let ok = validate_emc_buffer(1060, 2, 3, 1);
        assert!(ok.is_ok());
        assert!(validate_emc_buffer(1059, 2, 3, 1).is_err());
        assert!(validate_emc_buffer(1061, 2, 3, 1).is_err());
    }

    #[test]
    fn test_stream_strictly_increasing() {
        assert_eq!(
            stream_strictly_increasing(&[2, 5, 1, 4], &[0, 2, 4], 1),
            Ok(true)
        );
        // Duplicate in same pattern is not strictly increasing
        assert_eq!(
            stream_strictly_increasing(&[2, 2, 1, 4], &[0, 2, 4], 1),
            Ok(false)
        );
        // Empty leading and trailing patterns
        assert_eq!(
            stream_strictly_increasing(&[5, 2], &[0, 0, 2, 2], 1),
            Ok(false)
        );
        assert_eq!(
            stream_strictly_increasing(&[2, 5], &[0, 0, 2, 2], 1),
            Ok(true)
        );
    }
}
