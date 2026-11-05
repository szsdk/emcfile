mod simd;

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray3};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use simd::{decode_pattern_u32, encode_pattern_u32};

struct ThreadZstd {
    cctx: zstd_safe::CCtx<'static>,
    dctx: zstd_safe::DCtx<'static>,
    scratch: Vec<u8>,
    comp_scratch: Vec<u8>,
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
    fn ensure_scratch(&mut self, n: usize) {
        if self.scratch.len() < n {
            self.scratch.resize(n, 0);
        }
    }

    #[inline]
    fn ensure_comp_scratch(&mut self, n: usize) {
        if self.comp_scratch.len() < n {
            self.comp_scratch.resize(n, 0);
        }
    }
}

thread_local! {
    static THREAD_ZSTD: RefCell<ThreadZstd> = RefCell::new(ThreadZstd::new());
}

fn get_pool(n_threads: usize) -> Arc<rayon::ThreadPool> {
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
fn write_uvarint(buf: &mut [u8], mut pos: usize, mut val: u64) -> usize {
    while val >= 0x80 {
        buf[pos] = ((val as u8) & 0x7F) | 0x80;
        pos += 1;
        val >>= 7;
    }
    buf[pos] = val as u8;
    pos + 1
}

#[inline]
fn write_zigzag_i32(buf: &mut [u8], pos: usize, n: i32) -> usize {
    let zz = ((n << 1) ^ (n >> 31)) as u32 as u64;
    write_uvarint(buf, pos, zz)
}

#[inline]
fn read_zigzag(buf: &[u8], mut pos: usize) -> (i64, usize) {
    let b0 = buf[pos] as u64;
    if b0 < 0x80 {
        return (((b0 >> 1) as i64) ^ -((b0 & 1) as i64), pos + 1);
    }
    let b1 = buf[pos + 1] as u64;
    if b1 < 0x80 {
        let n = (b0 & 0x7F) | (b1 << 7);
        return (((n >> 1) as i64) ^ -((n & 1) as i64), pos + 2);
    }
    let mut n = (b0 & 0x7F) | ((b1 & 0x7F) << 7);
    let mut shift = 14u32;
    pos += 2;
    loop {
        let b = buf[pos] as u64;
        pos += 1;
        n |= (b & 0x7F) << shift;
        if b < 0x80 {
            return (((n >> 1) as i64) ^ -((n & 1) as i64), pos);
        }
        shift += 7;
    }
}

#[inline]
fn write_page_header(
    out_buf: &mut [u8],
    mut pos: usize,
    u_len: usize,
    c_len: usize,
    num_vals: usize,
) -> usize {
    out_buf[pos] = 0x15;
    out_buf[pos + 1] = 0x00;
    out_buf[pos + 2] = 0x15;
    pos = write_zigzag_i32(out_buf, pos + 3, u_len as i32);
    out_buf[pos] = 0x15;
    pos = write_zigzag_i32(out_buf, pos + 1, c_len as i32);
    out_buf[pos] = 0x2C;
    out_buf[pos + 1] = 0x15;
    pos = write_zigzag_i32(out_buf, pos + 2, num_vals as i32);
    out_buf[pos..pos + 9].copy_from_slice(&[0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x1C, 0x00, 0x00]);
    out_buf[pos + 9] = 0x00;
    pos + 10
}

#[allow(clippy::too_many_arguments)]
fn encode_payload_col_pages(
    cctx: &mut zstd_safe::CCtx<'static>,
    level: i32,
    src_u32: &[u32],
    pat_offsets: &[u64],
    ps: usize,
    pe: usize,
    page_pats: usize,
    scratch: &mut [u8],
    comp_scratch: &mut [u8],
    out_buf: &mut [u8],
    mut pos: usize,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
) -> Result<(usize, i64, i64), i32> {
    let mut tot_u = 0i64;
    let mut tot_c = 0i64;
    let mut p0 = ps;
    while p0 < pe {
        let p1 = (p0 + page_pats).min(pe);
        let page_nr = p1 - p0;
        let page_elems = (pat_offsets[p1] - pat_offsets[p0]) as usize;
        let u_len = (page_nr + page_elems) * 4;
        let mut b_off = 0usize;
        for p in p0..p1 {
            let s = pat_offsets[p] as usize;
            let e = pat_offsets[p + 1] as usize;
            let k = e - s;
            let byte_len = (k * 4) as u32;
            scratch[b_off..b_off + 4].copy_from_slice(&byte_len.to_le_bytes());
            b_off += 4;
            if k > 0 {
                let rc = encode_pattern_u32(
                    &src_u32[s..e],
                    scratch,
                    b_off,
                    is_delta,
                    is_shuffled,
                    check_sorted,
                );
                if rc != 0 {
                    return Err(rc);
                }
                b_off += k * 4;
            }
        }
        let c_len = cctx
            .compress(comp_scratch, &scratch[..u_len], level)
            .map_err(|_| -3i32)?;
        let pos0 = pos;
        let hdr_end = write_page_header(out_buf, pos, u_len, c_len, page_nr);
        out_buf[hdr_end..hdr_end + c_len].copy_from_slice(&comp_scratch[..c_len]);
        pos = hdr_end + c_len;
        let hdr_len = hdr_end - pos0;
        tot_u += (hdr_len + u_len) as i64;
        tot_c += (hdr_len + c_len) as i64;
        p0 = p1;
    }
    Ok((pos, tot_u, tot_c))
}

#[allow(clippy::too_many_arguments)]
fn encode_rg_into_vec(
    rg: usize,
    num_data: usize,
    rg_pats: usize,
    page_pats: usize,
    level: i32,
    po_src: &[u32],
    pm_src: &[u32],
    cm_src: &[u32],
    ones_idx: &[u64],
    multi_idx: &[u64],
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
    out_vec: &mut Vec<u8>,
    meta_rg: &mut [i64],
) -> i32 {
    let ps = rg * rg_pats;
    let pe = ((rg + 1) * rg_pats).min(num_data);
    let nr = pe - ps;
    let no = (ones_idx[pe] - ones_idx[ps]) as usize;
    let nm = (multi_idx[pe] - multi_idx[ps]) as usize;

    let mut max_page_elems = 0usize;
    let mut p0 = ps;
    while p0 < pe {
        let p1 = (p0 + page_pats).min(pe);
        let po_e = (ones_idx[p1] - ones_idx[p0]) as usize;
        let pm_e = (multi_idx[p1] - multi_idx[p0]) as usize;
        max_page_elems = max_page_elems.max(po_e).max(pm_e);
        p0 = p1;
    }

    let need_sc = ((max_page_elems + page_pats) * 4) + 64;
    let need_csc = zstd_safe::compress_bound(need_sc) + 64;
    let n_pages = (nr + page_pats - 1) / page_pats;
    let need_out = zstd_safe::compress_bound((no + nr) * 4)
        + zstd_safe::compress_bound((nm + nr) * 4) * 2
        + (n_pages * 3 + 2) * 64;

    if out_vec.len() < need_out {
        out_vec.resize(need_out, 0);
    }

    THREAD_ZSTD.with(|cell| {
        let mut ctx = cell.borrow_mut();
        ctx.ensure_scratch(need_sc);
        ctx.ensure_comp_scratch(need_csc);
        let zctx = &mut *ctx;
        let cctx = &mut zctx.cctx;
        let sc = &mut zctx.scratch[..need_sc];
        let csc = &mut zctx.comp_scratch[..need_csc];

        let mut pos = 0usize;
        match encode_payload_col_pages(
            cctx,
            level,
            po_src,
            ones_idx,
            ps,
            pe,
            page_pats,
            sc,
            csc,
            out_vec,
            pos,
            is_delta,
            is_shuffled,
            check_sorted,
        ) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[0] = tu;
                meta_rg[1] = tc;
                meta_rg[2] = no as i64;
            }
            Err(e) => return e,
        }
        match encode_payload_col_pages(
            cctx,
            level,
            pm_src,
            multi_idx,
            ps,
            pe,
            page_pats,
            sc,
            csc,
            out_vec,
            pos,
            is_delta,
            is_shuffled,
            check_sorted,
        ) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[3] = tu;
                meta_rg[4] = tc;
                meta_rg[5] = nm as i64;
            }
            Err(e) => return e,
        }
        match encode_payload_col_pages(
            cctx,
            level,
            cm_src,
            multi_idx,
            ps,
            pe,
            page_pats,
            sc,
            csc,
            out_vec,
            pos,
            false,
            is_shuffled,
            false,
        ) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[6] = tu;
                meta_rg[7] = tc;
                meta_rg[8] = nm as i64;
            }
            Err(e) => return e,
        }
        pos as i32
    })
}

fn write_all_fd(fd: i32, buf: &[u8]) -> bool {
    let mut f = ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(fd) });
    f.write_all(buf).is_ok()
}

#[allow(clippy::too_many_arguments)]
fn count_col_pages_inner(
    dctx: &mut zstd_safe::DCtx<'static>,
    mm_u8: &[u8],
    col_off: usize,
    col_tc: usize,
    col_elems: usize,
    page_buf: &mut Vec<u8>,
    rg_ps: usize,
    rg_pe: usize,
    p_firsts: &[i64],
    p_lasts: &[i64],
    span_counts: &mut [i64],
) -> i32 {
    let n_spans = p_firsts.len();
    if n_spans == 0 {
        return 0;
    }
    if col_elems == 0 {
        span_counts.fill(0);
        return 0;
    }
    let mut any_partial = false;
    for s_i in 0..n_spans {
        if (p_firsts[s_i] as usize) == rg_ps && (p_lasts[s_i] as usize) == rg_pe {
            span_counts[s_i] = col_elems as i64;
        } else {
            span_counts[s_i] = 0;
            any_partial = true;
        }
    }
    if !any_partial {
        return 0;
    }

    let min_pat = p_firsts[0] as usize;
    let max_pat = p_lasts.iter().map(|&x| x as usize).max().unwrap_or(rg_pe);

    let mut pos = col_off;
    let col_end = col_off + col_tc;
    let mut page_ps = rg_ps;
    let mut sp_start = 0usize;

    while page_ps < rg_pe && pos < col_end {
        let (u_len_i, p1) = read_zigzag(mm_u8, pos + 3);
        let (c_len_i, p2) = read_zigzag(mm_u8, p1 + 1);
        let (page_nr_i, p3) = read_zigzag(mm_u8, p2 + 2);
        let u_len = u_len_i as usize;
        let c_len = c_len_i as usize;
        let page_nr = page_nr_i as usize;
        let hdr_len = (p3 + 10) - pos;
        let page_pe = page_ps + page_nr;
        let page_elems = (u_len / 4).saturating_sub(page_nr);

        if page_ps >= max_pat {
            break;
        }
        while sp_start < n_spans && (p_lasts[sp_start] as usize) <= page_ps {
            sp_start += 1;
        }
        if sp_start >= n_spans {
            break;
        }

        if page_elems > 0 && page_pe > min_pat {
            let mut need_decomp = false;
            let mut has_overlap = false;
            let mut s_idx = sp_start;
            while s_idx < n_spans && (p_firsts[s_idx] as usize) < page_pe {
                let sp_f = p_firsts[s_idx] as usize;
                let sp_l = p_lasts[s_idx] as usize;
                if !(sp_f == rg_ps && sp_l == rg_pe) && sp_l > page_ps {
                    has_overlap = true;
                    if sp_f > page_ps || sp_l < page_pe {
                        need_decomp = true;
                        break;
                    }
                }
                s_idx += 1;
            }

            if has_overlap {
                if !need_decomp {
                    let mut s_i = sp_start;
                    while s_i < n_spans && (p_firsts[s_i] as usize) < page_pe {
                        let sp_f = p_firsts[s_i] as usize;
                        let sp_l = p_lasts[s_i] as usize;
                        if !(sp_f == rg_ps && sp_l == rg_pe) && sp_l > page_ps {
                            span_counts[s_i] += page_elems as i64;
                        }
                        s_i += 1;
                    }
                } else {
                    if page_buf.len() < u_len {
                        page_buf.resize(u_len.max(256 * 1024), 0);
                    }
                    let c_start = pos + hdr_len;
                    match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len])
                    {
                        Ok(d_sz) if d_sz == u_len => {}
                        _ => return -1,
                    }
                    let page_slice = &page_buf[..u_len];
                    let mut b_off = 0usize;
                    for p_idx in page_ps..page_pe {
                        let bl =
                            u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap())
                                as usize;
                        let k = (bl / 4) as i64;
                        b_off += 4 + bl;
                        let mut s_i = sp_start;
                        while s_i < n_spans && (p_firsts[s_i] as usize) <= p_idx {
                            let sp_f = p_firsts[s_i] as usize;
                            let sp_l = p_lasts[s_i] as usize;
                            if !(sp_f == rg_ps && sp_l == rg_pe) && p_idx < sp_l {
                                span_counts[s_i] += k;
                            }
                            s_i += 1;
                        }
                    }
                }
            }
        }

        pos += hdr_len + c_len;
        page_ps = page_pe;
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn decode_col_pages_inner(
    dctx: &mut zstd_safe::DCtx<'static>,
    mm_u8: &[u8],
    col_off: usize,
    col_tc: usize,
    col_elems: usize,
    page_buf: &mut Vec<u8>,
    rg_ps: usize,
    rg_pe: usize,
    out_u32: &mut [u32],
    mut out_counts: Option<&mut [u32]>,
    p_firsts: &[i64],
    p_lasts: &[i64],
    p_dst_pats: &[i64],
    dst_offs: &[i64],
    cur_dst: &mut [usize],
    is_delta: bool,
    is_shuffled: bool,
) -> i32 {
    let n_spans = p_firsts.len();
    if n_spans == 0 {
        return 0;
    }
    if col_elems == 0 {
        if let Some(ref mut counts) = out_counts {
            for s_i in 0..n_spans {
                let sp_f = p_firsts[s_i] as usize;
                let sp_l = p_lasts[s_i] as usize;
                let dp = p_dst_pats[s_i] as usize;
                counts[dp..dp + (sp_l - sp_f)].fill(0);
            }
        }
        return 0;
    }

    for s_i in 0..n_spans {
        cur_dst[s_i] = dst_offs[s_i] as usize;
    }

    let min_pat = p_firsts[0] as usize;
    let max_pat = p_lasts.iter().map(|&x| x as usize).max().unwrap_or(rg_pe);

    let mut pos = col_off;
    let col_end = col_off + col_tc;
    let mut page_ps = rg_ps;
    let mut sp_start = 0usize;

    while page_ps < rg_pe && pos < col_end {
        let (u_len_i, p1) = read_zigzag(mm_u8, pos + 3);
        let (c_len_i, p2) = read_zigzag(mm_u8, p1 + 1);
        let (page_nr_i, p3) = read_zigzag(mm_u8, p2 + 2);
        let u_len = u_len_i as usize;
        let c_len = c_len_i as usize;
        let page_nr = page_nr_i as usize;
        let hdr_len = (p3 + 10) - pos;
        let page_pe = page_ps + page_nr;
        let page_elems = (u_len / 4).saturating_sub(page_nr);

        if page_ps >= max_pat {
            break;
        }
        while sp_start < n_spans && (p_lasts[sp_start] as usize) <= page_ps {
            sp_start += 1;
        }
        if sp_start >= n_spans {
            break;
        }

        if page_pe > min_pat {
            let mut overlaps = false;
            let mut s_idx = sp_start;
            while s_idx < n_spans && (p_firsts[s_idx] as usize) < page_pe {
                if (p_lasts[s_idx] as usize) > page_ps {
                    overlaps = true;
                    break;
                }
                s_idx += 1;
            }

            if overlaps {
                if page_elems == 0 {
                    if let Some(ref mut counts) = out_counts {
                        let mut s_i = sp_start;
                        while s_i < n_spans && (p_firsts[s_i] as usize) < page_pe {
                            let sp_f = p_firsts[s_i] as usize;
                            let sp_l = p_lasts[s_i] as usize;
                            let f = sp_f.max(page_ps);
                            let l = sp_l.min(page_pe);
                            if l > f {
                                let dp = (p_dst_pats[s_i] as usize) + (f - sp_f);
                                counts[dp..dp + (l - f)].fill(0);
                            }
                            s_i += 1;
                        }
                    }
                } else {
                    if page_buf.len() < u_len {
                        page_buf.resize(u_len.max(256 * 1024), 0);
                    }
                    let c_start = pos + hdr_len;
                    match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len])
                    {
                        Ok(d_sz) if d_sz == u_len => {}
                        _ => return -1,
                    }
                    let page_slice = &page_buf[..u_len];
                    let mut b_off = 0usize;
                    for p_idx in page_ps..page_pe {
                        let bl =
                            u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap())
                                as usize;
                        let k = bl / 4;
                        let b_s = b_off + 4;
                        b_off = b_s + bl;

                        let mut s_i = sp_start;
                        while s_i < n_spans && (p_firsts[s_i] as usize) <= p_idx {
                            let sp_f = p_firsts[s_i] as usize;
                            let sp_l = p_lasts[s_i] as usize;
                            if p_idx < sp_l {
                                if let Some(ref mut counts) = out_counts {
                                    counts[(p_dst_pats[s_i] as usize) + (p_idx - sp_f)] = k as u32;
                                }
                                if k > 0 {
                                    let out_idx = cur_dst[s_i];
                                    decode_pattern_u32(
                                        page_slice,
                                        b_s,
                                        &mut out_u32[out_idx..out_idx + k],
                                        is_delta,
                                        is_shuffled,
                                    );
                                    cur_dst[s_i] = out_idx + k;
                                }
                            }
                            s_i += 1;
                        }
                    }
                }
            }
        }

        pos += hdr_len + c_len;
        page_ps = page_pe;
    }
    0
}

#[allow(clippy::too_many_arguments)]
fn decode_single_col_frame(
    dctx: &mut zstd_safe::DCtx<'static>,
    mm_u8: &[u8],
    col_off: usize,
    col_tc: usize,
    col_elems: usize,
    page_buf: &mut Vec<u8>,
    rg_ps: usize,
    rg_pe: usize,
    frame: usize,
    is_delta: bool,
    is_shuffled: bool,
) -> Result<Vec<u32>, i32> {
    if col_elems == 0 {
        return Ok(Vec::new());
    }
    let mut pos = col_off;
    let col_end = col_off + col_tc;
    let mut page_ps = rg_ps;

    while page_ps < rg_pe && pos < col_end {
        let (u_len_i, p1) = read_zigzag(mm_u8, pos + 3);
        let (c_len_i, p2) = read_zigzag(mm_u8, p1 + 1);
        let (page_nr_i, p3) = read_zigzag(mm_u8, p2 + 2);
        let u_len = u_len_i as usize;
        let c_len = c_len_i as usize;
        let page_nr = page_nr_i as usize;
        let hdr_len = (p3 + 10) - pos;
        let page_pe = page_ps + page_nr;

        if frame < page_pe {
            let page_elems = (u_len / 4).saturating_sub(page_nr);
            if page_elems == 0 {
                return Ok(Vec::new());
            }
            if page_buf.len() < u_len {
                page_buf.resize(u_len.max(256 * 1024), 0);
            }
            let c_start = pos + hdr_len;
            match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len]) {
                Ok(d_sz) if d_sz == u_len => {}
                _ => return Err(-1),
            }
            let page_slice = &page_buf[..u_len];
            let mut b_off = 0usize;
            for _ in page_ps..frame {
                let bl =
                    u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
                b_off += 4 + bl;
            }
            let bl = u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
            let k = bl / 4;
            if k == 0 {
                return Ok(Vec::new());
            }
            let mut out = Vec::with_capacity(k);
            unsafe {
                out.set_len(k);
            }
            decode_pattern_u32(page_slice, b_off + 4, &mut out[..k], is_delta, is_shuffled);
            return Ok(out);
        }

        pos += hdr_len + c_len;
        page_ps = page_pe;
    }
    Ok(Vec::new())
}

#[inline]
fn view_pybuffer_u8(buf: &PyBuffer<u8>) -> PyResult<&[u8]> {
    if !buf.is_c_contiguous() {
        return Err(PyValueError::new_err("buffer must be C-contiguous"));
    }
    let slice =
        unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, buf.len_bytes()) };
    Ok(slice)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn write_batch_zstd<'py>(
    py: Python<'py>,
    fd: i32,
    b_po: PyReadonlyArray1<'py, u32>,
    b_pm: PyReadonlyArray1<'py, u32>,
    b_cm: PyReadonlyArray1<'py, u32>,
    b_ones_idx: PyReadonlyArray1<'py, u64>,
    b_multi_idx: PyReadonlyArray1<'py, u64>,
    num_data: usize,
    rg_pats: usize,
    page_pats: usize,
    level: i32,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
    n_workers: usize,
    mut b_meta: PyReadwriteArray3<'py, i64>,
) -> PyResult<i64> {
    let po_src = b_po.as_slice()?;
    let pm_src = b_pm.as_slice()?;
    let cm_src = b_cm.as_slice()?;
    let ones_idx = b_ones_idx.as_slice()?;
    let multi_idx = b_multi_idx.as_slice()?;
    let meta_slice = b_meta.as_slice_mut()?;
    let b_rgs = (num_data + rg_pats - 1) / rg_pats;
    let n_workers = n_workers.max(1);
    let meta_addr = meta_slice.as_mut_ptr() as usize;

    let rc = py.detach(|| {
        if n_workers <= 1 || b_rgs <= 1 {
            let mut out_vec = Vec::new();
            for rg in 0..b_rgs {
                let meta_rg = unsafe {
                    std::slice::from_raw_parts_mut((meta_addr as *mut i64).add(rg * 9), 9)
                };
                let res = encode_rg_into_vec(
                    rg,
                    num_data,
                    rg_pats,
                    page_pats,
                    level,
                    po_src,
                    pm_src,
                    cm_src,
                    ones_idx,
                    multi_idx,
                    is_delta,
                    is_shuffled,
                    check_sorted,
                    &mut out_vec,
                    meta_rg,
                );
                if res < 0 {
                    return res;
                }
                if !write_all_fd(fd, &out_vec[..res as usize]) {
                    return -4;
                }
            }
            0
        } else {
            let ring_size = (n_workers * 3).max(4).min(b_rgs);
            let mut ring_bufs: Vec<Vec<u8>> = (0..ring_size).map(|_| Vec::new()).collect();
            let bufs_addr = ring_bufs.as_mut_ptr() as usize;
            let next_rg = AtomicUsize::new(0);
            let abort_err = AtomicI32::new(0);
            let pipe_mtx = Mutex::new((0usize, vec![usize::MAX; b_rgs]));
            let cv_writer = Condvar::new();
            let cv_worker = Condvar::new();

            let pool = get_pool(n_workers + 1);
            pool.install(|| {
                rayon::scope(|s| {
                    s.spawn(|_| {
                        for rg in 0..b_rgs {
                            let len = {
                                let mut g = pipe_mtx.lock().unwrap();
                                while g.1[rg] == usize::MAX
                                    && abort_err.load(Ordering::Relaxed) == 0
                                {
                                    g = cv_writer.wait(g).unwrap();
                                }
                                if abort_err.load(Ordering::Relaxed) != 0 {
                                    return;
                                }
                                g.1[rg]
                            };
                            let slot = rg % ring_size;
                            let out_vec = unsafe { &*(bufs_addr as *const Vec<u8>).add(slot) };
                            if !write_all_fd(fd, &out_vec[..len]) {
                                abort_err.store(-4, Ordering::Relaxed);
                                cv_worker.notify_all();
                                return;
                            }
                            let mut g = pipe_mtx.lock().unwrap();
                            g.0 = rg + 1;
                            drop(g);
                            cv_worker.notify_all();
                        }
                    });

                    for _ in 0..n_workers {
                        s.spawn(|_| loop {
                            if abort_err.load(Ordering::Relaxed) != 0 {
                                return;
                            }
                            let rg = next_rg.fetch_add(1, Ordering::Relaxed);
                            if rg >= b_rgs {
                                break;
                            }
                            if rg >= ring_size {
                                let mut g = pipe_mtx.lock().unwrap();
                                while g.0 + ring_size <= rg
                                    && abort_err.load(Ordering::Relaxed) == 0
                                {
                                    g = cv_worker.wait(g).unwrap();
                                }
                                if abort_err.load(Ordering::Relaxed) != 0 {
                                    return;
                                }
                            }
                            let slot = rg % ring_size;
                            let out_vec = unsafe { &mut *(bufs_addr as *mut Vec<u8>).add(slot) };
                            let meta_rg = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (meta_addr as *mut i64).add(rg * 9),
                                    9,
                                )
                            };
                            let res = encode_rg_into_vec(
                                rg,
                                num_data,
                                rg_pats,
                                page_pats,
                                level,
                                po_src,
                                pm_src,
                                cm_src,
                                ones_idx,
                                multi_idx,
                                is_delta,
                                is_shuffled,
                                check_sorted,
                                out_vec,
                                meta_rg,
                            );
                            if res < 0 {
                                abort_err.store(res, Ordering::Relaxed);
                                cv_writer.notify_all();
                                cv_worker.notify_all();
                                return;
                            }
                            let mut g = pipe_mtx.lock().unwrap();
                            g.1[rg] = res as usize;
                            drop(g);
                            cv_writer.notify_one();
                        });
                    }
                });
            });
            abort_err.load(Ordering::Relaxed)
        }
    });

    if rc == -2 {
        return Err(PyValueError::new_err(
            "position values must be sorted within every pattern",
        ));
    }
    if rc != 0 {
        return Err(PyRuntimeError::new_err(
            "Zstd compression or write failed in _parquet_rs",
        ));
    }
    Ok(0)
}

fn read_col_counts_rg(
    dctx: &mut zstd_safe::DCtx<'static>,
    mm_u8: &[u8],
    col_off: usize,
    col_tc: usize,
    col_elems: usize,
    page_buf: &mut Vec<u8>,
    rg_ps: usize,
    rg_pe: usize,
    out_counts: &mut [u32],
) -> i32 {
    let nr = rg_pe - rg_ps;
    if nr == 0 {
        return 0;
    }
    if col_elems == 0 {
        out_counts.fill(0);
        return 0;
    }
    if nr == 1 {
        out_counts[0] = col_elems as u32;
        return 0;
    }

    let mut pos = col_off;
    let col_end = col_off + col_tc;
    let mut page_ps = rg_ps;

    while page_ps < rg_pe && pos < col_end {
        let (u_len_i, p1) = read_zigzag(mm_u8, pos + 3);
        let (c_len_i, p2) = read_zigzag(mm_u8, p1 + 1);
        let (page_nr_i, p3) = read_zigzag(mm_u8, p2 + 2);
        let u_len = u_len_i as usize;
        let c_len = c_len_i as usize;
        let page_nr = page_nr_i as usize;
        let hdr_len = (p3 + 10) - pos;
        let page_pe = page_ps + page_nr;
        let page_elems = (u_len / 4).saturating_sub(page_nr);

        if page_elems == 0 {
            out_counts[page_ps - rg_ps..page_pe - rg_ps].fill(0);
        } else if page_nr == 1 {
            out_counts[page_ps - rg_ps] = page_elems as u32;
        } else {
            if page_buf.len() < u_len {
                page_buf.resize(u_len.max(256 * 1024), 0);
            }
            let c_start = pos + hdr_len;
            match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len]) {
                Ok(d_sz) if d_sz == u_len => {}
                _ => return -1,
            }
            let page_slice = &page_buf[..u_len];
            let mut b_off = 0usize;
            for p_idx in page_ps..page_pe {
                let bl =
                    u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
                out_counts[p_idx - rg_ps] = (bl / 4) as u32;
                b_off += 4 + bl;
            }
        }

        pos += hdr_len + c_len;
        page_ps = page_pe;
    }
    0
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn read_ones_multi<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    n_rgs: usize,
    rg_pats: usize,
    num_data: usize,
    mut b_ones: PyReadwriteArray1<'py, u32>,
    mut b_multi: PyReadwriteArray1<'py, u32>,
    n_workers: usize,
) -> PyResult<i64> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let out_ones = b_ones.as_slice_mut()?;
    let out_multi = b_multi.as_slice_mut()?;
    let ones_addr = out_ones.as_mut_ptr() as usize;
    let multi_addr = out_multi.as_mut_ptr() as usize;
    let n_workers = n_workers.max(1);

    let read_rg = |rg: usize| -> Result<(), i32> {
        let ps = rg * rg_pats;
        let pe = ((rg + 1) * rg_pats).min(num_data);
        let nr = pe - ps;
        let ones_rg = unsafe { std::slice::from_raw_parts_mut((ones_addr as *mut u32).add(ps), nr) };
        let multi_rg =
            unsafe { std::slice::from_raw_parts_mut((multi_addr as *mut u32).add(ps), nr) };

        let rc = THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;
            let dctx = &mut zctx.dctx;
            let scratch = &mut zctx.scratch;

            let r0 = read_col_counts_rg(
                dctx,
                mm_u8,
                col_info[rg * 9] as usize,
                col_info[rg * 9 + 1] as usize,
                col_info[rg * 9 + 2] as usize,
                scratch,
                ps,
                pe,
                ones_rg,
            );
            if r0 != 0 {
                return r0;
            }
            read_col_counts_rg(
                dctx,
                mm_u8,
                col_info[rg * 9 + 3] as usize,
                col_info[rg * 9 + 4] as usize,
                col_info[rg * 9 + 5] as usize,
                scratch,
                ps,
                pe,
                multi_rg,
            )
        });
        if rc == 0 {
            Ok(())
        } else {
            Err(rc)
        }
    };

    let res = py.detach(|| {
        if n_workers <= 1 || n_rgs <= 1 {
            (0..n_rgs).try_for_each(read_rg)
        } else {
            get_pool(n_workers.min(n_rgs))
                .install(|| (0..n_rgs).into_par_iter().try_for_each(read_rg))
        }
    });

    if res.is_err() {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    }
    Ok(0)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn count_span_elems<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    b_urgs: PyReadonlyArray1<'py, i64>,
    b_gstarts: PyReadonlyArray1<'py, i64>,
    b_gstops: PyReadonlyArray1<'py, i64>,
    b_gsids: PyReadonlyArray1<'py, i64>,
    rg_pats: usize,
    num_data: usize,
    b_firsts: PyReadonlyArray1<'py, i64>,
    b_lasts: PyReadonlyArray1<'py, i64>,
    mut b_ones_lens: PyReadwriteArray1<'py, i64>,
    mut b_multi_lens: PyReadwriteArray1<'py, i64>,
    n_workers: usize,
) -> PyResult<i64> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let unique_rgs = b_urgs.as_slice()?;
    let group_starts = b_gstarts.as_slice()?;
    let group_stops = b_gstops.as_slice()?;
    let group_span_ids = b_gsids.as_slice()?;
    let firsts = b_firsts.as_slice()?;
    let lasts = b_lasts.as_slice()?;

    let ones_lens = b_ones_lens.as_slice_mut()?;
    let multi_lens = b_multi_lens.as_slice_mut()?;
    let ones_lens_addr = ones_lens.as_mut_ptr() as usize;
    let multi_lens_addr = multi_lens.as_mut_ptr() as usize;

    let n_groups = unique_rgs.len();
    let n_workers = n_workers.max(1);

    let count_group = |g_idx: usize| -> Result<(), i32> {
        let rg = unique_rgs[g_idx] as usize;
        let ps = rg * rg_pats;
        let pe = ((rg + 1) * rg_pats).min(num_data);
        let gs0 = group_starts[g_idx] as usize;
        let gs1 = group_stops[g_idx] as usize;
        let n_sp = gs1 - gs0;

        let mut p_firsts = Vec::with_capacity(n_sp);
        let mut p_lasts = Vec::with_capacity(n_sp);
        let mut c_o = vec![0i64; n_sp];
        let mut c_m = vec![0i64; n_sp];

        for i in 0..n_sp {
            let sid = group_span_ids[gs0 + i] as usize;
            p_firsts.push(firsts[sid]);
            p_lasts.push(lasts[sid]);
        }

        let rc = THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;
            let dctx = &mut zctx.dctx;
            let scratch = &mut zctx.scratch;

            let r0 = count_col_pages_inner(
                dctx,
                mm_u8,
                col_info[rg * 9] as usize,
                col_info[rg * 9 + 1] as usize,
                col_info[rg * 9 + 2] as usize,
                scratch,
                ps,
                pe,
                &p_firsts,
                &p_lasts,
                &mut c_o,
            );
            if r0 != 0 {
                return r0;
            }
            count_col_pages_inner(
                dctx,
                mm_u8,
                col_info[rg * 9 + 3] as usize,
                col_info[rg * 9 + 4] as usize,
                col_info[rg * 9 + 5] as usize,
                scratch,
                ps,
                pe,
                &p_firsts,
                &p_lasts,
                &mut c_m,
            )
        });
        if rc != 0 {
            return Err(rc);
        }

        for i in 0..n_sp {
            let sid = group_span_ids[gs0 + i] as usize;
            unsafe {
                *(ones_lens_addr as *mut i64).add(sid) = c_o[i];
                *(multi_lens_addr as *mut i64).add(sid) = c_m[i];
            }
        }
        Ok(())
    };

    let res = py.detach(|| {
        if n_workers <= 1 || n_groups <= 1 {
            (0..n_groups).try_for_each(count_group)
        } else {
            get_pool(n_workers.min(n_groups))
                .install(|| (0..n_groups).into_par_iter().try_for_each(count_group))
        }
    });

    if res.is_err() {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    }
    Ok(0)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_rg_batch<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    b_urgs: PyReadonlyArray1<'py, i64>,
    b_gstarts: PyReadonlyArray1<'py, i64>,
    b_gstops: PyReadonlyArray1<'py, i64>,
    b_gsids: PyReadonlyArray1<'py, i64>,
    rg_pats: usize,
    num_data: usize,
    b_firsts: PyReadonlyArray1<'py, i64>,
    b_lasts: PyReadonlyArray1<'py, i64>,
    b_dst_pats: PyReadonlyArray1<'py, i64>,
    b_dst_ones: PyReadonlyArray1<'py, i64>,
    b_dst_multi: PyReadonlyArray1<'py, i64>,
    mut b_out_ones: PyReadwriteArray1<'py, u32>,
    mut b_out_multi: PyReadwriteArray1<'py, u32>,
    mut b_out_po: PyReadwriteArray1<'py, u32>,
    mut b_out_pm: PyReadwriteArray1<'py, u32>,
    mut b_out_cm: PyReadwriteArray1<'py, u32>,
    is_delta: bool,
    is_shuffled: bool,
    n_workers: usize,
) -> PyResult<i64> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let unique_rgs = b_urgs.as_slice()?;
    let group_starts = b_gstarts.as_slice()?;
    let group_stops = b_gstops.as_slice()?;
    let group_span_ids = b_gsids.as_slice()?;
    let firsts = b_firsts.as_slice()?;
    let lasts = b_lasts.as_slice()?;
    let dst_pats = b_dst_pats.as_slice()?;
    let dst_ones = b_dst_ones.as_slice()?;
    let dst_multi = b_dst_multi.as_slice()?;

    let out_ones_slice = b_out_ones.as_slice_mut()?;
    let out_multi_slice = b_out_multi.as_slice_mut()?;
    let out_po_slice = b_out_po.as_slice_mut()?;
    let out_pm_slice = b_out_pm.as_slice_mut()?;
    let out_cm_slice = b_out_cm.as_slice_mut()?;

    let out_ones_addr = out_ones_slice.as_mut_ptr() as usize;
    let out_ones_len = out_ones_slice.len();
    let out_multi_addr = out_multi_slice.as_mut_ptr() as usize;
    let out_multi_len = out_multi_slice.len();
    let out_po_addr = out_po_slice.as_mut_ptr() as usize;
    let out_po_len = out_po_slice.len();
    let out_pm_addr = out_pm_slice.as_mut_ptr() as usize;
    let out_pm_len = out_pm_slice.len();
    let out_cm_addr = out_cm_slice.as_mut_ptr() as usize;
    let out_cm_len = out_cm_slice.len();

    let n_groups = unique_rgs.len();
    let n_workers = n_workers.max(1);

    let decode_group = |g_idx: usize| -> Result<(), i32> {
        let rg = unique_rgs[g_idx] as usize;
        let ps = rg * rg_pats;
        let pe = ((rg + 1) * rg_pats).min(num_data);
        let gs0 = group_starts[g_idx] as usize;
        let gs1 = group_stops[g_idx] as usize;
        let n_sp = gs1 - gs0;

        let mut p_firsts = Vec::with_capacity(n_sp);
        let mut p_lasts = Vec::with_capacity(n_sp);
        let mut d_pats = Vec::with_capacity(n_sp);
        let mut d_ones = Vec::with_capacity(n_sp);
        let mut d_multi = Vec::with_capacity(n_sp);
        let mut cur_dst = vec![0usize; n_sp];
        let mut need_m_payload = false;

        for i in 0..n_sp {
            let sid = group_span_ids[gs0 + i] as usize;
            p_firsts.push(firsts[sid]);
            p_lasts.push(lasts[sid]);
            d_pats.push(dst_pats[sid]);
            d_ones.push(dst_ones[sid]);
            d_multi.push(dst_multi[sid]);
            if dst_multi[sid + 1] > dst_multi[sid] {
                need_m_payload = true;
            }
        }

        let out_ones =
            unsafe { std::slice::from_raw_parts_mut(out_ones_addr as *mut u32, out_ones_len) };
        let out_multi =
            unsafe { std::slice::from_raw_parts_mut(out_multi_addr as *mut u32, out_multi_len) };
        let out_po = unsafe { std::slice::from_raw_parts_mut(out_po_addr as *mut u32, out_po_len) };
        let out_pm = unsafe { std::slice::from_raw_parts_mut(out_pm_addr as *mut u32, out_pm_len) };
        let out_cm = unsafe { std::slice::from_raw_parts_mut(out_cm_addr as *mut u32, out_cm_len) };

        let rc = THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;
            let dctx = &mut zctx.dctx;
            let scratch = &mut zctx.scratch;

            let r = decode_col_pages_inner(
                dctx,
                mm_u8,
                col_info[rg * 9] as usize,
                col_info[rg * 9 + 1] as usize,
                col_info[rg * 9 + 2] as usize,
                scratch,
                ps,
                pe,
                out_po,
                Some(out_ones),
                &p_firsts,
                &p_lasts,
                &d_pats,
                &d_ones,
                &mut cur_dst,
                is_delta,
                is_shuffled,
            );
            if r != 0 {
                return r;
            }

            let r1 = decode_col_pages_inner(
                dctx,
                mm_u8,
                col_info[rg * 9 + 3] as usize,
                col_info[rg * 9 + 4] as usize,
                col_info[rg * 9 + 5] as usize,
                scratch,
                ps,
                pe,
                out_pm,
                Some(out_multi),
                &p_firsts,
                &p_lasts,
                &d_pats,
                &d_multi,
                &mut cur_dst,
                is_delta,
                is_shuffled,
            );
            if r1 != 0 {
                return r1;
            }

            if need_m_payload {
                let r2 = decode_col_pages_inner(
                    dctx,
                    mm_u8,
                    col_info[rg * 9 + 6] as usize,
                    col_info[rg * 9 + 7] as usize,
                    col_info[rg * 9 + 8] as usize,
                    scratch,
                    ps,
                    pe,
                    out_cm,
                    None,
                    &p_firsts,
                    &p_lasts,
                    &d_pats,
                    &d_multi,
                    &mut cur_dst,
                    false,
                    is_shuffled,
                );
                if r2 != 0 {
                    return r2;
                }
            }
            0
        });
        if rc == 0 {
            Ok(())
        } else {
            Err(rc)
        }
    };

    let res = py.detach(|| {
        if n_workers <= 1 || n_groups <= 1 {
            (0..n_groups).try_for_each(decode_group)
        } else {
            get_pool(n_workers.min(n_groups))
                .install(|| (0..n_groups).into_par_iter().try_for_each(decode_group))
        }
    });

    if res.is_err() {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    }
    Ok(0)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_single_frame<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    rg: usize,
    ps: usize,
    pe: usize,
    frame: usize,
    is_delta: bool,
    is_shuffled: bool,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;

    let res = THREAD_ZSTD.with(|cell| -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), i32> {
        let mut ctx = cell.borrow_mut();
        let zctx = &mut *ctx;
        let dctx = &mut zctx.dctx;
        let scratch = &mut zctx.scratch;

        let po = decode_single_col_frame(
            dctx,
            mm_u8,
            col_info[rg * 9] as usize,
            col_info[rg * 9 + 1] as usize,
            col_info[rg * 9 + 2] as usize,
            scratch,
            ps,
            pe,
            frame,
            is_delta,
            is_shuffled,
        )?;
        let pm = decode_single_col_frame(
            dctx,
            mm_u8,
            col_info[rg * 9 + 3] as usize,
            col_info[rg * 9 + 4] as usize,
            col_info[rg * 9 + 5] as usize,
            scratch,
            ps,
            pe,
            frame,
            is_delta,
            is_shuffled,
        )?;
        let cm = if pm.is_empty() {
            Vec::new()
        } else {
            decode_single_col_frame(
                dctx,
                mm_u8,
                col_info[rg * 9 + 6] as usize,
                col_info[rg * 9 + 7] as usize,
                col_info[rg * 9 + 8] as usize,
                scratch,
                ps,
                pe,
                frame,
                false,
                is_shuffled,
            )?
        };
        Ok((po, pm, cm))
    });

    match res {
        Ok((po, pm, cm)) => Ok((
            PyArray1::from_vec(py, po),
            PyArray1::from_vec(py, pm),
            PyArray1::from_vec(py, cm),
        )),
        Err(_) => Err(PyValueError::new_err("Corrupted Zstd page in Parquet file")),
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decode_col<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    c_idx: usize,
    n_rgs: usize,
    rg_pats: usize,
    num_data: usize,
    b_rg_offs: PyReadonlyArray1<'py, i64>,
    mut b_out: PyReadwriteArray1<'py, u32>,
    is_delta: bool,
    is_shuffled: bool,
    n_workers: usize,
) -> PyResult<i64> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let rg_offs = b_rg_offs.as_slice()?;
    let out_slice = b_out.as_slice_mut()?;
    let out_addr = out_slice.as_mut_ptr() as usize;
    let out_len = out_slice.len();
    let n_workers = n_workers.max(1);

    let decode_one_rg = |rg: usize| -> Result<(), i32> {
        let ps = rg * rg_pats;
        let pe = ((rg + 1) * rg_pats).min(num_data);
        let rg_base = rg_offs[rg];
        if rg_offs[rg + 1] <= rg_base {
            return Ok(());
        }
        let pf_arr = [ps as i64];
        let pl_arr = [pe as i64];
        let dp_arr = [0i64];
        let dst_arr = [rg_base];
        let mut cur_dst = [0usize];
        let out_u32 = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, out_len) };

        let rc = THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;
            let dctx = &mut zctx.dctx;
            let scratch = &mut zctx.scratch;
            decode_col_pages_inner(
                dctx,
                mm_u8,
                col_info[rg * 9 + c_idx * 3] as usize,
                col_info[rg * 9 + c_idx * 3 + 1] as usize,
                col_info[rg * 9 + c_idx * 3 + 2] as usize,
                scratch,
                ps,
                pe,
                out_u32,
                None,
                &pf_arr,
                &pl_arr,
                &dp_arr,
                &dst_arr,
                &mut cur_dst,
                is_delta,
                is_shuffled,
            )
        });
        if rc == 0 {
            Ok(())
        } else {
            Err(rc)
        }
    };

    let res = py.detach(|| {
        if n_workers <= 1 || n_rgs <= 1 {
            (0..n_rgs).try_for_each(decode_one_rg)
        } else {
            get_pool(n_workers.min(n_rgs))
                .install(|| (0..n_rgs).into_par_iter().try_for_each(decode_one_rg))
        }
    });

    if res.is_err() {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    }
    Ok(0)
}

#[pymodule]
fn _parquet_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_batch_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(read_ones_multi, m)?)?;
    m.add_function(wrap_pyfunction!(count_span_elems, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rg_batch, m)?)?;
    m.add_function(wrap_pyfunction!(decode_single_frame, m)?)?;
    m.add_function(wrap_pyfunction!(decode_col, m)?)?;
    Ok(())
}
