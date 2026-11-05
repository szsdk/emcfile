mod pool;
mod py_ffi;
mod simd;
mod zstd_ffi;

use std::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use pool::par_for_each;
use py_ffi::*;
use simd::{decode_pattern_u32, encode_pattern_u32};
use zstd_ffi::{get_zstd, ZstdApi, THREAD_ZSTD};

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
        let n = b0;
        return (((n >> 1) as i64) ^ -((n & 1) as i64), pos + 1);
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

fn encode_u32_col(
    api: &ZstdApi,
    cctx: *mut c_void,
    level: i32,
    arr: &[u32],
    comp_scratch: &mut [u8],
    out_buf: &mut [u8],
    pos: usize,
) -> Result<(usize, i64, i64), i32> {
    let nr = arr.len();
    let u_len = nr * 4;
    let c_bound = comp_scratch.len();
    let c_len = unsafe {
        (api.compress_cctx)(
            cctx,
            comp_scratch.as_mut_ptr() as *mut c_void,
            c_bound,
            arr.as_ptr() as *const c_void,
            u_len,
            level as c_int,
        )
    };
    if unsafe { (api.is_error)(c_len) } != 0 {
        return Err(-3);
    }
    let pos0 = pos;
    let hdr_end = write_page_header(out_buf, pos, u_len, c_len, nr);
    out_buf[hdr_end..hdr_end + c_len].copy_from_slice(&comp_scratch[..c_len]);
    let new_pos = hdr_end + c_len;
    let hdr_len = hdr_end - pos0;
    Ok((new_pos, (hdr_len + u_len) as i64, (hdr_len + c_len) as i64))
}

#[allow(clippy::too_many_arguments)]
fn encode_payload_col_pages(
    api: &ZstdApi,
    cctx: *mut c_void,
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
        let c_bound = comp_scratch.len();
        let c_len = unsafe {
            (api.compress_cctx)(
                cctx,
                comp_scratch.as_mut_ptr() as *mut c_void,
                c_bound,
                scratch.as_ptr() as *const c_void,
                u_len,
                level as c_int,
            )
        };
        if unsafe { (api.is_error)(c_len) } != 0 {
            return Err(-3);
        }
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
    api: &ZstdApi,
    rg: usize,
    num_data: usize,
    rg_pats: usize,
    page_pats: usize,
    level: i32,
    ones: &[u32],
    multi: &[u32],
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

    let need_sc = ((max_page_elems + page_pats) * 4).max(nr * 4) + 64;
    let need_csc = unsafe { (api.compress_bound)(need_sc) } + 64;
    let n_pages = (nr + page_pats - 1) / page_pats;
    let need_out = unsafe {
        (api.compress_bound)(nr * 4) * 2
            + (api.compress_bound)((no + nr) * 4)
            + (api.compress_bound)((nm + nr) * 4) * 2
            + (n_pages * 3 + 2) * 64
    };

    if out_vec.len() < need_out {
        out_vec.resize(need_out, 0);
    }

    THREAD_ZSTD.with(|cell| {
        let mut ctx = cell.borrow_mut();
        ctx.ensure_scratch(need_sc);
        ctx.ensure_comp_scratch(need_csc);
        let cctx = ctx.cctx;
        let zctx = &mut *ctx;
        let sc = &mut zctx.scratch[..need_sc];
        let csc = &mut zctx.comp_scratch[..need_csc];

        let mut pos = 0usize;
        match encode_u32_col(api, cctx, level, &ones[ps..pe], csc, out_vec, pos) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[0] = tu;
                meta_rg[1] = tc;
            }
            Err(e) => return e,
        }
        match encode_u32_col(api, cctx, level, &multi[ps..pe], csc, out_vec, pos) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[2] = tu;
                meta_rg[3] = tc;
            }
            Err(e) => return e,
        }
        match encode_payload_col_pages(
            api, cctx, level, po_src, ones_idx, ps, pe, page_pats, sc, csc, out_vec, pos, is_delta,
            is_shuffled, check_sorted,
        ) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[4] = tu;
                meta_rg[5] = tc;
            }
            Err(e) => return e,
        }
        match encode_payload_col_pages(
            api, cctx, level, pm_src, multi_idx, ps, pe, page_pats, sc, csc, out_vec, pos,
            is_delta, is_shuffled, check_sorted,
        ) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[6] = tu;
                meta_rg[7] = tc;
            }
            Err(e) => return e,
        }
        match encode_payload_col_pages(
            api, cctx, level, cm_src, multi_idx, ps, pe, page_pats, sc, csc, out_vec, pos, false,
            is_shuffled, false,
        ) {
            Ok((p, tu, tc)) => {
                pos = p;
                meta_rg[8] = tu;
                meta_rg[9] = tc;
            }
            Err(e) => return e,
        }
        pos as i32
    })
}

unsafe extern "C" {
    fn write(fd: c_int, buf: *const c_void, count: usize) -> isize;
}

fn write_all_fd(fd: c_int, mut buf: &[u8]) -> bool {
    while !buf.is_empty() {
        let n = unsafe { write(fd, buf.as_ptr() as *const c_void, buf.len()) };
        if n <= 0 {
            return false;
        }
        buf = &buf[n as usize..];
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn decode_col_pages_inner(
    api: &ZstdApi,
    dctx: *mut c_void,
    mm_u8: &[u8],
    col_off: usize,
    col_tc: usize,
    page_buf: &mut Vec<u8>,
    rg_ps: usize,
    rg_pe: usize,
    pat_offsets: &[i64],
    out_u32: &mut [u32],
    p_firsts: &[i64],
    p_lasts: &[i64],
    dst_offs: &[i64],
    is_delta: bool,
    is_shuffled: bool,
) -> i32 {
    let n_spans = p_firsts.len();
    if n_spans == 0 {
        return 0;
    }
    let min_pat = p_firsts[0] as usize;
    let max_pat = p_lasts[n_spans - 1] as usize;

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

        if page_ps >= max_pat {
            break;
        }
        while sp_start < n_spans && (p_lasts[sp_start] as usize) <= page_ps {
            sp_start += 1;
        }
        if sp_start >= n_spans {
            break;
        }

        let mut overlaps = false;
        let mut s_idx = sp_start;
        while s_idx < n_spans && (p_firsts[s_idx] as usize) < page_pe {
            let f = (p_firsts[s_idx] as usize).max(page_ps);
            let l = (p_lasts[s_idx] as usize).min(page_pe);
            if l > f && pat_offsets[l] > pat_offsets[f] {
                overlaps = true;
                break;
            }
            s_idx += 1;
        }

        if overlaps && page_pe > min_pat {
            if page_buf.len() < u_len {
                page_buf.resize(u_len.max(256 * 1024), 0);
            }
            let d_sz = unsafe {
                (api.decompress_dctx)(
                    dctx,
                    page_buf.as_mut_ptr() as *mut c_void,
                    u_len,
                    mm_u8.as_ptr().add(pos + hdr_len) as *const c_void,
                    c_len,
                )
            };
            if d_sz != u_len {
                return -1;
            }
            let page_base_elem = pat_offsets[page_ps] as usize;
            let page_slice = &page_buf[..u_len];
            let mut s_i = sp_start;
            while s_i < n_spans && (p_firsts[s_i] as usize) < page_pe {
                let sp_f = p_firsts[s_i] as usize;
                let sp_l = p_lasts[s_i] as usize;
                let f = sp_f.max(page_ps);
                let l = sp_l.min(page_pe);
                if l > f {
                    let mut out_idx =
                        (dst_offs[s_i] + (pat_offsets[f] - pat_offsets[sp_f])) as usize;
                    for p_idx in f..l {
                        let s = pat_offsets[p_idx] as usize;
                        let e = pat_offsets[p_idx + 1] as usize;
                        let k = e - s;
                        if k > 0 {
                            let b_s = (s - page_base_elem + (p_idx - page_ps) + 1) * 4;
                            decode_pattern_u32(
                                page_slice,
                                b_s,
                                &mut out_u32[out_idx..out_idx + k],
                                is_delta,
                                is_shuffled,
                            );
                            out_idx += k;
                        }
                    }
                }
                s_i += 1;
            }
        }

        pos += hdr_len + c_len;
        page_ps = page_pe;
    }
    0
}

unsafe extern "C" fn py_write_batch_zstd(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
) -> *mut PyObject {
    if nargs != 17 {
        return raise_value_error(b"write_batch_zstd expects 17 arguments\0");
    }
    let fd = arg_i64(args, 0) as c_int;
    let b_ones = match PyBuf::get_readonly(unsafe { *args.add(1) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_multi = match PyBuf::get_readonly(unsafe { *args.add(2) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_po = match PyBuf::get_readonly(unsafe { *args.add(3) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_pm = match PyBuf::get_readonly(unsafe { *args.add(4) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_cm = match PyBuf::get_readonly(unsafe { *args.add(5) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_ones_idx = match PyBuf::get_readonly(unsafe { *args.add(6) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_multi_idx = match PyBuf::get_readonly(unsafe { *args.add(7) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let num_data = arg_usize(args, 8);
    let rg_pats = arg_usize(args, 9);
    let page_pats = arg_usize(args, 10);
    let level = arg_i64(args, 11) as i32;
    let is_delta = arg_bool(args, 12);
    let is_shuffled = arg_bool(args, 13);
    let check_sorted = arg_bool(args, 14);
    let n_workers = arg_usize(args, 15).max(1);
    let mut b_meta = match PyBuf::get_writable(unsafe { *args.add(16) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };

    let api = match get_zstd() {
        Some(a) => a,
        None => return raise_runtime_error(b"libzstd not available\0"),
    };

    let ones = b_ones.as_slice::<u32>();
    let multi = b_multi.as_slice::<u32>();
    let po_src = b_po.as_slice::<u32>();
    let pm_src = b_pm.as_slice::<u32>();
    let cm_src = b_cm.as_slice::<u32>();
    let ones_idx = b_ones_idx.as_slice::<u64>();
    let multi_idx = b_multi_idx.as_slice::<u64>();
    let meta_slice = b_meta.as_mut_slice::<i64>();
    let b_rgs = (num_data + rg_pats - 1) / rg_pats;

    let rc = {
        let _nogil = AllowThreads::new();
        let meta_addr = meta_slice.as_mut_ptr() as usize;

        if n_workers <= 1 || b_rgs <= 1 {
            let mut out_vec = Vec::new();
            let mut single_rc = 0i32;
            for rg in 0..b_rgs {
                let meta_rg = unsafe {
                    std::slice::from_raw_parts_mut((meta_addr as *mut i64).add(rg * 10), 10)
                };
                let res = encode_rg_into_vec(
                    api,
                    rg,
                    num_data,
                    rg_pats,
                    page_pats,
                    level,
                    ones,
                    multi,
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
                    single_rc = res;
                    break;
                }
                if !write_all_fd(fd, &out_vec[..res as usize]) {
                    single_rc = -4;
                    break;
                }
            }
            single_rc
        } else {
            let ring_size = (n_workers * 3).max(4).min(b_rgs);
            let mut ring_bufs: Vec<Vec<u8>> = (0..ring_size).map(|_| Vec::new()).collect();
            let bufs_addr = ring_bufs.as_mut_ptr() as usize;
            let next_rg = AtomicUsize::new(0);
            let abort_err = AtomicI32::new(0);
            // state: (written_count, ready_len_per_rg)
            let pipe_mtx = Mutex::new((0usize, vec![usize::MAX; b_rgs]));
            let cv_writer = Condvar::new();
            let cv_worker = Condvar::new();

            par_for_each(n_workers + 1, n_workers + 1, |tid| {
                if tid == 0 {
                    for rg in 0..b_rgs {
                        let len = {
                            let mut g = pipe_mtx.lock().unwrap();
                            while g.1[rg] == usize::MAX && abort_err.load(Ordering::Relaxed) == 0 {
                                g = cv_writer.wait(g).unwrap();
                            }
                            if abort_err.load(Ordering::Relaxed) != 0 {
                                return abort_err.load(Ordering::Relaxed);
                            }
                            g.1[rg]
                        };
                        let slot = rg % ring_size;
                        let out_vec = unsafe { &*(bufs_addr as *const Vec<u8>).add(slot) };
                        if !write_all_fd(fd, &out_vec[..len]) {
                            abort_err.store(-4, Ordering::Relaxed);
                            cv_worker.notify_all();
                            return -4;
                        }
                        let mut g = pipe_mtx.lock().unwrap();
                        g.0 = rg + 1;
                        drop(g);
                        cv_worker.notify_all();
                    }
                    0
                } else {
                    loop {
                        if abort_err.load(Ordering::Relaxed) != 0 {
                            return abort_err.load(Ordering::Relaxed);
                        }
                        let rg = next_rg.fetch_add(1, Ordering::Relaxed);
                        if rg >= b_rgs {
                            break;
                        }
                        if rg >= ring_size {
                            let mut g = pipe_mtx.lock().unwrap();
                            while g.0 + ring_size <= rg && abort_err.load(Ordering::Relaxed) == 0 {
                                g = cv_worker.wait(g).unwrap();
                            }
                            if abort_err.load(Ordering::Relaxed) != 0 {
                                return abort_err.load(Ordering::Relaxed);
                            }
                        }
                        let slot = rg % ring_size;
                        let out_vec = unsafe { &mut *(bufs_addr as *mut Vec<u8>).add(slot) };
                        let meta_rg = unsafe {
                            std::slice::from_raw_parts_mut(
                                (meta_addr as *mut i64).add(rg * 10),
                                10,
                            )
                        };
                        let res = encode_rg_into_vec(
                            api,
                            rg,
                            num_data,
                            rg_pats,
                            page_pats,
                            level,
                            ones,
                            multi,
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
                            return res;
                        }
                        let mut g = pipe_mtx.lock().unwrap();
                        g.1[rg] = res as usize;
                        drop(g);
                        cv_writer.notify_one();
                    }
                    0
                }
            })
        }
    };

    if rc == -2 {
        return raise_value_error(b"position values must be sorted within every pattern\0");
    }
    if rc != 0 {
        return raise_runtime_error(b"Zstd compression or write failed in _parquet_rs\0");
    }
    unsafe { PyLong_FromLongLong(0) }
}

unsafe extern "C" fn py_read_ones_multi(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
) -> *mut PyObject {
    if nargs != 7 {
        return raise_value_error(b"read_ones_multi expects 7 arguments\0");
    }
    let b_mm = match PyBuf::get_readonly(unsafe { *args.add(0) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_col = match PyBuf::get_readonly(unsafe { *args.add(1) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let n_rgs = arg_usize(args, 2);
    let rg_pats = arg_usize(args, 3);
    let num_data = arg_usize(args, 4);
    let mut b_ones = match PyBuf::get_writable(unsafe { *args.add(5) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_multi = match PyBuf::get_writable(unsafe { *args.add(6) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };

    let api = match get_zstd() {
        Some(a) => a,
        None => return raise_runtime_error(b"libzstd not available\0"),
    };

    let mm_u8 = b_mm.as_slice::<u8>();
    let col_info = b_col.as_slice::<i64>();
    let out_ones = b_ones.as_mut_slice::<u32>();
    let out_multi = b_multi.as_mut_slice::<u32>();

    let rc = THREAD_ZSTD.with(|cell| {
        let ctx = cell.borrow();
        let dctx = ctx.dctx;
        for rg in 0..n_rgs {
            let ps = rg * rg_pats;
            let pe = ((rg + 1) * rg_pats).min(num_data);
            let u_expect = (pe - ps) * 4;

            let off0 = col_info[rg * 10] as usize;
            let tc0 = col_info[rg * 10 + 1] as usize;
            let (_, p1) = read_zigzag(mm_u8, off0 + 3);
            let (c0_i, _) = read_zigzag(mm_u8, p1 + 1);
            let c0 = c0_i as usize;
            let hdr0 = tc0 - c0;
            let d0 = unsafe {
                (api.decompress_dctx)(
                    dctx,
                    out_ones.as_mut_ptr().add(ps) as *mut c_void,
                    u_expect,
                    mm_u8.as_ptr().add(off0 + hdr0) as *const c_void,
                    c0,
                )
            };
            if d0 != u_expect {
                return -1;
            }

            let off1 = col_info[rg * 10 + 2] as usize;
            let tc1 = col_info[rg * 10 + 3] as usize;
            let (_, p1b) = read_zigzag(mm_u8, off1 + 3);
            let (c1_i, _) = read_zigzag(mm_u8, p1b + 1);
            let c1 = c1_i as usize;
            let hdr1 = tc1 - c1;
            let d1 = unsafe {
                (api.decompress_dctx)(
                    dctx,
                    out_multi.as_mut_ptr().add(ps) as *mut c_void,
                    u_expect,
                    mm_u8.as_ptr().add(off1 + hdr1) as *const c_void,
                    c1,
                )
            };
            if d1 != u_expect {
                return -1;
            }
        }
        0
    });

    if rc != 0 {
        return raise_value_error(b"Corrupted Zstd page in Parquet file\0");
    }
    unsafe { PyLong_FromLongLong(0) }
}

unsafe extern "C" fn py_decode_rg_batch(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
) -> *mut PyObject {
    if nargs != 20 {
        return raise_value_error(b"decode_rg_batch expects 20 arguments\0");
    }
    let b_mm = match PyBuf::get_readonly(unsafe { *args.add(0) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_col = match PyBuf::get_readonly(unsafe { *args.add(1) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_urgs = match PyBuf::get_readonly(unsafe { *args.add(2) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_gstarts = match PyBuf::get_readonly(unsafe { *args.add(3) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_gstops = match PyBuf::get_readonly(unsafe { *args.add(4) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_gsids = match PyBuf::get_readonly(unsafe { *args.add(5) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let rg_pats = arg_usize(args, 6);
    let num_data = arg_usize(args, 7);
    let b_ones_idx = match PyBuf::get_readonly(unsafe { *args.add(8) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_multi_idx = match PyBuf::get_readonly(unsafe { *args.add(9) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_firsts = match PyBuf::get_readonly(unsafe { *args.add(10) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_lasts = match PyBuf::get_readonly(unsafe { *args.add(11) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_dst_ones = match PyBuf::get_readonly(unsafe { *args.add(12) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_dst_multi = match PyBuf::get_readonly(unsafe { *args.add(13) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_out_po = match PyBuf::get_writable(unsafe { *args.add(14) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_out_pm = match PyBuf::get_writable(unsafe { *args.add(15) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_out_cm = match PyBuf::get_writable(unsafe { *args.add(16) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let is_delta = arg_bool(args, 17);
    let is_shuffled = arg_bool(args, 18);
    let n_workers = arg_usize(args, 19).max(1);

    let api = match get_zstd() {
        Some(a) => a,
        None => return raise_runtime_error(b"libzstd not available\0"),
    };

    let mm_u8 = b_mm.as_slice::<u8>();
    let col_info = b_col.as_slice::<i64>();
    let unique_rgs = b_urgs.as_slice::<i64>();
    let group_starts = b_gstarts.as_slice::<i64>();
    let group_stops = b_gstops.as_slice::<i64>();
    let group_span_ids = b_gsids.as_slice::<i64>();
    let ones_idx = b_ones_idx.as_slice::<i64>();
    let multi_idx = b_multi_idx.as_slice::<i64>();
    let firsts = b_firsts.as_slice::<i64>();
    let lasts = b_lasts.as_slice::<i64>();
    let dst_ones = b_dst_ones.as_slice::<i64>();
    let dst_multi = b_dst_multi.as_slice::<i64>();

    let out_po_slice = b_out_po.as_mut_slice::<u32>();
    let out_pm_slice = b_out_pm.as_mut_slice::<u32>();
    let out_cm_slice = b_out_cm.as_mut_slice::<u32>();

    let out_po_addr = out_po_slice.as_mut_ptr() as usize;
    let out_po_len = out_po_slice.len();
    let out_pm_addr = out_pm_slice.as_mut_ptr() as usize;
    let out_pm_len = out_pm_slice.len();
    let out_cm_addr = out_cm_slice.as_mut_ptr() as usize;
    let out_cm_len = out_cm_slice.len();

    let n_groups = unique_rgs.len();
    let rc = {
        let _nogil = AllowThreads::new();
        par_for_each(n_groups, n_workers, |g_idx| {
            let rg = unique_rgs[g_idx] as usize;
            let ps = rg * rg_pats;
            let pe = ((rg + 1) * rg_pats).min(num_data);
            let gs0 = group_starts[g_idx] as usize;
            let gs1 = group_stops[g_idx] as usize;
            let n_sp = gs1 - gs0;

            let mut p_firsts = Vec::with_capacity(n_sp);
            let mut p_lasts = Vec::with_capacity(n_sp);
            let mut d_ones = Vec::with_capacity(n_sp);
            let mut d_multi = Vec::with_capacity(n_sp);
            let mut need_o = false;
            let mut need_m = false;

            for i in 0..n_sp {
                let sid = group_span_ids[gs0 + i] as usize;
                let f = firsts[sid];
                let l = lasts[sid];
                p_firsts.push(f);
                p_lasts.push(l);
                d_ones.push(dst_ones[sid]);
                d_multi.push(dst_multi[sid]);
                if ones_idx[l as usize] > ones_idx[f as usize] {
                    need_o = true;
                }
                if multi_idx[l as usize] > multi_idx[f as usize] {
                    need_m = true;
                }
            }

            let out_po = unsafe { std::slice::from_raw_parts_mut(out_po_addr as *mut u32, out_po_len) };
            let out_pm = unsafe { std::slice::from_raw_parts_mut(out_pm_addr as *mut u32, out_pm_len) };
            let out_cm = unsafe { std::slice::from_raw_parts_mut(out_cm_addr as *mut u32, out_cm_len) };

            THREAD_ZSTD.with(|cell| {
                let mut ctx = cell.borrow_mut();
                let dctx = ctx.dctx;
                let scratch = &mut ctx.scratch;

                if need_o {
                    let r = decode_col_pages_inner(
                        api,
                        dctx,
                        mm_u8,
                        col_info[rg * 10 + 4] as usize,
                        col_info[rg * 10 + 5] as usize,
                        scratch,
                        ps,
                        pe,
                        ones_idx,
                        out_po,
                        &p_firsts,
                        &p_lasts,
                        &d_ones,
                        is_delta,
                        is_shuffled,
                    );
                    if r != 0 {
                        return r;
                    }
                }
                if need_m {
                    let r = decode_col_pages_inner(
                        api,
                        dctx,
                        mm_u8,
                        col_info[rg * 10 + 6] as usize,
                        col_info[rg * 10 + 7] as usize,
                        scratch,
                        ps,
                        pe,
                        multi_idx,
                        out_pm,
                        &p_firsts,
                        &p_lasts,
                        &d_multi,
                        is_delta,
                        is_shuffled,
                    );
                    if r != 0 {
                        return r;
                    }
                    let r2 = decode_col_pages_inner(
                        api,
                        dctx,
                        mm_u8,
                        col_info[rg * 10 + 8] as usize,
                        col_info[rg * 10 + 9] as usize,
                        scratch,
                        ps,
                        pe,
                        multi_idx,
                        out_cm,
                        &p_firsts,
                        &p_lasts,
                        &d_multi,
                        false,
                        is_shuffled,
                    );
                    if r2 != 0 {
                        return r2;
                    }
                }
                0
            })
        })
    };

    if rc != 0 {
        return raise_value_error(b"Corrupted Zstd page in Parquet file\0");
    }
    unsafe { PyLong_FromLongLong(0) }
}

unsafe extern "C" fn py_decode_single_frame(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
) -> *mut PyObject {
    if nargs != 13 {
        return raise_value_error(b"decode_single_frame expects 13 arguments\0");
    }
    let b_mm = match PyBuf::get_readonly(unsafe { *args.add(0) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_col = match PyBuf::get_readonly(unsafe { *args.add(1) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let rg = arg_usize(args, 2);
    let ps = arg_usize(args, 3);
    let pe = arg_usize(args, 4);
    let frame = arg_i64(args, 5);
    let b_ones_idx = match PyBuf::get_readonly(unsafe { *args.add(6) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_multi_idx = match PyBuf::get_readonly(unsafe { *args.add(7) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_po = match PyBuf::get_writable(unsafe { *args.add(8) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_pm = match PyBuf::get_writable(unsafe { *args.add(9) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_cm = match PyBuf::get_writable(unsafe { *args.add(10) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let is_delta = arg_bool(args, 11);
    let is_shuffled = arg_bool(args, 12);

    let api = match get_zstd() {
        Some(a) => a,
        None => return raise_runtime_error(b"libzstd not available\0"),
    };

    let mm_u8 = b_mm.as_slice::<u8>();
    let col_info = b_col.as_slice::<i64>();
    let ones_idx = b_ones_idx.as_slice::<i64>();
    let multi_idx = b_multi_idx.as_slice::<i64>();
    let out_po = b_po.as_mut_slice::<u32>();
    let out_pm = b_pm.as_mut_slice::<u32>();
    let out_cm = b_cm.as_mut_slice::<u32>();

    let pf_arr = [frame];
    let pl_arr = [frame + 1];
    let dst_arr = [0i64];

    let rc = THREAD_ZSTD.with(|cell| {
        let mut ctx = cell.borrow_mut();
        let dctx = ctx.dctx;
        let scratch = &mut ctx.scratch;

        if !out_po.is_empty() {
            let r = decode_col_pages_inner(
                api,
                dctx,
                mm_u8,
                col_info[rg * 10 + 4] as usize,
                col_info[rg * 10 + 5] as usize,
                scratch,
                ps,
                pe,
                ones_idx,
                out_po,
                &pf_arr,
                &pl_arr,
                &dst_arr,
                is_delta,
                is_shuffled,
            );
            if r != 0 {
                return r;
            }
        }
        if !out_pm.is_empty() {
            let r = decode_col_pages_inner(
                api,
                dctx,
                mm_u8,
                col_info[rg * 10 + 6] as usize,
                col_info[rg * 10 + 7] as usize,
                scratch,
                ps,
                pe,
                multi_idx,
                out_pm,
                &pf_arr,
                &pl_arr,
                &dst_arr,
                is_delta,
                is_shuffled,
            );
            if r != 0 {
                return r;
            }
            let r2 = decode_col_pages_inner(
                api,
                dctx,
                mm_u8,
                col_info[rg * 10 + 8] as usize,
                col_info[rg * 10 + 9] as usize,
                scratch,
                ps,
                pe,
                multi_idx,
                out_cm,
                &pf_arr,
                &pl_arr,
                &dst_arr,
                false,
                is_shuffled,
            );
            if r2 != 0 {
                return r2;
            }
        }
        0
    });

    if rc != 0 {
        return raise_value_error(b"Corrupted Zstd page in Parquet file\0");
    }
    unsafe { PyLong_FromLongLong(0) }
}

unsafe extern "C" fn py_decode_col(
    _self: *mut PyObject,
    args: *const *mut PyObject,
    nargs: Py_ssize_t,
) -> *mut PyObject {
    if nargs != 11 {
        return raise_value_error(b"decode_col expects 11 arguments\0");
    }
    let b_mm = match PyBuf::get_readonly(unsafe { *args.add(0) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let b_col = match PyBuf::get_readonly(unsafe { *args.add(1) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let c_idx = arg_usize(args, 2);
    let n_rgs = arg_usize(args, 3);
    let rg_pats = arg_usize(args, 4);
    let num_data = arg_usize(args, 5);
    let b_offs = match PyBuf::get_readonly(unsafe { *args.add(6) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let mut b_out = match PyBuf::get_writable(unsafe { *args.add(7) }) {
        Some(b) => b,
        None => return std::ptr::null_mut(),
    };
    let is_delta = arg_bool(args, 8);
    let is_shuffled = arg_bool(args, 9);
    let n_workers = arg_usize(args, 10).max(1);

    let api = match get_zstd() {
        Some(a) => a,
        None => return raise_runtime_error(b"libzstd not available\0"),
    };

    let mm_u8 = b_mm.as_slice::<u8>();
    let col_info = b_col.as_slice::<i64>();
    let offsets = b_offs.as_slice::<i64>();
    let out_slice = b_out.as_mut_slice::<u32>();
    let out_addr = out_slice.as_mut_ptr() as usize;
    let out_len = out_slice.len();

    let rc = {
        let _nogil = AllowThreads::new();
        par_for_each(n_rgs, n_workers, |rg| {
            let ps = rg * rg_pats;
            let pe = ((rg + 1) * rg_pats).min(num_data);
            let rg_base = offsets[ps];
            if offsets[pe] <= rg_base {
                return 0;
            }
            let pf_arr = [ps as i64];
            let pl_arr = [pe as i64];
            let dst_arr = [rg_base];
            let out_u32 = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, out_len) };

            THREAD_ZSTD.with(|cell| {
                let mut ctx = cell.borrow_mut();
                let dctx = ctx.dctx;
                let scratch = &mut ctx.scratch;
                decode_col_pages_inner(
                    api,
                    dctx,
                    mm_u8,
                    col_info[rg * 10 + c_idx * 2] as usize,
                    col_info[rg * 10 + c_idx * 2 + 1] as usize,
                    scratch,
                    ps,
                    pe,
                    offsets,
                    out_u32,
                    &pf_arr,
                    &pl_arr,
                    &dst_arr,
                    is_delta,
                    is_shuffled,
                )
            })
        })
    };

    if rc != 0 {
        return raise_value_error(b"Corrupted Zstd page in Parquet file\0");
    }
    unsafe { PyLong_FromLongLong(0) }
}

static METHODS: [PyMethodDef; 6] = [
    PyMethodDef {
        ml_name: b"write_batch_zstd\0".as_ptr() as *const c_char,
        ml_meth: Some(py_write_batch_zstd),
        ml_flags: METH_FASTCALL,
        ml_doc: b"Encode and write a batch of Parquet row groups in parallel.\0".as_ptr()
            as *const c_char,
    },
    PyMethodDef {
        ml_name: b"read_ones_multi\0".as_ptr() as *const c_char,
        ml_meth: Some(py_read_ones_multi),
        ml_flags: METH_FASTCALL,
        ml_doc: b"Read ones and multi columns directly via Zstd.\0".as_ptr() as *const c_char,
    },
    PyMethodDef {
        ml_name: b"decode_rg_batch\0".as_ptr() as *const c_char,
        ml_meth: Some(py_decode_rg_batch),
        ml_flags: METH_FASTCALL,
        ml_doc: b"Decode a batch of row groups and spans in parallel.\0".as_ptr() as *const c_char,
    },
    PyMethodDef {
        ml_name: b"decode_single_frame\0".as_ptr() as *const c_char,
        ml_meth: Some(py_decode_single_frame),
        ml_flags: METH_FASTCALL,
        ml_doc: b"Decode a single pattern frame from sub-pages.\0".as_ptr() as *const c_char,
    },
    PyMethodDef {
        ml_name: b"decode_col\0".as_ptr() as *const c_char,
        ml_meth: Some(py_decode_col),
        ml_flags: METH_FASTCALL,
        ml_doc: b"Decode a single full column in parallel.\0".as_ptr() as *const c_char,
    },
    PyMethodDef {
        ml_name: std::ptr::null(),
        ml_meth: None,
        ml_flags: 0,
        ml_doc: std::ptr::null(),
    },
];

static mut MODULE_DEF: PyModuleDef = PyModuleDef {
    m_base: PyModuleDef_Base {
        ob_base: PyObject {
            ob_refcnt: 1,
            ob_type: std::ptr::null_mut(),
        },
        m_init: None,
        m_index: 0,
        m_copy: std::ptr::null_mut(),
    },
    m_name: b"_parquet_rs\0".as_ptr() as *const c_char,
    m_doc: b"Rust + SIMD + Zstd Parquet acceleration for emcfile.\0".as_ptr() as *const c_char,
    m_size: -1,
    m_methods: METHODS.as_ptr(),
    m_slots: std::ptr::null(),
    m_traverse: std::ptr::null(),
    m_clear: std::ptr::null(),
    m_free: std::ptr::null(),
};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyInit__parquet_rs() -> *mut PyObject {
    unsafe { PyModule_Create2(&raw mut MODULE_DEF, PYTHON_ABI_VERSION) }
}
