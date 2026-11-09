use std::io::Write;
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

use numpy::{
    PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray1,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::simd::{decode_pattern_u32, encode_pattern_u32};
use crate::{get_pool, view_pybuffer_u8, THREAD_ZSTD};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";

const SCHEMA_PREFIX: &[u8] = &[
    0x15, 0x04, 0x19, 0x4c, 0x35, 0x00, 0x18, 0x06, 0x73, 0x63, 0x68, 0x65, 0x6d, 0x61, 0x15, 0x06,
    0x00, 0x15, 0x0c, 0x25, 0x00, 0x18, 0x0a, 0x70, 0x6c, 0x61, 0x63, 0x65, 0x5f, 0x6f, 0x6e, 0x65,
    0x73, 0x00, 0x15, 0x0c, 0x25, 0x00, 0x18, 0x0b, 0x70, 0x6c, 0x61, 0x63, 0x65, 0x5f, 0x6d, 0x75,
    0x6c, 0x74, 0x69, 0x00, 0x15, 0x0c, 0x25, 0x00, 0x18, 0x0b, 0x63, 0x6f, 0x75, 0x6e, 0x74, 0x5f,
    0x6d, 0x75, 0x6c, 0x74, 0x69, 0x00,
];

const CREATED_BY_SUFFIX: &[u8] = &[
    0x18, 0x20, 0x70, 0x61, 0x72, 0x71, 0x75, 0x65, 0x74, 0x2d, 0x63, 0x70, 0x70, 0x2d, 0x61, 0x72,
    0x72, 0x6f, 0x77, 0x20, 0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x20, 0x32, 0x36, 0x2e, 0x30,
    0x2e, 0x31, 0x19, 0x3c, 0x1c, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
];

const COL_ENCODING_STATS_BYTES: &[u8] = &[0x49, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15, 0x02, 0x00];
const COL_NAMES: [&[u8]; 3] = [b"place_ones", b"place_multi", b"count_multi"];

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
fn push_uvarint(out: &mut Vec<u8>, mut val: u64) {
    while val >= 0x80 {
        out.push(((val as u8) & 0x7F) | 0x80);
        val >>= 7;
    }
    out.push(val as u8);
}

#[inline]
fn push_zigzag(out: &mut Vec<u8>, n: i64) {
    let zz = ((n << 1) ^ (n >> 63)) as u64;
    push_uvarint(out, zz);
}

#[inline]
fn try_read_uvarint(buf: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let n_len = buf.len();
    if pos >= n_len {
        return None;
    }
    let b0 = buf[pos] as u64;
    if b0 < 0x80 {
        return Some((b0, pos + 1));
    }
    if pos + 1 >= n_len {
        return None;
    }
    let b1 = buf[pos + 1] as u64;
    if b1 < 0x80 {
        return Some(((b0 & 0x7F) | (b1 << 7), pos + 2));
    }
    let mut n = (b0 & 0x7F) | ((b1 & 0x7F) << 7);
    let mut shift = 14u32;
    pos += 2;
    while pos < n_len {
        let b = buf[pos] as u64;
        pos += 1;
        n |= (b & 0x7F) << shift;
        if b < 0x80 {
            return Some((n, pos));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

#[inline]
fn try_read_zigzag(buf: &[u8], pos: usize) -> Option<(i64, usize)> {
    let (n, next_pos) = try_read_uvarint(buf, pos)?;
    Some((((n >> 1) as i64) ^ -((n & 1) as i64), next_pos))
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn push_col_chunk_meta(
    out: &mut Vec<u8>,
    col_type: i64,
    col_name: &[u8],
    codec: i64,
    num_values: i64,
    total_uncomp: i64,
    total_comp: i64,
    page_offset: i64,
    data_bytes: i64,
) {
    out.extend_from_slice(b"\x26\x00\x1c\x15");
    push_zigzag(out, col_type);
    out.extend_from_slice(b"\x19\x25\x00\x06\x19\x18");
    push_uvarint(out, col_name.len() as u64);
    out.extend_from_slice(col_name);
    out.push(0x15);
    push_zigzag(out, codec);
    out.push(0x16);
    push_zigzag(out, num_values);
    out.push(0x16);
    push_zigzag(out, total_uncomp);
    out.push(0x16);
    push_zigzag(out, total_comp);
    out.push(0x26);
    push_zigzag(out, page_offset);
    out.extend_from_slice(COL_ENCODING_STATS_BYTES);
    out.extend_from_slice(b"\x3c\x16");
    push_zigzag(out, data_bytes);
    out.extend_from_slice(b"\x19\x06\x19\x06\x00\x00\x00");
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
        let c_len = if level <= 0 {
            u_len
        } else {
            cctx.compress(comp_scratch, &scratch[..u_len], level)
                .map_err(|_| -3i32)?
        };
        let pos0 = pos;
        let hdr_end = write_page_header(out_buf, pos, u_len, c_len, page_nr);
        if level <= 0 {
            out_buf[hdr_end..hdr_end + u_len].copy_from_slice(&scratch[..u_len]);
        } else {
            out_buf[hdr_end..hdr_end + c_len].copy_from_slice(&comp_scratch[..c_len]);
        }
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
    col_tc_raw: i64,
    col_elems: usize,
    page_buf: &mut Vec<u8>,
    rg_ps: usize,
    rg_pe: usize,
    p_firsts: &[i64],
    p_lasts: &[i64],
    span_counts: &mut [i64],
) -> i32 {
    let is_zstd = col_tc_raw >= 0;
    let col_tc = col_tc_raw.unsigned_abs() as usize;
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
        let Some((u_len_i, p1)) = try_read_zigzag(mm_u8, pos.saturating_add(3)) else {
            return -1;
        };
        let Some((c_len_i, p2)) = try_read_zigzag(mm_u8, p1.saturating_add(1)) else {
            return -1;
        };
        let Some((page_nr_i, p3)) = try_read_zigzag(mm_u8, p2.saturating_add(2)) else {
            return -1;
        };
        if u_len_i < 0 || c_len_i < 0 || page_nr_i <= 0 {
            return -1;
        }
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
                    let c_start = pos + hdr_len;
                    let page_slice: &[u8] = if !is_zstd {
                        if c_len != u_len || c_start.saturating_add(u_len) > mm_u8.len() {
                            return -1;
                        }
                        &mm_u8[c_start..c_start + u_len]
                    } else {
                        if c_start.saturating_add(c_len) > mm_u8.len() {
                            return -1;
                        }
                        if page_buf.len() < u_len {
                            page_buf.resize(u_len.max(256 * 1024), 0);
                        }
                        match dctx
                            .decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len])
                        {
                            Ok(d_sz) if d_sz == u_len => &page_buf[..u_len],
                            _ => return -1,
                        }
                    };
                    let mut b_off = 0usize;
                    for p_idx in page_ps..page_pe {
                        if b_off.saturating_add(4) > page_slice.len() {
                            return -1;
                        }
                        let bl =
                            u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap())
                                as usize;
                        let k = (bl / 4) as i64;
                        b_off = b_off.saturating_add(4).saturating_add(bl);
                        if b_off > page_slice.len() {
                            return -1;
                        }
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
    col_tc_raw: i64,
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
    let is_zstd = col_tc_raw >= 0;
    let col_tc = col_tc_raw.unsigned_abs() as usize;
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
        let Some((u_len_i, p1)) = try_read_zigzag(mm_u8, pos.saturating_add(3)) else {
            return -1;
        };
        let Some((c_len_i, p2)) = try_read_zigzag(mm_u8, p1.saturating_add(1)) else {
            return -1;
        };
        let Some((page_nr_i, p3)) = try_read_zigzag(mm_u8, p2.saturating_add(2)) else {
            return -1;
        };
        if u_len_i < 0 || c_len_i < 0 || page_nr_i <= 0 {
            return -1;
        }
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
                    let c_start = pos + hdr_len;
                    let page_slice: &[u8] = if !is_zstd {
                        if c_len != u_len || c_start.saturating_add(u_len) > mm_u8.len() {
                            return -1;
                        }
                        &mm_u8[c_start..c_start + u_len]
                    } else {
                        if c_start.saturating_add(c_len) > mm_u8.len() {
                            return -1;
                        }
                        if page_buf.len() < u_len {
                            page_buf.resize(u_len.max(256 * 1024), 0);
                        }
                        match dctx
                            .decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len])
                        {
                            Ok(d_sz) if d_sz == u_len => &page_buf[..u_len],
                            _ => return -1,
                        }
                    };
                    let mut b_off = 0usize;
                    for p_idx in page_ps..page_pe {
                        if b_off.saturating_add(4) > page_slice.len() {
                            return -1;
                        }
                        let bl =
                            u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap())
                                as usize;
                        if bl % 4 != 0 {
                            return -1;
                        }
                        let k = bl / 4;
                        let b_s = b_off + 4;
                        b_off = b_s.saturating_add(bl);
                        if b_off > page_slice.len() {
                            return -1;
                        }

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
                                    if out_idx.saturating_add(k) > out_u32.len() {
                                        return -1;
                                    }
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
    col_tc_raw: i64,
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
    let is_zstd = col_tc_raw >= 0;
    let col_tc = col_tc_raw.unsigned_abs() as usize;
    let mut pos = col_off;
    let col_end = col_off + col_tc;
    let mut page_ps = rg_ps;

    while page_ps < rg_pe && pos < col_end {
        let Some((u_len_i, p1)) = try_read_zigzag(mm_u8, pos.saturating_add(3)) else {
            return Err(-1);
        };
        let Some((c_len_i, p2)) = try_read_zigzag(mm_u8, p1.saturating_add(1)) else {
            return Err(-1);
        };
        let Some((page_nr_i, p3)) = try_read_zigzag(mm_u8, p2.saturating_add(2)) else {
            return Err(-1);
        };
        if u_len_i < 0 || c_len_i < 0 || page_nr_i <= 0 {
            return Err(-1);
        }
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
            let c_start = pos + hdr_len;
            let page_slice: &[u8] = if !is_zstd {
                if c_len != u_len || c_start.saturating_add(u_len) > mm_u8.len() {
                    return Err(-1);
                }
                &mm_u8[c_start..c_start + u_len]
            } else {
                if c_start.saturating_add(c_len) > mm_u8.len() {
                    return Err(-1);
                }
                if page_buf.len() < u_len {
                    page_buf.resize(u_len.max(256 * 1024), 0);
                }
                match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len]) {
                    Ok(d_sz) if d_sz == u_len => &page_buf[..u_len],
                    _ => return Err(-1),
                }
            };
            let mut b_off = 0usize;
            for _ in page_ps..frame {
                if b_off.saturating_add(4) > page_slice.len() {
                    return Err(-1);
                }
                let bl =
                    u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
                b_off = b_off.saturating_add(4).saturating_add(bl);
            }
            if b_off.saturating_add(4) > page_slice.len() {
                return Err(-1);
            }
            let bl = u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
            if bl % 4 != 0 || b_off.saturating_add(4).saturating_add(bl) > page_slice.len() {
                return Err(-1);
            }
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

#[pyfunction]
pub fn parse_parquet_footer<'py>(
    py: Python<'py>,
    footer_bytes: &[u8],
) -> PyResult<Option<(Bound<'py, PyDict>, Bound<'py, PyArray3<i64>>)>> {
    if !footer_bytes.starts_with(SCHEMA_PREFIX) {
        return Ok(None);
    }
    let mut pos = SCHEMA_PREFIX.len();
    let n_len = footer_bytes.len();
    if pos >= n_len || footer_bytes[pos] != 0x16 {
        return Ok(None);
    }
    let (_, p1) = match try_read_zigzag(footer_bytes, pos + 1) {
        Some(v) => v,
        None => return Ok(None),
    };
    pos = p1;
    if pos + 1 >= n_len || footer_bytes[pos] != 0x19 {
        return Ok(None);
    }
    let hdr_b = footer_bytes[pos + 1];
    let n_rgs: usize = if hdr_b == 0xFC {
        let (v, p2) = match try_read_uvarint(footer_bytes, pos + 2) {
            Some(v) => v,
            None => return Ok(None),
        };
        pos = p2;
        v as usize
    } else {
        pos += 2;
        (hdr_b >> 4) as usize
    };

    let mut col_info_vec = vec![0i64; n_rgs * 9];
    let col_name_lens = [10usize, 11usize, 11usize];
    let mut sum_ones = 0i64;
    let mut sum_multi = 0i64;
    for rg in 0..n_rgs {
        if pos + 1 >= n_len || footer_bytes[pos] != 0x19 || footer_bytes[pos + 1] != 0x3C {
            return Ok(None);
        }
        pos += 2;
        for c in 0..3 {
            if pos + 4 >= n_len || footer_bytes[pos] != 0x26 || footer_bytes[pos + 3] != 0x15 {
                return Ok(None);
            }
            let (_, p_ct) = match try_read_zigzag(footer_bytes, pos + 4) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_ct + 7 + col_name_lens[c];
            if pos >= n_len || footer_bytes[pos] != 0x15 {
                return Ok(None);
            }
            let (codec, p_cd) = match try_read_zigzag(footer_bytes, pos + 1) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_cd;
            if codec != 6 && codec != 0 {
                return Ok(None);
            }
            if pos >= n_len || footer_bytes[pos] != 0x16 {
                return Ok(None);
            }
            let (_, p_nv) = match try_read_zigzag(footer_bytes, pos + 1) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_nv;
            if pos >= n_len || footer_bytes[pos] != 0x16 {
                return Ok(None);
            }
            let (_, p_tu) = match try_read_zigzag(footer_bytes, pos + 1) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_tu;
            if pos >= n_len || footer_bytes[pos] != 0x16 {
                return Ok(None);
            }
            let (tc, p_tc) = match try_read_zigzag(footer_bytes, pos + 1) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_tc;
            if pos >= n_len || footer_bytes[pos] != 0x26 {
                return Ok(None);
            }
            let (poff, p_po) = match try_read_zigzag(footer_bytes, pos + 1) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_po + 9;
            if pos + 1 >= n_len || footer_bytes[pos] != 0x3C || footer_bytes[pos + 1] != 0x16 {
                return Ok(None);
            }
            let (data_bytes, p_db) = match try_read_zigzag(footer_bytes, pos + 2) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_db + 7;
            let n_elems = data_bytes >> 2;
            col_info_vec[rg * 9 + c * 3] = poff;
            col_info_vec[rg * 9 + c * 3 + 1] = if codec == 0 { -tc } else { tc };
            col_info_vec[rg * 9 + c * 3 + 2] = n_elems;
            if c == 0 {
                sum_ones += n_elems;
            } else if c == 1 {
                sum_multi += n_elems;
            }
        }
        for _ in 0..4 {
            if pos >= n_len {
                return Ok(None);
            }
            let (_, p_next) = match try_read_zigzag(footer_bytes, pos + 1) {
                Some(v) => v,
                None => return Ok(None),
            };
            pos = p_next;
        }
        if pos >= n_len || footer_bytes[pos] != 0x00 {
            return Ok(None);
        }
        pos += 1;
    }

    if pos + 2 > n_len || footer_bytes[pos] != 0x19 {
        return Ok(None);
    }
    let kv_hdr = footer_bytes[pos + 1];
    let n_kv: usize = if kv_hdr == 0xFC {
        let (v, p_kv) = match try_read_uvarint(footer_bytes, pos + 2) {
            Some(v) => v,
            None => return Ok(None),
        };
        pos = p_kv;
        v as usize
    } else {
        pos += 2;
        (kv_hdr >> 4) as usize
    };

    let mut version = "2";
    let mut position_encoding = "absolute";
    let mut num_pix: Option<i64> = None;
    let mut num_data: Option<i64> = None;
    let mut rg_pats: i64 = 1;
    let mut shuffle = false;
    let mut payload_codec = "none";

    for _ in 0..n_kv {
        if pos >= n_len || footer_bytes[pos] != 0x18 {
            return Ok(None);
        }
        let (k_len_u, pk) = match try_read_uvarint(footer_bytes, pos + 1) {
            Some(v) => v,
            None => return Ok(None),
        };
        let k_len = k_len_u as usize;
        pos = pk;
        if pos + k_len > n_len {
            return Ok(None);
        }
        let k_slice = &footer_bytes[pos..pos + k_len];
        pos += k_len;

        if pos >= n_len || footer_bytes[pos] != 0x18 {
            return Ok(None);
        }
        let (v_len_u, pv) = match try_read_uvarint(footer_bytes, pos + 1) {
            Some(v) => v,
            None => return Ok(None),
        };
        let v_len = v_len_u as usize;
        pos = pv;
        if pos + v_len > n_len {
            return Ok(None);
        }
        let v_slice = &footer_bytes[pos..pos + v_len];
        pos += v_len;

        if pos >= n_len || footer_bytes[pos] != 0x00 {
            return Ok(None);
        }
        pos += 1;

        let v_str = match std::str::from_utf8(v_slice) {
            Ok(s) => s,
            Err(_) => continue,
        };
        match k_slice {
            b"version" => version = v_str,
            b"position_encoding" => position_encoding = v_str,
            b"num_pix" => num_pix = v_str.parse::<i64>().ok(),
            b"num_data" => num_data = v_str.parse::<i64>().ok(),
            b"rg_pats" => {
                if let Ok(v) = v_str.parse::<i64>() {
                    rg_pats = v;
                }
            }
            b"shuffle" => shuffle = v_slice == b"1",
            b"payload_codec" => payload_codec = v_str,
            _ => {}
        }
    }

    if version != "2" {
        return Err(PyValueError::new_err(format!(
            "Unsupported Parquet EMC format version {version:?}"
        )));
    }
    if position_encoding != "absolute" && position_encoding != "delta" {
        return Err(PyValueError::new_err(format!(
            "Unsupported Parquet position encoding {position_encoding:?}"
        )));
    }
    let (Some(npix), Some(ndata)) = (num_pix, num_data) else {
        return Ok(None);
    };

    let meta_dict = PyDict::new(py);
    meta_dict.set_item("version", version)?;
    meta_dict.set_item("num_pix", npix)?;
    meta_dict.set_item("num_data", ndata)?;
    meta_dict.set_item("rg_pats", rg_pats)?;
    meta_dict.set_item("position_encoding", position_encoding)?;
    meta_dict.set_item("shuffle", shuffle)?;
    meta_dict.set_item("payload_codec", payload_codec)?;
    meta_dict.set_item("num_ones", sum_ones)?;
    meta_dict.set_item("num_multi", sum_multi)?;

    let arr1 = PyArray1::from_vec(py, col_info_vec);
    let arr3 = arr1.reshape([n_rgs, 3, 3])?;
    Ok(Some((meta_dict, arr3)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn write_batch_zstd<'py>(
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
) -> PyResult<Bound<'py, PyArray3<i64>>> {
    let po_src = b_po.as_slice()?;
    let pm_src = b_pm.as_slice()?;
    let cm_src = b_cm.as_slice()?;
    let ones_idx = b_ones_idx.as_slice()?;
    let multi_idx = b_multi_idx.as_slice()?;
    let b_rgs = (num_data + rg_pats - 1) / rg_pats;
    let n_workers = n_workers.max(1);

    let mut meta_vec = vec![0i64; b_rgs * 9];
    let meta_addr = meta_vec.as_mut_ptr() as usize;

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
                                let _g = pipe_mtx.lock().unwrap();
                                abort_err.store(-4, Ordering::Relaxed);
                                drop(_g);
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
                                let _g = pipe_mtx.lock().unwrap();
                                abort_err.store(res, Ordering::Relaxed);
                                drop(_g);
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
    let arr1 = PyArray1::from_vec(py, meta_vec);
    Ok(arr1.reshape([b_rgs, 3, 3])?)
}

#[pyfunction]
pub fn write_parquet_footer<'py>(
    py: Python<'py>,
    fd: i32,
    num_data: i64,
    rg_pats: usize,
    batch_pats: Vec<usize>,
    meta_batches: Vec<PyReadonlyArray3<'py, i64>>,
    kv_items: Vec<(Vec<u8>, Vec<u8>)>,
) -> PyResult<()> {
    let mut slices = Vec::with_capacity(meta_batches.len());
    for mb in &meta_batches {
        slices.push(mb.as_slice()?);
    }

    let mut codec = 6i64;
    for (k, v) in &kv_items {
        if k == b"compression" && v == b"none" {
            codec = 0;
        }
    }

    let ok = py.detach(|| {
        let mut total_rgs = 0usize;
        for &bp in &batch_pats {
            if bp > 0 {
                total_rgs += (bp + rg_pats - 1) / rg_pats;
            }
        }

        let mut footer = Vec::with_capacity(SCHEMA_PREFIX.len() + total_rgs * 220 + 512);
        footer.extend_from_slice(SCHEMA_PREFIX);
        footer.push(0x16);
        push_zigzag(&mut footer, num_data);

        if total_rgs < 15 {
            footer.extend_from_slice(&[0x19, ((total_rgs as u8) << 4) | 0x0C]);
        } else {
            footer.extend_from_slice(b"\x19\xfc");
            push_uvarint(&mut footer, total_rgs as u64);
        }

        let mut file_pos = 4i64;
        for (&b_pats, &meta_slice) in batch_pats.iter().zip(slices.iter()) {
            if b_pats == 0 {
                continue;
            }
            let b_rgs = (b_pats + rg_pats - 1) / rg_pats;
            for rg in 0..b_rgs {
                let ps = rg * rg_pats;
                let pe = ((rg + 1) * rg_pats).min(b_pats);
                let nr = (pe - ps) as i64;
                let rg_start_pos = file_pos;
                let mut total_uncomp_rg = 0i64;
                let mut total_comp_rg = 0i64;

                footer.extend_from_slice(b"\x19\x3c");
                for c in 0..3 {
                    let tu = meta_slice[rg * 9 + c * 3];
                    let tc = meta_slice[rg * 9 + c * 3 + 1];
                    let n_el = meta_slice[rg * 9 + c * 3 + 2];
                    push_col_chunk_meta(
                        &mut footer,
                        6,
                        COL_NAMES[c],
                        codec,
                        nr,
                        tu,
                        tc,
                        file_pos,
                        n_el * 4,
                    );
                    file_pos += tc;
                    total_uncomp_rg += tu;
                    total_comp_rg += tc;
                }

                footer.push(0x16);
                push_zigzag(&mut footer, total_uncomp_rg);
                footer.push(0x16);
                push_zigzag(&mut footer, nr);
                footer.push(0x26);
                push_zigzag(&mut footer, rg_start_pos);
                footer.push(0x16);
                push_zigzag(&mut footer, total_comp_rg);
                footer.push(0x00);
            }
        }

        let n_kv = kv_items.len();
        if n_kv > 0 {
            if n_kv < 15 {
                footer.extend_from_slice(&[0x19, ((n_kv as u8) << 4) | 0x0C]);
            } else {
                footer.extend_from_slice(b"\x19\xfc");
                push_uvarint(&mut footer, n_kv as u64);
            }
            for (k, v) in &kv_items {
                footer.push(0x18);
                push_uvarint(&mut footer, k.len() as u64);
                footer.extend_from_slice(k);
                footer.push(0x18);
                push_uvarint(&mut footer, v.len() as u64);
                footer.extend_from_slice(v);
                footer.push(0x00);
            }
        }

        footer.push(if n_kv > 0 { 0x18 } else { 0x28 });
        footer.extend_from_slice(&CREATED_BY_SUFFIX[1..]);
        let footer_len = footer.len() as u32;
        footer.extend_from_slice(&footer_len.to_le_bytes());
        footer.extend_from_slice(PARQUET_MAGIC);

        write_all_fd(fd, &footer)
    });

    if !ok {
        return Err(PyRuntimeError::new_err(
            "Failed to write Parquet footer in _parquet_rs",
        ));
    }
    Ok(())
}

fn read_col_counts_rg(
    dctx: &mut zstd_safe::DCtx<'static>,
    mm_u8: &[u8],
    col_off: usize,
    col_tc_raw: i64,
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

    let is_zstd = col_tc_raw >= 0;
    let col_tc = col_tc_raw.unsigned_abs() as usize;
    let mut pos = col_off;
    let col_end = col_off + col_tc;
    let mut page_ps = rg_ps;

    while page_ps < rg_pe && pos < col_end {
        let Some((u_len_i, p1)) = try_read_zigzag(mm_u8, pos + 3) else {
            return -1;
        };
        let Some((c_len_i, p2)) = try_read_zigzag(mm_u8, p1 + 1) else {
            return -1;
        };
        let Some((page_nr_i, p3)) = try_read_zigzag(mm_u8, p2 + 2) else {
            return -1;
        };
        if u_len_i < 0 || c_len_i < 0 || page_nr_i <= 0 {
            return -1;
        }
        let u_len = u_len_i as usize;
        let c_len = c_len_i as usize;
        let page_nr = page_nr_i as usize;
        let hdr_len = (p3 + 10) - pos;
        let page_pe = page_ps + page_nr;
        if page_pe > rg_pe {
            return -1;
        }
        let page_elems = (u_len / 4).saturating_sub(page_nr);

        if page_elems == 0 {
            out_counts[page_ps - rg_ps..page_pe - rg_ps].fill(0);
        } else if page_nr == 1 {
            out_counts[page_ps - rg_ps] = page_elems as u32;
        } else {
            let c_start = pos + hdr_len;
            let page_slice: &[u8] = if !is_zstd {
                if c_len != u_len || c_start + u_len > mm_u8.len() {
                    return -1;
                }
                &mm_u8[c_start..c_start + u_len]
            } else {
                if c_start + c_len > mm_u8.len() {
                    return -1;
                }
                if page_buf.len() < u_len {
                    page_buf.resize(u_len.max(256 * 1024), 0);
                }
                match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len]) {
                    Ok(d_sz) if d_sz == u_len => &page_buf[..u_len],
                    _ => return -1,
                }
            };
            let mut b_off = 0usize;
            for p_idx in page_ps..page_pe {
                if b_off + 4 > page_slice.len() {
                    return -1;
                }
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
pub fn read_ones_multi<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    rg_pats: usize,
    num_data: usize,
    n_workers: usize,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let n_rgs = col_info.len() / 9;
    let n_workers = n_workers.max(1);

    let out_ones_arr = unsafe { PyArray1::<u32>::new(py, [num_data], false) };
    let out_multi_arr = unsafe { PyArray1::<u32>::new(py, [num_data], false) };
    let ones_addr = unsafe { out_ones_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { out_multi_arr.as_slice_mut()? }.as_mut_ptr() as usize;

    let read_rg = |rg: usize| -> Result<(), i32> {
        let ps = rg * rg_pats;
        let pe = ((rg + 1) * rg_pats).min(num_data);
        let nr = pe - ps;
        let ones_rg =
            unsafe { std::slice::from_raw_parts_mut((ones_addr as *mut u32).add(ps), nr) };
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
                col_info[rg * 9 + 1],
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
                col_info[rg * 9 + 4],
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
    Ok((out_ones_arr, out_multi_arr))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn read_patterns<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    b_ranges: PyReadonlyArray2<'py, i64>,
    b_ones_idx: Option<PyReadonlyArray1<'py, u64>>,
    b_multi_idx: Option<PyReadonlyArray1<'py, u64>>,
    rg_pats: usize,
    num_data: usize,
    is_delta: bool,
    is_shuffled: bool,
    n_workers: usize,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
)> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let ranges_slice = b_ranges.as_slice()?;
    let ones_idx_opt = b_ones_idx.as_ref().map(|a| a.as_slice()).transpose()?;
    let multi_idx_opt = b_multi_idx.as_ref().map(|a| a.as_slice()).transpose()?;

    let prep_res = py.detach(|| -> Result<_, i32> {
        let n_in_ranges = ranges_slice.len() / 2;
        let nd_i64 = num_data as i64;
        for i in 0..n_in_ranges {
            let s = ranges_slice[i * 2];
            let e = ranges_slice[i * 2 + 1];
            if s < 0 || e > nd_i64 || s > e {
                return Err(-2);
            }
        }
        let is_full = n_in_ranges == 1
            && ranges_slice[0] == 0
            && (ranges_slice[1] as usize) == num_data;
        let rp = rg_pats as i64;

        let mut total_spans = 0usize;
        for i in 0..n_in_ranges {
            let s = ranges_slice[i * 2];
            let e = ranges_slice[i * 2 + 1];
            if e > s {
                let rg_s = (s / rp) as usize;
                let rg_e = ((e - 1) / rp) as usize;
                total_spans += rg_e - rg_s + 1;
            }
        }
        if total_spans == 0 {
            return Ok(None);
        }

        let mut rgs = Vec::with_capacity(total_spans);
        let mut firsts = Vec::with_capacity(total_spans);
        let mut lasts = Vec::with_capacity(total_spans);
        let mut dst_pats = Vec::with_capacity(total_spans + 1);
        dst_pats.push(0i64);
        let mut acc_pats = 0i64;

        for i in 0..n_in_ranges {
            let s = ranges_slice[i * 2];
            let e = ranges_slice[i * 2 + 1];
            if e > s {
                let rg_s = (s / rp) as usize;
                let rg_e = ((e - 1) / rp) as usize;
                for rg in rg_s..=rg_e {
                    let f = s.max((rg as i64) * rp);
                    let l = e.min(((rg + 1) as i64) * rp);
                    rgs.push(rg);
                    firsts.push(f);
                    lasts.push(l);
                    acc_pats += l - f;
                    dst_pats.push(acc_pats);
                }
            }
        }
        let total_sel_pats = acc_pats as usize;

        let (unique_rgs, group_starts, group_stops, group_span_ids) = if is_full {
            let urgs: Vec<usize> = (0..total_spans).collect();
            let gstarts: Vec<usize> = (0..total_spans).collect();
            let gstops: Vec<usize> = (1..=total_spans).collect();
            let gsids: Vec<usize> = (0..total_spans).collect();
            (urgs, gstarts, gstops, gsids)
        } else {
            let mut gsids: Vec<usize> = (0..total_spans).collect();
            gsids.sort_unstable_by_key(|&sid| (rgs[sid], firsts[sid], sid));
            let mut urgs = Vec::with_capacity(total_spans);
            let mut gstarts = Vec::with_capacity(total_spans);
            let mut gstops = Vec::with_capacity(total_spans);
            let mut i = 0usize;
            while i < total_spans {
                let rg = rgs[gsids[i]];
                urgs.push(rg);
                gstarts.push(i);
                i += 1;
                while i < total_spans && rgs[gsids[i]] == rg {
                    i += 1;
                }
                gstops.push(i);
            }
            (urgs, gstarts, gstops, gsids)
        };

        let n_groups = unique_rgs.len();
        let n_workers = n_workers.max(1).min(n_groups.max(1));

        let mut dst_ones = Vec::with_capacity(total_spans + 1);
        let mut dst_multi = Vec::with_capacity(total_spans + 1);
        dst_ones.push(0i64);
        dst_multi.push(0i64);

        if is_full {
            let mut ao = 0i64;
            let mut am = 0i64;
            for rg in 0..total_spans {
                ao += col_info[rg * 9 + 2];
                am += col_info[rg * 9 + 5];
                dst_ones.push(ao);
                dst_multi.push(am);
            }
        } else if let (Some(ones_idx), Some(multi_idx)) = (ones_idx_opt, multi_idx_opt) {
            let mut ao = 0i64;
            let mut am = 0i64;
            for sid in 0..total_spans {
                let f = firsts[sid] as usize;
                let l = lasts[sid] as usize;
                ao += (ones_idx[l] - ones_idx[f]) as i64;
                am += (multi_idx[l] - multi_idx[f]) as i64;
                dst_ones.push(ao);
                dst_multi.push(am);
            }
        } else {
            let mut ones_lens = vec![0i64; total_spans];
            let mut multi_lens = vec![0i64; total_spans];
            let ones_lens_addr = ones_lens.as_mut_ptr() as usize;
            let multi_lens_addr = multi_lens.as_mut_ptr() as usize;

            let count_group = |g_idx: usize| -> Result<(), i32> {
                let rg = unique_rgs[g_idx];
                let ps = rg * rg_pats;
                let pe = ((rg + 1) * rg_pats).min(num_data);
                let gs0 = group_starts[g_idx];
                let gs1 = group_stops[g_idx];
                let n_sp = gs1 - gs0;

                let mut p_firsts = Vec::with_capacity(n_sp);
                let mut p_lasts = Vec::with_capacity(n_sp);
                let mut c_o = vec![0i64; n_sp];
                let mut c_m = vec![0i64; n_sp];

                for i in 0..n_sp {
                    let sid = group_span_ids[gs0 + i];
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
                        col_info[rg * 9 + 1],
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
                        col_info[rg * 9 + 4],
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
                    let sid = group_span_ids[gs0 + i];
                    unsafe {
                        *(ones_lens_addr as *mut i64).add(sid) = c_o[i];
                        *(multi_lens_addr as *mut i64).add(sid) = c_m[i];
                    }
                }
                Ok(())
            };

            if n_workers <= 1 || n_groups <= 1 {
                (0..n_groups).try_for_each(count_group)?;
            } else {
                get_pool(n_workers)
                    .install(|| (0..n_groups).into_par_iter().try_for_each(count_group))?;
            }

            let mut ao = 0i64;
            let mut am = 0i64;
            for sid in 0..total_spans {
                ao += ones_lens[sid];
                am += multi_lens[sid];
                dst_ones.push(ao);
                dst_multi.push(am);
            }
        }

        Ok(Some((
            firsts,
            lasts,
            dst_pats,
            dst_ones,
            dst_multi,
            unique_rgs,
            group_starts,
            group_stops,
            group_span_ids,
            total_sel_pats,
            n_workers,
        )))
    });

    let prep = match prep_res {
        Ok(Some(p)) => p,
        Ok(None) => {
            return Ok((
                PyArray1::from_vec(py, Vec::new()),
                PyArray1::from_vec(py, Vec::new()),
                PyArray1::from_vec(py, Vec::new()),
                PyArray1::from_vec(py, Vec::new()),
                PyArray1::from_vec(py, Vec::new()),
            ));
        }
        Err(-2) => return Err(PyIndexError::new_err("Pattern range is out of bounds")),
        Err(_) => return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file")),
    };

    let (
        firsts,
        lasts,
        dst_pats,
        dst_ones,
        dst_multi,
        unique_rgs,
        group_starts,
        group_stops,
        group_span_ids,
        total_sel_pats,
        n_workers,
    ) = prep;

    let total_ones = *dst_ones.last().unwrap() as usize;
    let total_multi = *dst_multi.last().unwrap() as usize;
    let n_groups = unique_rgs.len();

    let out_ones_arr = unsafe { PyArray1::<u32>::new(py, [total_sel_pats], false) };
    let out_multi_arr = unsafe { PyArray1::<u32>::new(py, [total_sel_pats], false) };
    let out_po_arr = unsafe { PyArray1::<u32>::new(py, [total_ones], false) };
    let out_pm_arr = unsafe { PyArray1::<u32>::new(py, [total_multi], false) };
    let out_cm_arr = unsafe { PyArray1::<u32>::new(py, [total_multi], false) };

    let out_ones_addr = unsafe { out_ones_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let out_multi_addr = unsafe { out_multi_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let out_po_addr = unsafe { out_po_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let out_pm_addr = unsafe { out_pm_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let out_cm_addr = unsafe { out_cm_arr.as_slice_mut()? }.as_mut_ptr() as usize;

    let dec_res = py.detach(|| -> Result<(), i32> {
        let decode_group = |g_idx: usize| -> Result<(), i32> {
            let rg = unique_rgs[g_idx];
            let ps = rg * rg_pats;
            let pe = ((rg + 1) * rg_pats).min(num_data);
            let gs0 = group_starts[g_idx];
            let gs1 = group_stops[g_idx];
            let n_sp = gs1 - gs0;

            let mut p_firsts = Vec::with_capacity(n_sp);
            let mut p_lasts = Vec::with_capacity(n_sp);
            let mut d_pats = Vec::with_capacity(n_sp);
            let mut d_ones = Vec::with_capacity(n_sp);
            let mut d_multi = Vec::with_capacity(n_sp);
            let mut cur_dst = vec![0usize; n_sp];
            let mut need_m_payload = false;

            for i in 0..n_sp {
                let sid = group_span_ids[gs0 + i];
                p_firsts.push(firsts[sid]);
                p_lasts.push(lasts[sid]);
                d_pats.push(dst_pats[sid]);
                d_ones.push(dst_ones[sid]);
                d_multi.push(dst_multi[sid]);
                if dst_multi[sid + 1] > dst_multi[sid] {
                    need_m_payload = true;
                }
            }

            let ones_sl = unsafe {
                std::slice::from_raw_parts_mut(out_ones_addr as *mut u32, total_sel_pats)
            };
            let multi_sl = unsafe {
                std::slice::from_raw_parts_mut(out_multi_addr as *mut u32, total_sel_pats)
            };
            let po_sl =
                unsafe { std::slice::from_raw_parts_mut(out_po_addr as *mut u32, total_ones) };
            let pm_sl =
                unsafe { std::slice::from_raw_parts_mut(out_pm_addr as *mut u32, total_multi) };
            let cm_sl =
                unsafe { std::slice::from_raw_parts_mut(out_cm_addr as *mut u32, total_multi) };

            let rc = THREAD_ZSTD.with(|cell| {
                let mut ctx = cell.borrow_mut();
                let zctx = &mut *ctx;
                let dctx = &mut zctx.dctx;
                let scratch = &mut zctx.scratch;

                let r = decode_col_pages_inner(
                    dctx,
                    mm_u8,
                    col_info[rg * 9] as usize,
                    col_info[rg * 9 + 1],
                    col_info[rg * 9 + 2] as usize,
                    scratch,
                    ps,
                    pe,
                    po_sl,
                    Some(ones_sl),
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
                    col_info[rg * 9 + 4],
                    col_info[rg * 9 + 5] as usize,
                    scratch,
                    ps,
                    pe,
                    pm_sl,
                    Some(multi_sl),
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
                        col_info[rg * 9 + 7],
                        col_info[rg * 9 + 8] as usize,
                        scratch,
                        ps,
                        pe,
                        cm_sl,
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

        if n_workers <= 1 || n_groups <= 1 {
            (0..n_groups).try_for_each(decode_group)
        } else {
            get_pool(n_workers).install(|| (0..n_groups).into_par_iter().try_for_each(decode_group))
        }
    });

    if dec_res.is_err() {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    }
    Ok((
        out_ones_arr,
        out_multi_arr,
        out_po_arr,
        out_pm_arr,
        out_cm_arr,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_single_frame<'py>(
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
            col_info[rg * 9 + 1],
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
            col_info[rg * 9 + 4],
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
                col_info[rg * 9 + 7],
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
pub fn decode_col<'py>(
    py: Python<'py>,
    b_mm: PyBuffer<u8>,
    b_col: PyReadonlyArray3<'py, i64>,
    c_idx: usize,
    rg_pats: usize,
    num_data: usize,
    is_delta: bool,
    is_shuffled: bool,
    n_workers: usize,
) -> PyResult<Bound<'py, PyArray1<u32>>> {
    let mm_u8 = view_pybuffer_u8(&b_mm)?;
    let col_info = b_col.as_slice()?;
    let n_rgs = col_info.len() / 9;
    let n_workers = n_workers.max(1);

    let mut rg_offs = Vec::with_capacity(n_rgs + 1);
    rg_offs.push(0i64);
    let mut acc = 0i64;
    for rg in 0..n_rgs {
        acc += col_info[rg * 9 + c_idx * 3 + 2];
        rg_offs.push(acc);
    }
    let total_elems = acc as usize;
    if total_elems == 0 {
        return Ok(PyArray1::from_vec(py, Vec::new()));
    }

    let out_arr = unsafe { PyArray1::<u32>::new(py, [total_elems], false) };
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;

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
        let out_u32 =
            unsafe { std::slice::from_raw_parts_mut(out_addr as *mut u32, total_elems) };

        let rc = THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;
            let dctx = &mut zctx.dctx;
            let scratch = &mut zctx.scratch;
            decode_col_pages_inner(
                dctx,
                mm_u8,
                col_info[rg * 9 + c_idx * 3] as usize,
                col_info[rg * 9 + c_idx * 3 + 1],
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
    Ok(out_arr)
}

#[pyfunction]
pub fn split_ranges_by_rg<'py>(
    py: Python<'py>,
    b_ranges: PyReadonlyArray2<'py, i64>,
    rg_pats: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let ranges_slice = b_ranges.as_slice()?;
    let n_in = ranges_slice.len() / 2;
    let rp = rg_pats as i64;
    let mut total_spans = 0usize;
    for i in 0..n_in {
        let s = ranges_slice[i * 2];
        let e = ranges_slice[i * 2 + 1];
        if e > s {
            let rg_s = (s / rp) as usize;
            let rg_e = ((e - 1) / rp) as usize;
            total_spans += rg_e - rg_s + 1;
        }
    }
    let mut rgs = Vec::with_capacity(total_spans);
    let mut firsts = Vec::with_capacity(total_spans);
    let mut lasts = Vec::with_capacity(total_spans);
    for i in 0..n_in {
        let s = ranges_slice[i * 2];
        let e = ranges_slice[i * 2 + 1];
        if e > s {
            let rg_s = s / rp;
            let rg_e = (e - 1) / rp;
            for rg in rg_s..=rg_e {
                rgs.push(rg);
                firsts.push(s.max(rg * rp));
                lasts.push(e.min((rg + 1) * rp));
            }
        }
    }
    Ok((
        PyArray1::from_vec(py, rgs),
        PyArray1::from_vec(py, firsts),
        PyArray1::from_vec(py, lasts),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn encode_rg_u8<'py>(
    py: Python<'py>,
    b_src: PyReadonlyArray1<'py, u32>,
    b_offs: PyReadonlyArray1<'py, u64>,
    ps: usize,
    pe: usize,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let src = b_src.as_slice()?;
    let offs = b_offs.as_slice()?;
    let s_rg = offs[ps] as usize;
    let e_rg = offs[pe] as usize;
    let n_rg = e_rg - s_rg;
    if n_rg == 0 {
        return Ok(PyArray1::from_vec(py, Vec::new()));
    }
    let mut out_u8: Vec<u8> = Vec::with_capacity(n_rg * 4);
    unsafe {
        out_u8.set_len(n_rg * 4);
    }
    for p in ps..pe {
        let s = offs[p] as usize;
        let e = offs[p + 1] as usize;
        if e > s {
            let b_s = (s - s_rg) * 4;
            let rc = encode_pattern_u32(
                &src[s..e],
                &mut out_u8,
                b_s,
                is_delta,
                is_shuffled,
                check_sorted,
            );
            if rc == -2 {
                return Err(PyValueError::new_err(
                    "position values must be sorted within every pattern",
                ));
            }
        }
    }
    Ok(PyArray1::from_vec(py, out_u8))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_rg_spans_u8<'py>(
    b_rg_u8: PyReadonlyArray1<'py, u8>,
    rg_ps: usize,
    b_local_offs: PyReadonlyArray1<'py, i64>,
    mut b_out_u32: PyReadwriteArray1<'py, u32>,
    b_firsts: PyReadonlyArray1<'py, i64>,
    b_lasts: PyReadonlyArray1<'py, i64>,
    b_dst_offs: PyReadonlyArray1<'py, i64>,
    is_delta: bool,
    is_shuffled: bool,
) -> PyResult<()> {
    let rg_u8 = b_rg_u8.as_slice()?;
    let local_offs = b_local_offs.as_slice()?;
    let out_u32 = b_out_u32.as_slice_mut()?;
    let p_firsts = b_firsts.as_slice()?;
    let p_lasts = b_lasts.as_slice()?;
    let dst_offs = b_dst_offs.as_slice()?;

    for s_i in 0..p_firsts.len() {
        let f = p_firsts[s_i] as usize;
        let last_p = p_lasts[s_i] as usize;
        let mut out_idx = dst_offs[s_i] as usize;
        for p in f..last_p {
            let loc_p = p - rg_ps;
            let s = local_offs[loc_p] as usize;
            let e = local_offs[loc_p + 1] as usize;
            let k = e - s;
            if k > 0 {
                decode_pattern_u32(
                    rg_u8,
                    s * 4,
                    &mut out_u32[out_idx..out_idx + k],
                    is_delta,
                    is_shuffled,
                );
                out_idx += k;
            }
        }
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_parquet_footer, m)?)?;
    m.add_function(wrap_pyfunction!(write_batch_zstd, m)?)?;
    m.add_function(wrap_pyfunction!(write_parquet_footer, m)?)?;
    m.add_function(wrap_pyfunction!(read_ones_multi, m)?)?;
    m.add_function(wrap_pyfunction!(read_patterns, m)?)?;
    m.add_function(wrap_pyfunction!(decode_single_frame, m)?)?;
    m.add_function(wrap_pyfunction!(decode_col, m)?)?;
    m.add_function(wrap_pyfunction!(split_ranges_by_rg, m)?)?;
    m.add_function(wrap_pyfunction!(encode_rg_u8, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rg_spans_u8, m)?)?;
    Ok(())
}
