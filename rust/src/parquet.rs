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
use crate::{
    as_c_slice_2d, as_c_slice_3d, get_pool, view_pybuffer_u8_detachable, RawFileTarget, THREAD_ZSTD,
};

type FooterParseResult<'py> = Option<(Bound<'py, PyDict>, Bound<'py, PyArray3<i64>>)>;
type PairArraysU32<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>);
type TripleArraysU32<'py> = (
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
);
type TripleVecU32 = (Vec<u32>, Vec<u32>, Vec<u32>);
type SplitRangesArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

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

const COL_ENCODING_STATS_PREFIX: &[u8] = &[0x49, 0x1c, 0x15, 0x00, 0x15, 0x00, 0x15];
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
        if shift == 63 && b > 1 {
            return None;
        }
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
    num_pages: i64,
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
    out.extend_from_slice(COL_ENCODING_STATS_PREFIX);
    push_zigzag(out, num_pages);
    out.push(0x00);
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
) -> Result<usize, i32> {
    let u_i32 = i32::try_from(u_len).map_err(|_| -5i32)?;
    let c_i32 = i32::try_from(c_len).map_err(|_| -5i32)?;
    let nv_i32 = i32::try_from(num_vals).map_err(|_| -5i32)?;
    out_buf[pos] = 0x15;
    out_buf[pos + 1] = 0x00;
    out_buf[pos + 2] = 0x15;
    pos = write_zigzag_i32(out_buf, pos + 3, u_i32);
    out_buf[pos] = 0x15;
    pos = write_zigzag_i32(out_buf, pos + 1, c_i32);
    out_buf[pos] = 0x2C;
    out_buf[pos + 1] = 0x15;
    pos = write_zigzag_i32(out_buf, pos + 2, nv_i32);
    out_buf[pos..pos + 9].copy_from_slice(&[0x15, 0x00, 0x15, 0x06, 0x15, 0x06, 0x1C, 0x00, 0x00]);
    out_buf[pos + 9] = 0x00;
    Ok(pos + 10)
}

#[allow(clippy::too_many_arguments)]
fn encode_payload_col_pages(
    cctx: &mut zstd_safe::CCtx<'static>,
    is_compressed: bool,
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
        let page_elems = usize::try_from(pat_offsets[p1] - pat_offsets[p0]).map_err(|_| -5i32)?;
        let u_len = page_nr
            .checked_add(page_elems)
            .and_then(|x| x.checked_mul(4))
            .ok_or(-5i32)?;
        if u_len > scratch.len() {
            return Err(-5);
        }
        let mut b_off = 0usize;
        for p in p0..p1 {
            let s = pat_offsets[p] as usize;
            let e = pat_offsets[p + 1] as usize;
            let k = e - s;
            let byte_len_usize = k.checked_mul(4).ok_or(-5i32)?;
            let byte_len = u32::try_from(byte_len_usize).map_err(|_| -5i32)?;
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
                b_off += byte_len_usize;
            }
        }
        let c_len = if !is_compressed {
            u_len
        } else {
            cctx.compress(comp_scratch, &scratch[..u_len], level)
                .map_err(|_| -3i32)?
        };
        let pos0 = pos;
        let hdr_end = write_page_header(out_buf, pos, u_len, c_len, page_nr)?;
        if !is_compressed {
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
    is_compressed: bool,
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
) -> Result<usize, i32> {
    let ps = rg * rg_pats;
    let pe = ((rg + 1) * rg_pats).min(num_data);
    let nr = pe - ps;
    let no = usize::try_from(ones_idx[pe] - ones_idx[ps]).map_err(|_| -5i32)?;
    let nm = usize::try_from(multi_idx[pe] - multi_idx[ps]).map_err(|_| -5i32)?;

    let mut max_page_elems = 0usize;
    let mut p0 = ps;
    while p0 < pe {
        let p1 = (p0 + page_pats).min(pe);
        let po_e = usize::try_from(ones_idx[p1] - ones_idx[p0]).map_err(|_| -5i32)?;
        let pm_e = usize::try_from(multi_idx[p1] - multi_idx[p0]).map_err(|_| -5i32)?;
        max_page_elems = max_page_elems.max(po_e).max(pm_e);
        p0 = p1;
    }

    let need_sc = max_page_elems
        .checked_add(page_pats)
        .and_then(|x| x.checked_mul(4))
        .and_then(|x| x.checked_add(64))
        .ok_or(-5i32)?;
    let need_csc = zstd_safe::compress_bound(need_sc)
        .checked_add(64)
        .ok_or(-5i32)?;
    let n_pages = nr.div_ceil(page_pats);
    let po_raw_bytes = no
        .checked_add(nr)
        .and_then(|x| x.checked_mul(4))
        .ok_or(-5i32)?;
    let pm_raw_bytes = nm
        .checked_add(nr)
        .and_then(|x| x.checked_mul(4))
        .ok_or(-5i32)?;
    let hdr_slack = n_pages
        .checked_mul(3)
        .and_then(|x| x.checked_add(2))
        .and_then(|x| x.checked_mul(64))
        .ok_or(-5i32)?;
    let need_out = zstd_safe::compress_bound(po_raw_bytes)
        .checked_add(
            zstd_safe::compress_bound(pm_raw_bytes)
                .checked_mul(2)
                .ok_or(-5i32)?,
        )
        .and_then(|x| x.checked_add(hdr_slack))
        .ok_or(-5i32)?;

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
        let (p0_end, tu0, tc0) = encode_payload_col_pages(
            cctx,
            is_compressed,
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
        )?;
        pos = p0_end;
        meta_rg[0] = tu0;
        meta_rg[1] = tc0;
        meta_rg[2] = no as i64;

        let (p1_end, tu1, tc1) = encode_payload_col_pages(
            cctx,
            is_compressed,
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
        )?;
        pos = p1_end;
        meta_rg[3] = tu1;
        meta_rg[4] = tc1;
        meta_rg[5] = nm as i64;

        let (p2_end, tu2, tc2) = encode_payload_col_pages(
            cctx,
            is_compressed,
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
        )?;
        pos = p2_end;
        meta_rg[6] = tu2;
        meta_rg[7] = tc2;
        meta_rg[8] = nm as i64;
        Ok(pos)
    })
}

/// Validate and parse a Parquet v1 DataPageHeader written by `write_page_header`.
///
/// Returns `(u_len, c_len, page_nr, c_start, next_pos)`.
#[inline]
fn parse_page_header(
    mm_u8: &[u8],
    pos: usize,
    col_end: usize,
    max_u_len: usize,
) -> Option<(usize, usize, usize, usize, usize)> {
    let p0 = pos.checked_add(3)?;
    let (u_len_i, p1) = try_read_zigzag(mm_u8, p0)?;
    let (c_len_i, p2) = try_read_zigzag(mm_u8, p1.checked_add(1)?)?;
    let (page_nr_i, p3) = try_read_zigzag(mm_u8, p2.checked_add(2)?)?;
    if u_len_i < 0 || c_len_i < 0 || page_nr_i <= 0 {
        return None;
    }
    let u_len = usize::try_from(u_len_i).ok()?;
    let c_len = usize::try_from(c_len_i).ok()?;
    let page_nr = usize::try_from(page_nr_i).ok()?;
    if !u_len.is_multiple_of(4) || (u_len / 4) < page_nr || u_len > max_u_len {
        return None;
    }
    let c_start = p3.checked_add(10)?;
    if c_start <= pos {
        return None;
    }
    let next_pos = c_start.checked_add(c_len)?;
    if next_pos > col_end || next_pos > mm_u8.len() {
        return None;
    }
    Some((u_len, c_len, page_nr, c_start, next_pos))
}

#[inline]
fn decompress_or_slice_page<'a>(
    dctx: &mut zstd_safe::DCtx<'static>,
    mm_u8: &'a [u8],
    page_buf: &'a mut Vec<u8>,
    is_zstd: bool,
    c_start: usize,
    c_len: usize,
    u_len: usize,
) -> Option<&'a [u8]> {
    if !is_zstd {
        if c_len != u_len {
            return None;
        }
        Some(&mm_u8[c_start..c_start + u_len])
    } else {
        if page_buf.len() < u_len {
            page_buf.resize(u_len.max(256 * 1024), 0);
        }
        match dctx.decompress(&mut page_buf[..u_len], &mm_u8[c_start..c_start + c_len]) {
            Ok(d_sz) if d_sz == u_len => Some(&page_buf[..u_len]),
            _ => None,
        }
    }
}

#[inline]
fn validate_rg_col_entry(col_info: &[i64], rg: usize, c: usize, mm_len: usize) -> Option<(usize, i64, usize)> {
    let base = rg * 9 + c * 3;
    let poff = col_info[base];
    let tc_raw = col_info[base + 1];
    let elems = col_info[base + 2];
    if poff < 4 || elems < 0 || tc_raw == i64::MIN {
        return None;
    }
    let col_off = usize::try_from(poff).ok()?;
    let col_tc = usize::try_from(tc_raw.unsigned_abs()).ok()?;
    let col_elems = usize::try_from(elems).ok()?;
    let col_end = col_off.checked_add(col_tc)?;
    if col_end > mm_len {
        return None;
    }
    Some((col_off, tc_raw, col_elems))
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
    let Some(col_end) = col_off.checked_add(col_tc) else {
        return -1;
    };
    let max_u_len = col_elems
        .saturating_add(rg_pe - rg_ps)
        .saturating_mul(4);

    let mut pos = col_off;
    let mut page_ps = rg_ps;
    let mut sp_start = 0usize;

    while page_ps < rg_pe && pos < col_end {
        let Some((u_len, c_len, page_nr, c_start, next_pos)) =
            parse_page_header(mm_u8, pos, col_end, max_u_len)
        else {
            return -1;
        };
        let Some(page_pe) = page_ps.checked_add(page_nr) else {
            return -1;
        };
        if page_pe > rg_pe {
            return -1;
        }
        let page_elems = (u_len / 4) - page_nr;

        if page_ps >= max_pat {
            break;
        }
        while sp_start < n_spans && (p_lasts[sp_start] as usize) <= page_ps {
            sp_start += 1;
        }
        if sp_start >= n_spans {
            page_ps = page_pe;
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
                    let Some(page_slice) = decompress_or_slice_page(
                        dctx, mm_u8, page_buf, is_zstd, c_start, c_len, u_len,
                    ) else {
                        return -1;
                    };
                    let mut b_off = 0usize;
                    for p_idx in page_ps..page_pe {
                        if b_off.saturating_add(4) > page_slice.len() {
                            return -1;
                        }
                        let bl =
                            u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap())
                                as usize;
                        if !bl.is_multiple_of(4) {
                            return -1;
                        }
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
                    if b_off != page_slice.len() {
                        return -1;
                    }
                }
            }
        }

        pos = next_pos;
        page_ps = page_pe;
    }
    if page_ps < max_pat {
        return -1;
    }
    0
}

#[derive(Clone, Copy)]
struct RawOutputSink {
    data_ptr: *mut u32,
    data_len: usize,
    counts: Option<(*mut u32, usize)>,
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
    sink: RawOutputSink,
    p_firsts: &[i64],
    p_lasts: &[i64],
    p_dst_pats: &[i64],
    dst_offs: &[i64],
    dst_ends: &[i64],
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
        for s_i in 0..n_spans {
            if dst_offs[s_i] != dst_ends[s_i] {
                return -1;
            }
            if let Some((c_ptr, c_len)) = sink.counts {
                let sp_f = p_firsts[s_i] as usize;
                let sp_l = p_lasts[s_i] as usize;
                let dp = p_dst_pats[s_i] as usize;
                let pat_len = sp_l - sp_f;
                if dp.checked_add(pat_len).is_none_or(|e| e > c_len) {
                    return -1;
                }
                let c_sl = unsafe { std::slice::from_raw_parts_mut(c_ptr.add(dp), pat_len) };
                c_sl.fill(0);
            }
        }
        return 0;
    }

    for s_i in 0..n_spans {
        let start_o = dst_offs[s_i] as usize;
        let end_o = dst_ends[s_i] as usize;
        if start_o > end_o || end_o > sink.data_len {
            return -1;
        }
        cur_dst[s_i] = start_o;
    }

    let min_pat = p_firsts[0] as usize;
    let max_pat = p_lasts.iter().map(|&x| x as usize).max().unwrap_or(rg_pe);
    let Some(col_end) = col_off.checked_add(col_tc) else {
        return -1;
    };
    let max_u_len = col_elems
        .saturating_add(rg_pe - rg_ps)
        .saturating_mul(4);

    let mut pos = col_off;
    let mut page_ps = rg_ps;
    let mut sp_start = 0usize;

    while page_ps < rg_pe && pos < col_end {
        let Some((u_len, c_len, page_nr, c_start, next_pos)) =
            parse_page_header(mm_u8, pos, col_end, max_u_len)
        else {
            return -1;
        };
        let Some(page_pe) = page_ps.checked_add(page_nr) else {
            return -1;
        };
        if page_pe > rg_pe {
            return -1;
        }
        let page_elems = (u_len / 4) - page_nr;

        if page_ps >= max_pat {
            break;
        }
        while sp_start < n_spans && (p_lasts[sp_start] as usize) <= page_ps {
            sp_start += 1;
        }
        if sp_start >= n_spans {
            page_ps = page_pe;
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
                    if let Some((c_ptr, c_len_tot)) = sink.counts {
                        let mut s_i = sp_start;
                        while s_i < n_spans && (p_firsts[s_i] as usize) < page_pe {
                            let sp_f = p_firsts[s_i] as usize;
                            let sp_l = p_lasts[s_i] as usize;
                            let f = sp_f.max(page_ps);
                            let l = sp_l.min(page_pe);
                            if l > f {
                                let dp = (p_dst_pats[s_i] as usize) + (f - sp_f);
                                let pat_len = l - f;
                                if dp.checked_add(pat_len).is_none_or(|e| e > c_len_tot) {
                                    return -1;
                                }
                                let c_sl = unsafe {
                                    std::slice::from_raw_parts_mut(c_ptr.add(dp), pat_len)
                                };
                                c_sl.fill(0);
                            }
                            s_i += 1;
                        }
                    }
                } else {
                    let Some(page_slice) = decompress_or_slice_page(
                        dctx, mm_u8, page_buf, is_zstd, c_start, c_len, u_len,
                    ) else {
                        return -1;
                    };
                    let mut b_off = 0usize;
                    for p_idx in page_ps..page_pe {
                        if b_off.saturating_add(4) > page_slice.len() {
                            return -1;
                        }
                        let bl =
                            u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap())
                                as usize;
                        if !bl.is_multiple_of(4) {
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
                                if let Some((c_ptr, c_len_tot)) = sink.counts {
                                    let c_idx_out = (p_dst_pats[s_i] as usize) + (p_idx - sp_f);
                                    if c_idx_out >= c_len_tot {
                                        return -1;
                                    }
                                    unsafe {
                                        *c_ptr.add(c_idx_out) = k as u32;
                                    }
                                }
                                if k > 0 {
                                    let out_idx = cur_dst[s_i];
                                    let span_end = dst_ends[s_i] as usize;
                                    if out_idx.checked_add(k).is_none_or(|e| e > span_end) {
                                        return -1;
                                    }
                                    let dst_pat = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            sink.data_ptr.add(out_idx),
                                            k,
                                        )
                                    };
                                    decode_pattern_u32(
                                        page_slice,
                                        b_s,
                                        dst_pat,
                                        is_delta,
                                        is_shuffled,
                                    );
                                    cur_dst[s_i] = out_idx + k;
                                }
                            }
                            s_i += 1;
                        }
                    }
                    if b_off != page_slice.len() {
                        return -1;
                    }
                }
            }
        }

        pos = next_pos;
        page_ps = page_pe;
    }

    if page_ps < max_pat {
        return -1;
    }
    for s_i in 0..n_spans {
        if cur_dst[s_i] != dst_ends[s_i] as usize {
            return -1;
        }
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
    let Some(col_end) = col_off.checked_add(col_tc) else {
        return Err(-1);
    };
    let max_u_len = col_elems
        .saturating_add(rg_pe - rg_ps)
        .saturating_mul(4);
    let mut pos = col_off;
    let mut page_ps = rg_ps;

    while page_ps < rg_pe && pos < col_end {
        let Some((u_len, c_len, page_nr, c_start, next_pos)) =
            parse_page_header(mm_u8, pos, col_end, max_u_len)
        else {
            return Err(-1);
        };
        let Some(page_pe) = page_ps.checked_add(page_nr) else {
            return Err(-1);
        };
        if page_pe > rg_pe {
            return Err(-1);
        }

        if frame < page_pe {
            let page_elems = (u_len / 4) - page_nr;
            if page_elems == 0 {
                return Ok(Vec::new());
            }
            let Some(page_slice) =
                decompress_or_slice_page(dctx, mm_u8, page_buf, is_zstd, c_start, c_len, u_len)
            else {
                return Err(-1);
            };
            let mut b_off = 0usize;
            for _ in page_ps..frame {
                if b_off.saturating_add(4) > page_slice.len() {
                    return Err(-1);
                }
                let bl =
                    u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
                if !bl.is_multiple_of(4) {
                    return Err(-1);
                }
                b_off = b_off.saturating_add(4).saturating_add(bl);
            }
            if b_off.saturating_add(4) > page_slice.len() {
                return Err(-1);
            }
            let bl = u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
            if !bl.is_multiple_of(4)
                || b_off.saturating_add(4).saturating_add(bl) > page_slice.len()
            {
                return Err(-1);
            }
            let k = bl / 4;
            if k == 0 {
                return Ok(Vec::new());
            }
            let mut out = vec![0u32; k];
            decode_pattern_u32(page_slice, b_off + 4, &mut out[..k], is_delta, is_shuffled);
            return Ok(out);
        }

        pos = next_pos;
        page_ps = page_pe;
    }
    Err(-1)
}

#[pyfunction]
pub fn parse_parquet_footer<'py>(
    py: Python<'py>,
    footer_bytes: &[u8],
) -> PyResult<FooterParseResult<'py>> {
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
        let Ok(v_usize) = usize::try_from(v) else {
            return Ok(None);
        };
        pos = p2;
        v_usize
    } else {
        pos += 2;
        (hdr_b >> 4) as usize
    };

    if n_rgs > n_len / 32 {
        return Ok(None);
    }
    let Some(info_len) = n_rgs.checked_mul(9) else {
        return Ok(None);
    };
    let mut col_info_vec = vec![0i64; info_len];
    let col_name_lens = [10usize, 11usize, 11usize];
    let mut sum_ones = 0i64;
    let mut sum_multi = 0i64;
    for rg in 0..n_rgs {
        if pos.checked_add(1).is_none_or(|p| p >= n_len)
            || footer_bytes[pos] != 0x19
            || footer_bytes[pos + 1] != 0x3C
        {
            return Ok(None);
        }
        pos += 2;
        for c in 0..3 {
            if pos.checked_add(4).is_none_or(|p| p >= n_len)
                || footer_bytes[pos] != 0x26
                || footer_bytes[pos + 3] != 0x15
            {
                return Ok(None);
            }
            let (_, p_ct) = match try_read_zigzag(footer_bytes, pos + 4) {
                Some(v) => v,
                None => return Ok(None),
            };
            let Some(pos_cd) = p_ct.checked_add(7 + col_name_lens[c]) else {
                return Ok(None);
            };
            pos = pos_cd;
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
            let Some(p_stats) = p_po.checked_add(COL_ENCODING_STATS_PREFIX.len()) else {
                return Ok(None);
            };
            if p_stats > n_len || &footer_bytes[p_po..p_stats] != COL_ENCODING_STATS_PREFIX {
                return Ok(None);
            }
            let (_, p_np) = match try_read_zigzag(footer_bytes, p_stats) {
                Some(v) => v,
                None => return Ok(None),
            };
            if p_np >= n_len || footer_bytes[p_np] != 0x00 {
                return Ok(None);
            }
            pos = p_np + 1;
            if pos.checked_add(1).is_none_or(|p| p >= n_len)
                || footer_bytes[pos] != 0x3C
                || footer_bytes[pos + 1] != 0x16
            {
                return Ok(None);
            }
            let (data_bytes, p_db) = match try_read_zigzag(footer_bytes, pos + 2) {
                Some(v) => v,
                None => return Ok(None),
            };
            let Some(pos_after_col) = p_db.checked_add(7) else {
                return Ok(None);
            };
            pos = pos_after_col;
            if poff < 4 || tc < 0 || data_bytes < 0 || data_bytes % 4 != 0 {
                return Ok(None);
            }
            let n_elems = data_bytes >> 2;
            col_info_vec[rg * 9 + c * 3] = poff;
            col_info_vec[rg * 9 + c * 3 + 1] = if codec == 0 { -tc } else { tc };
            col_info_vec[rg * 9 + c * 3 + 2] = n_elems;
            if c == 0 {
                sum_ones = match sum_ones.checked_add(n_elems) {
                    Some(v) => v,
                    None => return Ok(None),
                };
            } else if c == 1 {
                sum_multi = match sum_multi.checked_add(n_elems) {
                    Some(v) => v,
                    None => return Ok(None),
                };
            }
        }
        if col_info_vec[rg * 9 + 5] != col_info_vec[rg * 9 + 8] {
            return Ok(None);
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

    if pos.checked_add(2).is_none_or(|p| p > n_len) || footer_bytes[pos] != 0x19 {
        return Ok(None);
    }
    let kv_hdr = footer_bytes[pos + 1];
    let n_kv: usize = if kv_hdr == 0xFC {
        let (v, p_kv) = match try_read_uvarint(footer_bytes, pos + 2) {
            Some(v) => v,
            None => return Ok(None),
        };
        let Ok(v_usize) = usize::try_from(v) else {
            return Ok(None);
        };
        pos = p_kv;
        v_usize
    } else {
        pos += 2;
        (kv_hdr >> 4) as usize
    };
    if n_kv > n_len / 3 {
        return Ok(None);
    }

    let mut version = "2";
    let mut position_encoding = "absolute";
    let mut num_pix: Option<i64> = None;
    let mut num_data: Option<i64> = None;
    let mut rg_pats: i64 = 1;
    let mut shuffle = false;
    let mut payload_codec = "none";
    let mut compression: Option<&str> = None;

    for _ in 0..n_kv {
        if pos >= n_len || footer_bytes[pos] != 0x18 {
            return Ok(None);
        }
        let (k_len_u, pk) = match try_read_uvarint(footer_bytes, pos + 1) {
            Some(v) => v,
            None => return Ok(None),
        };
        let Ok(k_len) = usize::try_from(k_len_u) else {
            return Ok(None);
        };
        pos = pk;
        let Some(k_end) = pos.checked_add(k_len) else {
            return Ok(None);
        };
        if k_end > n_len {
            return Ok(None);
        }
        let k_slice = &footer_bytes[pos..k_end];
        pos = k_end;

        if pos >= n_len || footer_bytes[pos] != 0x18 {
            return Ok(None);
        }
        let (v_len_u, pv) = match try_read_uvarint(footer_bytes, pos + 1) {
            Some(v) => v,
            None => return Ok(None),
        };
        let Ok(v_len) = usize::try_from(v_len_u) else {
            return Ok(None);
        };
        pos = pv;
        let Some(v_end) = pos.checked_add(v_len) else {
            return Ok(None);
        };
        if v_end > n_len {
            return Ok(None);
        }
        let v_slice = &footer_bytes[pos..v_end];
        pos = v_end;

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
            b"compression" => compression = Some(v_str),
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
    if npix < 0 || ndata < 0 || rg_pats <= 0 {
        return Ok(None);
    }
    let (Ok(ndata_u), Ok(rg_pats_u)) = (usize::try_from(ndata), usize::try_from(rg_pats)) else {
        return Ok(None);
    };
    let expected_rgs = if ndata_u == 0 {
        0
    } else {
        ndata_u.div_ceil(rg_pats_u)
    };
    if n_rgs != expected_rgs {
        return Ok(None);
    }

    let inferred_compression = if n_rgs > 0 && col_info_vec[1] > 0 {
        "zstd"
    } else {
        "uncompressed"
    };
    let meta_dict = PyDict::new(py);
    meta_dict.set_item("version", version)?;
    meta_dict.set_item("num_pix", npix)?;
    meta_dict.set_item("num_data", ndata)?;
    meta_dict.set_item("rg_pats", rg_pats)?;
    meta_dict.set_item("position_encoding", position_encoding)?;
    meta_dict.set_item("shuffle", shuffle)?;
    meta_dict.set_item("payload_codec", payload_codec)?;
    meta_dict.set_item("compression", compression.unwrap_or(inferred_compression))?;
    meta_dict.set_item("num_ones", sum_ones)?;
    meta_dict.set_item("num_multi", sum_multi)?;

    let arr1 = PyArray1::from_vec(py, col_info_vec);
    let arr3 = arr1.reshape([n_rgs, 3, 3])?;
    Ok(Some((meta_dict, arr3)))
}

#[pyfunction]
#[pyo3(signature = (
    file,
    b_po,
    b_pm,
    b_cm,
    b_ones_idx,
    b_multi_idx,
    num_data,
    rg_pats,
    page_pats,
    level,
    is_delta,
    is_shuffled,
    check_sorted,
    n_workers,
    is_compressed = None
))]
#[allow(clippy::too_many_arguments)]
pub fn write_batch_zstd<'py>(
    py: Python<'py>,
    file: &Bound<'py, PyAny>,
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
    is_compressed: Option<bool>,
) -> PyResult<Bound<'py, PyArray3<i64>>> {
    if rg_pats == 0 || page_pats == 0 {
        return Err(PyValueError::new_err(
            "rg_pats and page_pats must be positive",
        ));
    }
    let use_zstd = is_compressed.unwrap_or(level > 0);
    let target = RawFileTarget::from_py(py, file)?;
    if num_data == 0 {
        let arr1 = PyArray1::from_vec(py, Vec::<i64>::new());
        return arr1
            .reshape([0, 3, 3])
            .map_err(|e| PyValueError::new_err(e.to_string()));
    }

    let po_src = b_po.as_slice()?;
    let pm_src = b_pm.as_slice()?;
    let cm_src = b_cm.as_slice()?;
    let ones_idx = b_ones_idx.as_slice()?;
    let multi_idx = b_multi_idx.as_slice()?;

    if ones_idx.len() <= num_data || multi_idx.len() <= num_data {
        return Err(PyValueError::new_err(
            "offset arrays must have length at least num_data + 1",
        ));
    }
    if ones_idx[0] != 0
        || multi_idx[0] != 0
        || !ones_idx[..=num_data].windows(2).all(|w| w[1] >= w[0])
        || !multi_idx[..=num_data].windows(2).all(|w| w[1] >= w[0])
    {
        return Err(PyValueError::new_err(
            "offset arrays must start at 0 and be non-decreasing",
        ));
    }
    let need_po = usize::try_from(ones_idx[num_data])
        .map_err(|_| PyValueError::new_err("ones_idx overflows usize"))?;
    let need_pm = usize::try_from(multi_idx[num_data])
        .map_err(|_| PyValueError::new_err("multi_idx overflows usize"))?;
    if need_po > po_src.len() || need_pm > pm_src.len() || need_pm > cm_src.len() {
        return Err(PyValueError::new_err(
            "event arrays are shorter than offset totals",
        ));
    }

    let b_rgs = num_data.div_ceil(rg_pats);
    let n_workers = n_workers.max(1);
    let meta_len = b_rgs
        .checked_mul(9)
        .ok_or_else(|| PyValueError::new_err("row-group metadata size overflow"))?;
    let mut meta_vec = vec![0i64; meta_len];

    let rc = py.detach(|| -> i32 {
        if n_workers <= 1 || b_rgs <= 1 {
            let mut out_vec = Vec::new();
            for rg in 0..b_rgs {
                let meta_rg = &mut meta_vec[rg * 9..(rg + 1) * 9];
                let written = match encode_rg_into_vec(
                    rg,
                    num_data,
                    rg_pats,
                    page_pats,
                    use_zstd,
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
                ) {
                    Ok(n) => n,
                    Err(e) => return e,
                };
                if !target.write_all(&out_vec[..written]) {
                    return -4;
                }
            }
            0
        } else {
            let pool = get_pool(n_workers.min(b_rgs));
            let actual_workers = n_workers.min(pool.current_num_threads()).min(b_rgs);
            if actual_workers <= 1 {
                let mut out_vec = Vec::new();
                for rg in 0..b_rgs {
                    let meta_rg = &mut meta_vec[rg * 9..(rg + 1) * 9];
                    let written = match encode_rg_into_vec(
                        rg,
                        num_data,
                        rg_pats,
                        page_pats,
                        use_zstd,
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
                    ) {
                        Ok(n) => n,
                        Err(e) => return e,
                    };
                    if !target.write_all(&out_vec[..written]) {
                        return -4;
                    }
                }
                return 0;
            }

            struct PipeState {
                written_upto: usize,
                ready: Vec<Option<(usize, [i64; 9])>>,
            }

            let ring_size = (actual_workers * 3).max(4).min(b_rgs);
            let ring_bufs: Vec<Mutex<Vec<u8>>> =
                (0..ring_size).map(|_| Mutex::new(Vec::new())).collect();
            let pipe_mtx = Mutex::new(PipeState {
                written_upto: 0,
                ready: vec![None; b_rgs],
            });
            let cv_writer = Condvar::new();
            let cv_worker = Condvar::new();
            let next_rg = AtomicUsize::new(0);
            let abort_err = AtomicI32::new(0);

            pool.in_place_scope(|s| {
                for _ in 0..actual_workers {
                    s.spawn(|_| loop {
                        if abort_err.load(Ordering::Acquire) != 0 {
                            return;
                        }
                        let rg = next_rg.fetch_add(1, Ordering::Relaxed);
                        if rg >= b_rgs {
                            break;
                        }
                        if rg >= ring_size {
                            let mut st = pipe_mtx.lock().unwrap_or_else(|e| e.into_inner());
                            while st.written_upto + ring_size <= rg
                                && abort_err.load(Ordering::Acquire) == 0
                            {
                                st = cv_worker.wait(st).unwrap_or_else(|e| e.into_inner());
                            }
                            if abort_err.load(Ordering::Acquire) != 0 {
                                return;
                            }
                        }

                        let mut meta_rg = [0i64; 9];
                        let encode_res = {
                            let mut out_vec = ring_bufs[rg % ring_size]
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                            encode_rg_into_vec(
                                rg,
                                num_data,
                                rg_pats,
                                page_pats,
                                use_zstd,
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
                                &mut meta_rg,
                            )
                        };

                        match encode_res {
                            Ok(written) => {
                                let mut st = pipe_mtx.lock().unwrap_or_else(|e| e.into_inner());
                                st.ready[rg] = Some((written, meta_rg));
                                drop(st);
                                cv_writer.notify_one();
                            }
                            Err(e) => {
                                let _ = abort_err.compare_exchange(
                                    0,
                                    e,
                                    Ordering::AcqRel,
                                    Ordering::Acquire,
                                );
                                let _st = pipe_mtx.lock().unwrap_or_else(|e| e.into_inner());
                                cv_writer.notify_all();
                                cv_worker.notify_all();
                                return;
                            }
                        }
                    });
                }

                for rg in 0..b_rgs {
                    let (written, meta_rg) = {
                        let mut st = pipe_mtx.lock().unwrap_or_else(|e| e.into_inner());
                        while st.ready[rg].is_none() && abort_err.load(Ordering::Acquire) == 0 {
                            st = cv_writer.wait(st).unwrap_or_else(|e| e.into_inner());
                        }
                        if abort_err.load(Ordering::Acquire) != 0 {
                            return;
                        }
                        st.ready[rg].unwrap()
                    };
                    meta_vec[rg * 9..(rg + 1) * 9].copy_from_slice(&meta_rg);
                    let write_ok = {
                        let out_vec = ring_bufs[rg % ring_size]
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        target.write_all(&out_vec[..written])
                    };
                    if !write_ok {
                        let _ = abort_err.compare_exchange(
                            0,
                            -4,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        );
                        let _st = pipe_mtx.lock().unwrap_or_else(|e| e.into_inner());
                        cv_worker.notify_all();
                        return;
                    }
                    let mut st = pipe_mtx.lock().unwrap_or_else(|e| e.into_inner());
                    st.written_upto = rg + 1;
                    drop(st);
                    cv_worker.notify_all();
                }
            });

            abort_err.load(Ordering::Acquire)
        }
    });

    if rc == -2 {
        return Err(PyValueError::new_err(
            "position values must be sorted within every pattern",
        ));
    }
    if rc == -5 {
        return Err(PyValueError::new_err(
            "Parquet page or row-group size exceeds 32-bit Thrift limit",
        ));
    }
    if rc != 0 {
        return Err(PyRuntimeError::new_err(
            "Zstd compression or write failed in _parquet_rs",
        ));
    }
    let arr1 = PyArray1::from_vec(py, meta_vec);
    arr1.reshape([b_rgs, 3, 3])
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (file, num_data, rg_pats, batch_pats, meta_batches, kv_items, page_pats = None))]
#[allow(clippy::too_many_arguments)]
pub fn write_parquet_footer<'py>(
    py: Python<'py>,
    file: &Bound<'py, PyAny>,
    num_data: i64,
    rg_pats: usize,
    batch_pats: Vec<usize>,
    meta_batches: Vec<PyReadonlyArray3<'py, i64>>,
    kv_items: Vec<(Vec<u8>, Vec<u8>)>,
    page_pats: Option<usize>,
) -> PyResult<()> {
    if rg_pats == 0 {
        return Err(PyValueError::new_err("rg_pats must be positive"));
    }
    let page_pats_val = page_pats.unwrap_or(rg_pats);
    if page_pats_val == 0 {
        return Err(PyValueError::new_err("page_pats must be positive"));
    }
    if batch_pats.len() != meta_batches.len() {
        return Err(PyValueError::new_err(
            "batch_pats and meta_batches must have matching lengths",
        ));
    }
    let target = RawFileTarget::from_py(py, file)?;

    let mut slices = Vec::with_capacity(meta_batches.len());
    let mut total_rgs = 0usize;
    for (&bp, mb) in batch_pats.iter().zip(meta_batches.iter()) {
        let sl = as_c_slice_3d(mb, 3, 3, "meta_batch")?;
        if bp > 0 {
            let b_rgs = bp.div_ceil(rg_pats);
            if sl.len() != b_rgs * 9 {
                return Err(PyValueError::new_err(
                    "meta_batch row-group count does not match batch_pats",
                ));
            }
            total_rgs = total_rgs
                .checked_add(b_rgs)
                .ok_or_else(|| PyValueError::new_err("total_rgs overflow"))?;
        }
        slices.push(sl);
    }

    let mut codec = 6i64;
    for (k, v) in &kv_items {
        if k == b"compression" && v == b"none" {
            codec = 0;
        }
    }

    let res = py.detach(|| -> Result<(), i32> {
        let cap = SCHEMA_PREFIX
            .len()
            .saturating_add(total_rgs.saturating_mul(220))
            .saturating_add(512);
        let mut footer = Vec::with_capacity(cap);
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
            let b_rgs = b_pats.div_ceil(rg_pats);
            for rg in 0..b_rgs {
                let ps = rg * rg_pats;
                let pe = ((rg + 1) * rg_pats).min(b_pats);
                let nr_usize = pe - ps;
                let nr = nr_usize as i64;
                let num_pages = nr_usize.div_ceil(page_pats_val).max(1) as i64;
                let rg_start_pos = file_pos;
                let mut total_uncomp_rg = 0i64;
                let mut total_comp_rg = 0i64;

                footer.extend_from_slice(b"\x19\x3c");
                for (c, &col_name) in COL_NAMES.iter().enumerate() {
                    let tu = meta_slice[rg * 9 + c * 3];
                    let tc = meta_slice[rg * 9 + c * 3 + 1];
                    let n_el = meta_slice[rg * 9 + c * 3 + 2];
                    let data_bytes = n_el.checked_mul(4).ok_or(-5i32)?;
                    push_col_chunk_meta(
                        &mut footer,
                        6,
                        col_name,
                        codec,
                        nr,
                        tu,
                        tc,
                        file_pos,
                        num_pages,
                        data_bytes,
                    );
                    file_pos = file_pos.checked_add(tc).ok_or(-5i32)?;
                    total_uncomp_rg = total_uncomp_rg.checked_add(tu).ok_or(-5i32)?;
                    total_comp_rg = total_comp_rg.checked_add(tc).ok_or(-5i32)?;
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
        let footer_len = u32::try_from(footer.len()).map_err(|_| -5i32)?;
        footer.extend_from_slice(&footer_len.to_le_bytes());
        footer.extend_from_slice(PARQUET_MAGIC);

        if !target.write_all(&footer) {
            return Err(-4);
        }
        Ok(())
    });

    match res {
        Ok(()) => Ok(()),
        Err(-5) => Err(PyValueError::new_err(
            "Parquet footer exceeds 4 GiB format limit",
        )),
        Err(_) => Err(PyRuntimeError::new_err(
            "Failed to write Parquet footer in _parquet_rs",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
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
        let Ok(c_u32) = u32::try_from(col_elems) else {
            return -1;
        };
        out_counts[0] = c_u32;
        return 0;
    }

    let is_zstd = col_tc_raw >= 0;
    let col_tc = col_tc_raw.unsigned_abs() as usize;
    let Some(col_end) = col_off.checked_add(col_tc) else {
        return -1;
    };
    let max_u_len = col_elems.saturating_add(nr).saturating_mul(4);
    let mut pos = col_off;
    let mut page_ps = rg_ps;
    let mut sum_elems = 0usize;

    while page_ps < rg_pe && pos < col_end {
        let Some((u_len, c_len, page_nr, c_start, next_pos)) =
            parse_page_header(mm_u8, pos, col_end, max_u_len)
        else {
            return -1;
        };
        let Some(page_pe) = page_ps.checked_add(page_nr) else {
            return -1;
        };
        if page_pe > rg_pe {
            return -1;
        }
        let page_elems = (u_len / 4) - page_nr;

        if page_elems == 0 {
            out_counts[page_ps - rg_ps..page_pe - rg_ps].fill(0);
        } else if page_nr == 1 {
            let Ok(pe_u32) = u32::try_from(page_elems) else {
                return -1;
            };
            out_counts[page_ps - rg_ps] = pe_u32;
            sum_elems = sum_elems.saturating_add(page_elems);
        } else {
            let Some(page_slice) =
                decompress_or_slice_page(dctx, mm_u8, page_buf, is_zstd, c_start, c_len, u_len)
            else {
                return -1;
            };
            let mut b_off = 0usize;
            for p_idx in page_ps..page_pe {
                if b_off.saturating_add(4) > page_slice.len() {
                    return -1;
                }
                let bl =
                    u32::from_le_bytes(page_slice[b_off..b_off + 4].try_into().unwrap()) as usize;
                if !bl.is_multiple_of(4) {
                    return -1;
                }
                let k = bl / 4;
                out_counts[p_idx - rg_ps] = k as u32;
                sum_elems = sum_elems.saturating_add(k);
                b_off = b_off.saturating_add(4).saturating_add(bl);
                if b_off > page_slice.len() {
                    return -1;
                }
            }
            if b_off != page_slice.len() {
                return -1;
            }
        }

        pos = next_pos;
        page_ps = page_pe;
    }

    if page_ps != rg_pe || sum_elems != col_elems {
        return -1;
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
) -> PyResult<PairArraysU32<'py>> {
    if rg_pats == 0 {
        return Err(PyValueError::new_err("rg_pats must be positive"));
    }
    if num_data == 0 {
        return Ok((
            PyArray1::<u32>::zeros(py, [0], false),
            PyArray1::<u32>::zeros(py, [0], false),
        ));
    }
    let mm_cow = view_pybuffer_u8_detachable(py, &b_mm)?;
    let mm_u8: &[u8] = &mm_cow;
    let col_info = as_c_slice_3d(&b_col, 3, 3, "b_col")?;
    let n_rgs = num_data.div_ceil(rg_pats);
    if col_info.len() / 9 < n_rgs {
        return Err(PyValueError::new_err(
            "b_col row-group count is smaller than expected for num_data",
        ));
    }
    let n_workers = n_workers.max(1);

    let out_ones_arr = PyArray1::<u32>::zeros(py, [num_data], false);
    let out_multi_arr = PyArray1::<u32>::zeros(py, [num_data], false);
    let ones_addr = unsafe { out_ones_arr.as_slice_mut()? }.as_mut_ptr() as usize;
    let multi_addr = unsafe { out_multi_arr.as_slice_mut()? }.as_mut_ptr() as usize;

    let read_rg = |rg: usize| -> Result<(), i32> {
        let Some((off0, tc0, el0)) = validate_rg_col_entry(col_info, rg, 0, mm_u8.len()) else {
            return Err(-1);
        };
        let Some((off1, tc1, el1)) = validate_rg_col_entry(col_info, rg, 1, mm_u8.len()) else {
            return Err(-1);
        };
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

            let r0 = read_col_counts_rg(dctx, mm_u8, off0, tc0, el0, scratch, ps, pe, ones_rg);
            if r0 != 0 {
                return r0;
            }
            read_col_counts_rg(dctx, mm_u8, off1, tc1, el1, scratch, ps, pe, multi_rg)
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
    if rg_pats == 0 {
        return Err(PyValueError::new_err("rg_pats must be positive"));
    }
    let rp = i64::try_from(rg_pats).map_err(|_| PyValueError::new_err("rg_pats overflows i64"))?;
    let nd_i64 =
        i64::try_from(num_data).map_err(|_| PyValueError::new_err("num_data overflows i64"))?;

    let mm_cow = view_pybuffer_u8_detachable(py, &b_mm)?;
    let mm_u8: &[u8] = &mm_cow;
    let col_info = as_c_slice_3d(&b_col, 3, 3, "b_col")?;
    let ranges_slice = as_c_slice_2d(&b_ranges, 2, "b_ranges")?;
    let ones_idx_opt = b_ones_idx.as_ref().map(|a| a.as_slice()).transpose()?;
    let multi_idx_opt = b_multi_idx.as_ref().map(|a| a.as_slice()).transpose()?;

    if ones_idx_opt.is_some() != multi_idx_opt.is_some() {
        return Err(PyValueError::new_err(
            "Pattern counts do not match Parquet v2 payload size",
        ));
    }
    if let (Some(ones_idx), Some(multi_idx)) = (ones_idx_opt, multi_idx_opt) {
        if ones_idx.len() != num_data + 1
            || multi_idx.len() != num_data + 1
            || ones_idx[0] != 0
            || multi_idx[0] != 0
            || !ones_idx.windows(2).all(|w| w[1] >= w[0])
            || !multi_idx.windows(2).all(|w| w[1] >= w[0])
        {
            return Err(PyValueError::new_err(
                "Pattern counts do not match Parquet v2 payload size",
            ));
        }
    }

    let n_in_ranges = ranges_slice.len() / 2;
    for i in 0..n_in_ranges {
        let s = ranges_slice[i * 2];
        let e = ranges_slice[i * 2 + 1];
        if s < 0 || e > nd_i64 || s > e {
            return Err(PyIndexError::new_err("Pattern range is out of bounds"));
        }
    }
    if num_data == 0 || n_in_ranges == 0 {
        return Ok((
            PyArray1::<u32>::zeros(py, [0], false),
            PyArray1::<u32>::zeros(py, [0], false),
            PyArray1::<u32>::zeros(py, [0], false),
            PyArray1::<u32>::zeros(py, [0], false),
            PyArray1::<u32>::zeros(py, [0], false),
        ));
    }

    let req_rgs = num_data.div_ceil(rg_pats);
    if col_info.len() / 9 < req_rgs {
        return Err(PyValueError::new_err(
            "b_col row-group count is smaller than expected for num_data",
        ));
    }

    let prep_res = py.detach(|| -> Result<_, i32> {
        let is_full =
            n_in_ranges == 1 && ranges_slice[0] == 0 && (ranges_slice[1] as usize) == num_data;

        let mut total_spans = 0usize;
        for i in 0..n_in_ranges {
            let s = ranges_slice[i * 2];
            let e = ranges_slice[i * 2 + 1];
            if e > s {
                let rg_s = (s / rp) as usize;
                let rg_e = ((e - 1) / rp) as usize;
                total_spans = total_spans.checked_add(rg_e - rg_s + 1).ok_or(-1)?;
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
                    acc_pats = acc_pats.checked_add(l - f).ok_or(-1)?;
                    dst_pats.push(acc_pats);
                }
            }
        }
        let total_sel_pats = usize::try_from(acc_pats).map_err(|_| -1)?;

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

        for &rg in &unique_rgs {
            if validate_rg_col_entry(col_info, rg, 0, mm_u8.len()).is_none()
                || validate_rg_col_entry(col_info, rg, 1, mm_u8.len()).is_none()
                || validate_rg_col_entry(col_info, rg, 2, mm_u8.len()).is_none()
                || col_info[rg * 9 + 5] != col_info[rg * 9 + 8]
            {
                return Err(-1);
            }
        }

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
                ao = ao.checked_add(col_info[rg * 9 + 2]).ok_or(-1)?;
                am = am.checked_add(col_info[rg * 9 + 5]).ok_or(-1)?;
                dst_ones.push(ao);
                dst_multi.push(am);
            }
        } else if let (Some(ones_idx), Some(multi_idx)) = (ones_idx_opt, multi_idx_opt) {
            let mut ao = 0i64;
            let mut am = 0i64;
            for sid in 0..total_spans {
                let f = firsts[sid] as usize;
                let l = lasts[sid] as usize;
                let do_s = i64::try_from(ones_idx[l] - ones_idx[f]).map_err(|_| -1)?;
                let dm_s = i64::try_from(multi_idx[l] - multi_idx[f]).map_err(|_| -1)?;
                ao = ao.checked_add(do_s).ok_or(-1)?;
                am = am.checked_add(dm_s).ok_or(-1)?;
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

                for &sid in &group_span_ids[gs0..gs1] {
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

                for (i, &sid) in group_span_ids[gs0..gs1].iter().enumerate() {
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
                ao = ao.checked_add(ones_lens[sid]).ok_or(-1)?;
                am = am.checked_add(multi_lens[sid]).ok_or(-1)?;
                dst_ones.push(ao);
                dst_multi.push(am);
            }
        }

        let total_ones = usize::try_from(*dst_ones.last().unwrap()).map_err(|_| -1)?;
        let total_multi = usize::try_from(*dst_multi.last().unwrap()).map_err(|_| -1)?;

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
            total_ones,
            total_multi,
            n_workers,
        )))
    });

    let prep = match prep_res {
        Ok(Some(p)) => p,
        Ok(None) => {
            return Ok((
                PyArray1::<u32>::zeros(py, [0], false),
                PyArray1::<u32>::zeros(py, [0], false),
                PyArray1::<u32>::zeros(py, [0], false),
                PyArray1::<u32>::zeros(py, [0], false),
                PyArray1::<u32>::zeros(py, [0], false),
            ));
        }
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
        total_ones,
        total_multi,
        n_workers,
    ) = prep;

    let n_groups = unique_rgs.len();

    let out_ones_arr = PyArray1::<u32>::zeros(py, [total_sel_pats], false);
    let out_multi_arr = PyArray1::<u32>::zeros(py, [total_sel_pats], false);
    let out_po_arr = PyArray1::<u32>::zeros(py, [total_ones], false);
    let out_pm_arr = PyArray1::<u32>::zeros(py, [total_multi], false);
    let out_cm_arr = PyArray1::<u32>::zeros(py, [total_multi], false);

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
            let mut d_ones_end = Vec::with_capacity(n_sp);
            let mut d_multi = Vec::with_capacity(n_sp);
            let mut d_multi_end = Vec::with_capacity(n_sp);
            let mut cur_dst = vec![0usize; n_sp];
            let mut need_m_payload = false;

            for &sid in &group_span_ids[gs0..gs1] {
                p_firsts.push(firsts[sid]);
                p_lasts.push(lasts[sid]);
                d_pats.push(dst_pats[sid]);
                d_ones.push(dst_ones[sid]);
                d_ones_end.push(dst_ones[sid + 1]);
                d_multi.push(dst_multi[sid]);
                d_multi_end.push(dst_multi[sid + 1]);
                if dst_multi[sid + 1] > dst_multi[sid] {
                    need_m_payload = true;
                }
            }

            let po_sink = RawOutputSink {
                data_ptr: out_po_addr as *mut u32,
                data_len: total_ones,
                counts: Some((out_ones_addr as *mut u32, total_sel_pats)),
            };
            let pm_sink = RawOutputSink {
                data_ptr: out_pm_addr as *mut u32,
                data_len: total_multi,
                counts: Some((out_multi_addr as *mut u32, total_sel_pats)),
            };
            let cm_sink = RawOutputSink {
                data_ptr: out_cm_addr as *mut u32,
                data_len: total_multi,
                counts: None,
            };

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
                    po_sink,
                    &p_firsts,
                    &p_lasts,
                    &d_pats,
                    &d_ones,
                    &d_ones_end,
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
                    pm_sink,
                    &p_firsts,
                    &p_lasts,
                    &d_pats,
                    &d_multi,
                    &d_multi_end,
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
                        cm_sink,
                        &p_firsts,
                        &p_lasts,
                        &d_pats,
                        &d_multi,
                        &d_multi_end,
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
) -> PyResult<TripleArraysU32<'py>> {
    let mm_cow = view_pybuffer_u8_detachable(py, &b_mm)?;
    let mm_u8: &[u8] = &mm_cow;
    let col_info = as_c_slice_3d(&b_col, 3, 3, "b_col")?;
    if rg >= col_info.len() / 9 || ps > frame || frame >= pe {
        return Err(PyIndexError::new_err("Frame index is out of bounds"));
    }
    let Some((off0, tc0, el0)) = validate_rg_col_entry(col_info, rg, 0, mm_u8.len()) else {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    };
    let Some((off1, tc1, el1)) = validate_rg_col_entry(col_info, rg, 1, mm_u8.len()) else {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    };
    let Some((off2, tc2, el2)) = validate_rg_col_entry(col_info, rg, 2, mm_u8.len()) else {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    };
    if el1 != el2 {
        return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
    }

    let res = py.detach(|| -> Result<TripleVecU32, i32> {
        THREAD_ZSTD.with(|cell| {
            let mut ctx = cell.borrow_mut();
            let zctx = &mut *ctx;
            let dctx = &mut zctx.dctx;
            let scratch = &mut zctx.scratch;

            let po = decode_single_col_frame(
                dctx,
                mm_u8,
                off0,
                tc0,
                el0,
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
                off1,
                tc1,
                el1,
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
                let decoded_cm = decode_single_col_frame(
                    dctx,
                    mm_u8,
                    off2,
                    tc2,
                    el2,
                    scratch,
                    ps,
                    pe,
                    frame,
                    false,
                    is_shuffled,
                )?;
                if decoded_cm.len() != pm.len() {
                    return Err(-1);
                }
                decoded_cm
            };
            Ok((po, pm, cm))
        })
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
    if rg_pats == 0 || c_idx >= 3 {
        return Err(PyValueError::new_err("Invalid rg_pats or column index"));
    }
    if num_data == 0 {
        return Ok(PyArray1::<u32>::zeros(py, [0], false));
    }
    let mm_cow = view_pybuffer_u8_detachable(py, &b_mm)?;
    let mm_u8: &[u8] = &mm_cow;
    let col_info = as_c_slice_3d(&b_col, 3, 3, "b_col")?;
    let n_rgs = num_data.div_ceil(rg_pats);
    if col_info.len() / 9 < n_rgs {
        return Err(PyValueError::new_err(
            "b_col row-group count is smaller than expected for num_data",
        ));
    }
    let n_workers = n_workers.max(1);

    let mut rg_offs = Vec::with_capacity(n_rgs + 1);
    rg_offs.push(0i64);
    let mut acc = 0i64;
    for rg in 0..n_rgs {
        if validate_rg_col_entry(col_info, rg, c_idx, mm_u8.len()).is_none() {
            return Err(PyValueError::new_err("Corrupted Zstd page in Parquet file"));
        }
        acc = acc
            .checked_add(col_info[rg * 9 + c_idx * 3 + 2])
            .ok_or_else(|| PyValueError::new_err("Column element count overflow"))?;
        rg_offs.push(acc);
    }
    let total_elems = usize::try_from(acc)
        .map_err(|_| PyValueError::new_err("Column element count overflows usize"))?;
    if total_elems == 0 {
        return Ok(PyArray1::<u32>::zeros(py, [0], false));
    }

    let out_arr = PyArray1::<u32>::zeros(py, [total_elems], false);
    let out_addr = unsafe { out_arr.as_slice_mut()? }.as_mut_ptr() as usize;

    let decode_one_rg = |rg: usize| -> Result<(), i32> {
        let ps = rg * rg_pats;
        let pe = ((rg + 1) * rg_pats).min(num_data);
        let rg_base = rg_offs[rg] as usize;
        let rg_end = rg_offs[rg + 1] as usize;
        if rg_end <= rg_base {
            return Ok(());
        }
        let rg_len = rg_end - rg_base;
        let pf_arr = [ps as i64];
        let pl_arr = [pe as i64];
        let dp_arr = [0i64];
        let dst_arr = [0i64];
        let dst_end_arr = [rg_len as i64];
        let mut cur_dst = [0usize];
        let sink = RawOutputSink {
            data_ptr: unsafe { (out_addr as *mut u32).add(rg_base) },
            data_len: rg_len,
            counts: None,
        };

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
                sink,
                &pf_arr,
                &pl_arr,
                &dp_arr,
                &dst_arr,
                &dst_end_arr,
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
) -> PyResult<SplitRangesArrays<'py>> {
    if rg_pats == 0 {
        return Err(PyValueError::new_err("rg_pats must be positive"));
    }
    let rp = i64::try_from(rg_pats).map_err(|_| PyValueError::new_err("rg_pats overflows i64"))?;
    let ranges_slice = as_c_slice_2d(&b_ranges, 2, "b_ranges")?;
    let n_in = ranges_slice.len() / 2;
    let mut total_spans = 0usize;
    for i in 0..n_in {
        let s = ranges_slice[i * 2];
        let e = ranges_slice[i * 2 + 1];
        if s < 0 || s > e {
            return Err(PyIndexError::new_err("Pattern range is out of bounds"));
        }
        if e > s {
            let rg_s = (s / rp) as usize;
            let rg_e = ((e - 1) / rp) as usize;
            total_spans = total_spans
                .checked_add(rg_e - rg_s + 1)
                .ok_or_else(|| PyValueError::new_err("total_spans overflow"))?;
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
    if ps > pe || pe >= offs.len() {
        return Err(PyValueError::new_err(
            "ps and pe are out of bounds for offsets",
        ));
    }
    if !offs[ps..=pe].windows(2).all(|w| w[1] >= w[0]) {
        return Err(PyValueError::new_err(
            "offsets must be monotonically non-decreasing",
        ));
    }
    let s_rg =
        usize::try_from(offs[ps]).map_err(|_| PyValueError::new_err("offset overflows usize"))?;
    let e_rg =
        usize::try_from(offs[pe]).map_err(|_| PyValueError::new_err("offset overflows usize"))?;
    if e_rg > src.len() {
        return Err(PyValueError::new_err("offsets exceed src length"));
    }
    let n_rg = e_rg - s_rg;
    if n_rg == 0 {
        return Ok(PyArray1::from_vec(py, Vec::new()));
    }
    let byte_len = n_rg
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("encoded byte size overflow"))?;
    let mut out_u8: Vec<u8> = vec![0u8; byte_len];
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

    let n_spans = p_firsts.len();
    if p_lasts.len() != n_spans || dst_offs.len() != n_spans {
        return Err(PyValueError::new_err(
            "firsts, lasts, and dst_offs must have identical lengths",
        ));
    }
    if !local_offs.is_empty() {
        if local_offs[0] < 0 || !local_offs.windows(2).all(|w| w[1] >= w[0]) {
            return Err(PyValueError::new_err(
                "local_offs must be non-negative and non-decreasing",
            ));
        }
        let max_off = usize::try_from(*local_offs.last().unwrap())
            .map_err(|_| PyValueError::new_err("local_offs overflows usize"))?;
        if max_off
            .checked_mul(4)
            .is_none_or(|bytes| bytes > rg_u8.len())
        {
            return Err(PyValueError::new_err("local_offs exceed rg_u8 buffer size"));
        }
    }

    for s_i in 0..n_spans {
        if p_firsts[s_i] < 0 || p_lasts[s_i] < p_firsts[s_i] || dst_offs[s_i] < 0 {
            return Err(PyIndexError::new_err("span bounds or dst_offs are invalid"));
        }
        let f = usize::try_from(p_firsts[s_i])
            .map_err(|_| PyIndexError::new_err("first index overflows usize"))?;
        let last_p = usize::try_from(p_lasts[s_i])
            .map_err(|_| PyIndexError::new_err("last index overflows usize"))?;
        let mut out_idx = usize::try_from(dst_offs[s_i])
            .map_err(|_| PyIndexError::new_err("dst_off overflows usize"))?;
        if last_p > f {
            if f < rg_ps {
                return Err(PyIndexError::new_err("span starts before rg_ps"));
            }
            let max_loc = last_p - rg_ps;
            if max_loc >= local_offs.len() {
                return Err(PyIndexError::new_err("span exceeds local_offs length"));
            }
        }
        for p in f..last_p {
            let loc_p = p - rg_ps;
            let s = local_offs[loc_p] as usize;
            let e = local_offs[loc_p + 1] as usize;
            let k = e - s;
            if k > 0 {
                let end_out = out_idx
                    .checked_add(k)
                    .ok_or_else(|| PyValueError::new_err("out_idx overflow"))?;
                if end_out > out_u32.len() {
                    return Err(PyValueError::new_err("decoded span exceeds out_u32 length"));
                }
                decode_pattern_u32(
                    rg_u8,
                    s * 4,
                    &mut out_u32[out_idx..end_out],
                    is_delta,
                    is_shuffled,
                );
                out_idx = end_out;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_and_zigzag_roundtrip_and_overflow() {
        let mut buf = Vec::new();
        for &val in &[0i64, 1, -1, 127, -128, 1_000_000, -1_000_000, i64::MAX, i64::MIN] {
            buf.clear();
            push_zigzag(&mut buf, val);
            let (decoded, end_pos) = try_read_zigzag(&buf, 0).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(end_pos, buf.len());
        }
        // 10-byte varint with overflow bits in 10th byte (> 1) must be rejected
        let overflow_varint = [0xFFu8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x02];
        assert!(try_read_uvarint(&overflow_varint, 0).is_none());
    }

    #[test]
    fn test_parse_page_header_rejects_huge_or_truncated_headers() {
        let mut page = vec![0u8; 64];
        let hdr_end = write_page_header(&mut page, 0, 16, 12, 2).unwrap();
        let total = hdr_end + 12;
        assert_eq!(
            parse_page_header(&page[..total], 0, total, 16),
            Some((16, 12, 2, hdr_end, total))
        );
        // Reject when max_u_len is smaller than claimed u_len (prevents decompression bomb OOM)
        assert!(parse_page_header(&page[..total], 0, total, 8).is_none());
        // Reject when col_end truncates payload
        assert!(parse_page_header(&page[..total], 0, total - 1, 16).is_none());
    }

    #[test]
    fn test_read_col_counts_rg_rejects_truncated_pages() {
        let mut dctx = zstd_safe::DCtx::create();
        let mut scratch = Vec::new();
        let mut out_counts = [0u32; 4];
        // Build a single page covering only 2 of 4 expected patterns
        let mut buf = vec![0u8; 64];
        let hdr_end = write_page_header(&mut buf, 0, 16, 16, 2).unwrap();
        buf[hdr_end..hdr_end + 4].copy_from_slice(&4u32.to_le_bytes());
        buf[hdr_end + 4..hdr_end + 8].copy_from_slice(&10u32.to_le_bytes());
        buf[hdr_end + 8..hdr_end + 12].copy_from_slice(&4u32.to_le_bytes());
        buf[hdr_end + 12..hdr_end + 16].copy_from_slice(&20u32.to_le_bytes());
        let col_end = hdr_end + 16;

        let rc = read_col_counts_rg(
            &mut dctx,
            &buf[..col_end],
            0,
            -(col_end as i64),
            2,
            &mut scratch,
            0,
            4,
            &mut out_counts,
        );
        assert_eq!(rc, -1);
    }
}
