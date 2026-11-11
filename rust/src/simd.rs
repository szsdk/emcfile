#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fused single-pass pattern-segment delta encoding + optional sortedness check + byte-shuffle
/// into 4 planes of stride `chunk_len` at `local_off` in `dst_u8`.
/// Returns `0` on success, or `-2` if `check_sorted` is true and an unsorted pair is found.
#[inline]
pub fn delta_and_shuffle_segment_u32(
    src: &[u32],
    init_prev: u32,
    check_sorted: bool,
    chunk_len: usize,
    local_off: usize,
    dst_u8: &mut [u8],
) -> i32 {
    let k = src.len();
    assert!(
        local_off
            .checked_add(k)
            .is_some_and(|end| end <= chunk_len),
        "segment [local_off..local_off + k] exceeds chunk_len"
    );
    assert!(
        chunk_len.checked_mul(4).is_some_and(|b| b <= dst_u8.len()),
        "dst_u8 too small for 4 * chunk_len bytes"
    );
    if k == 0 {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3") {
            unsafe {
                return delta_and_shuffle_segment_u32_sse41(
                    src,
                    init_prev,
                    check_sorted,
                    chunk_len,
                    local_off,
                    dst_u8,
                );
            }
        }
    }
    delta_and_shuffle_segment_u32_scalar(src, init_prev, check_sorted, chunk_len, local_off, dst_u8)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
unsafe fn delta_and_shuffle_segment_u32_sse41(
    src: &[u32],
    init_prev: u32,
    check_sorted: bool,
    chunk_len: usize,
    local_off: usize,
    dst_u8: &mut [u8],
) -> i32 {
    let k = src.len();
    let s_ptr = src.as_ptr();
    let d0 = unsafe { dst_u8.as_mut_ptr().add(local_off) };
    let d1 = unsafe { d0.add(chunk_len) };
    let d2 = unsafe { d0.add(chunk_len * 2) };
    let d3 = unsafe { d0.add(chunk_len * 3) };

    let mut i = 0usize;
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    let mut prev_vec = _mm_cvtsi32_si128(init_prev as i32);

    if !check_sorted {
        while i + 16 <= k {
            unsafe {
                let x0 = _mm_loadu_si128(s_ptr.add(i) as *const __m128i);
                let x1 = _mm_loadu_si128(s_ptr.add(i + 4) as *const __m128i);
                let x2 = _mm_loadu_si128(s_ptr.add(i + 8) as *const __m128i);
                let x3 = _mm_loadu_si128(s_ptr.add(i + 12) as *const __m128i);

                let p0 = _mm_or_si128(_mm_slli_si128(x0, 4), prev_vec);
                let p1 = _mm_or_si128(_mm_slli_si128(x1, 4), _mm_srli_si128(x0, 12));
                let p2 = _mm_or_si128(_mm_slli_si128(x2, 4), _mm_srli_si128(x1, 12));
                let p3 = _mm_or_si128(_mm_slli_si128(x3, 4), _mm_srli_si128(x2, 12));
                prev_vec = _mm_srli_si128(x3, 12);

                let v0 = _mm_shuffle_epi8(_mm_sub_epi32(x0, p0), shuf);
                let v1 = _mm_shuffle_epi8(_mm_sub_epi32(x1, p1), shuf);
                let v2 = _mm_shuffle_epi8(_mm_sub_epi32(x2, p2), shuf);
                let v3 = _mm_shuffle_epi8(_mm_sub_epi32(x3, p3), shuf);

                let t0 = _mm_unpacklo_epi32(v0, v1);
                let t1 = _mm_unpackhi_epi32(v0, v1);
                let t2 = _mm_unpacklo_epi32(v2, v3);
                let t3 = _mm_unpackhi_epi32(v2, v3);

                _mm_storeu_si128(d0.add(i) as *mut __m128i, _mm_unpacklo_epi64(t0, t2));
                _mm_storeu_si128(d1.add(i) as *mut __m128i, _mm_unpackhi_epi64(t0, t2));
                _mm_storeu_si128(d2.add(i) as *mut __m128i, _mm_unpacklo_epi64(t1, t3));
                _mm_storeu_si128(d3.add(i) as *mut __m128i, _mm_unpackhi_epi64(t1, t3));
            }
            i += 16;
        }
    } else {
        while i + 8 <= k {
            unsafe {
                let lo = _mm_loadu_si128(s_ptr.add(i) as *const __m128i);
                let hi = _mm_loadu_si128(s_ptr.add(i + 4) as *const __m128i);

                let prev_lo = _mm_or_si128(_mm_slli_si128(lo, 4), prev_vec);
                let prev_hi = _mm_or_si128(_mm_slli_si128(hi, 4), _mm_srli_si128(lo, 12));
                prev_vec = _mm_srli_si128(hi, 12);

                let cmp_lo = _mm_cmpeq_epi32(_mm_min_epu32(lo, prev_lo), prev_lo);
                let cmp_hi = _mm_cmpeq_epi32(_mm_min_epu32(hi, prev_hi), prev_hi);
                if _mm_movemask_epi8(_mm_and_si128(cmp_lo, cmp_hi)) != 0xFFFF {
                    return -2;
                }

                let s_lo = _mm_shuffle_epi8(_mm_sub_epi32(lo, prev_lo), shuf);
                let s_hi = _mm_shuffle_epi8(_mm_sub_epi32(hi, prev_hi), shuf);
                let b01 = _mm_unpacklo_epi32(s_lo, s_hi);
                let b23 = _mm_unpackhi_epi32(s_lo, s_hi);

                _mm_storel_epi64(d0.add(i) as *mut __m128i, b01);
                _mm_storel_epi64(d1.add(i) as *mut __m128i, _mm_srli_si128(b01, 8));
                _mm_storel_epi64(d2.add(i) as *mut __m128i, b23);
                _mm_storel_epi64(d3.add(i) as *mut __m128i, _mm_srli_si128(b23, 8));
            }
            i += 8;
        }
    }

    let mut prev = if i == 0 {
        init_prev
    } else {
        unsafe { *s_ptr.add(i - 1) }
    };
    while i < k {
        unsafe {
            let cur = *s_ptr.add(i);
            if check_sorted && cur < prev {
                return -2;
            }
            let v = cur.wrapping_sub(prev);
            prev = cur;
            *d0.add(i) = (v & 0xFF) as u8;
            *d1.add(i) = ((v >> 8) & 0xFF) as u8;
            *d2.add(i) = ((v >> 16) & 0xFF) as u8;
            *d3.add(i) = ((v >> 24) & 0xFF) as u8;
        }
        i += 1;
    }
    0
}

fn delta_and_shuffle_segment_u32_scalar(
    src: &[u32],
    init_prev: u32,
    check_sorted: bool,
    chunk_len: usize,
    local_off: usize,
    dst_u8: &mut [u8],
) -> i32 {
    let mut prev = init_prev;
    for (i, &cur) in src.iter().enumerate() {
        if check_sorted && cur < prev {
            return -2;
        }
        let v = cur.wrapping_sub(prev);
        prev = cur;
        let idx = local_off + i;
        dst_u8[idx] = (v & 0xFF) as u8;
        dst_u8[chunk_len + idx] = ((v >> 8) & 0xFF) as u8;
        dst_u8[chunk_len * 2 + idx] = ((v >> 16) & 0xFF) as u8;
        dst_u8[chunk_len * 3 + idx] = ((v >> 24) & 0xFF) as u8;
    }
    0
}

/// Byte-shuffle a tile `src` (`t_len` u32 elements) into 4 planes of stride `chunk_len` at `local_off` in `dst_u8`.
#[inline]
pub fn shuffle_tile_u32(
    src: &[u32],
    chunk_len: usize,
    local_off: usize,
    dst_u8: &mut [u8],
) {
    let count = src.len();
    assert!(
        local_off
            .checked_add(count)
            .is_some_and(|end| end <= chunk_len),
        "tile [local_off..local_off + count] exceeds chunk_len"
    );
    assert!(
        chunk_len.checked_mul(4).is_some_and(|b| b <= dst_u8.len()),
        "dst_u8 too small for 4 * chunk_len bytes"
    );
    if count == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            unsafe {
                shuffle_tile_u32_ssse3(src, chunk_len, local_off, dst_u8);
            }
            return;
        }
    }
    for (i, &v) in src.iter().enumerate() {
        let idx = local_off + i;
        dst_u8[idx] = (v & 0xFF) as u8;
        dst_u8[chunk_len + idx] = ((v >> 8) & 0xFF) as u8;
        dst_u8[chunk_len * 2 + idx] = ((v >> 16) & 0xFF) as u8;
        dst_u8[chunk_len * 3 + idx] = ((v >> 24) & 0xFF) as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn shuffle_tile_u32_ssse3(
    src: &[u32],
    chunk_len: usize,
    local_off: usize,
    dst_u8: &mut [u8],
) {
    let count = src.len();
    let s_ptr = src.as_ptr();
    let d0 = unsafe { dst_u8.as_mut_ptr().add(local_off) };
    let d1 = unsafe { d0.add(chunk_len) };
    let d2 = unsafe { d0.add(chunk_len * 2) };
    let d3 = unsafe { d0.add(chunk_len * 3) };

    let mut i = 0usize;
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    while i + 16 <= count {
        unsafe {
            let v0 = _mm_shuffle_epi8(
                _mm_loadu_si128(s_ptr.add(i) as *const __m128i),
                shuf,
            );
            let v1 = _mm_shuffle_epi8(
                _mm_loadu_si128(s_ptr.add(i + 4) as *const __m128i),
                shuf,
            );
            let v2 = _mm_shuffle_epi8(
                _mm_loadu_si128(s_ptr.add(i + 8) as *const __m128i),
                shuf,
            );
            let v3 = _mm_shuffle_epi8(
                _mm_loadu_si128(s_ptr.add(i + 12) as *const __m128i),
                shuf,
            );

            let t0 = _mm_unpacklo_epi32(v0, v1);
            let t1 = _mm_unpackhi_epi32(v0, v1);
            let t2 = _mm_unpacklo_epi32(v2, v3);
            let t3 = _mm_unpackhi_epi32(v2, v3);

            let b0 = _mm_unpacklo_epi64(t0, t2);
            let b1 = _mm_unpackhi_epi64(t0, t2);
            let b2 = _mm_unpacklo_epi64(t1, t3);
            let b3 = _mm_unpackhi_epi64(t1, t3);

            _mm_storeu_si128(d0.add(i) as *mut __m128i, b0);
            _mm_storeu_si128(d1.add(i) as *mut __m128i, b1);
            _mm_storeu_si128(d2.add(i) as *mut __m128i, b2);
            _mm_storeu_si128(d3.add(i) as *mut __m128i, b3);
        }
        i += 16;
    }
    while i < count {
        unsafe {
            let v = *s_ptr.add(i);
            *d0.add(i) = (v & 0xFF) as u8;
            *d1.add(i) = ((v >> 8) & 0xFF) as u8;
            *d2.add(i) = ((v >> 16) & 0xFF) as u8;
            *d3.add(i) = ((v >> 24) & 0xFF) as u8;
        }
        i += 1;
    }
}

/// Zero-pad planes `[count .. chunk_len]` in a 4-plane shuffled buffer of stride `chunk_len`.
#[inline]
pub fn zero_pad_chunk_planes(count: usize, chunk_len: usize, dst_u8: &mut [u8]) {
    assert!(count <= chunk_len, "count exceeds chunk_len");
    assert!(
        chunk_len.checked_mul(4).is_some_and(|b| b <= dst_u8.len()),
        "dst_u8 too small for 4 * chunk_len bytes"
    );
    if count < chunk_len {
        let rem = chunk_len - count;
        unsafe {
            let d0 = dst_u8.as_mut_ptr();
            core::ptr::write_bytes(d0.add(count), 0, rem);
            core::ptr::write_bytes(d0.add(chunk_len + count), 0, rem);
            core::ptr::write_bytes(d0.add(chunk_len * 2 + count), 0, rem);
            core::ptr::write_bytes(d0.add(chunk_len * 3 + count), 0, rem);
        }
    }
}

/// Byte-shuffle `src` (`count` u32 elements, `count <= chunk_len`) into 4 planes of length `chunk_len` bytes in `dst_u8`.
/// Zero-pads planes from `count` to `chunk_len` if `count < chunk_len`.
#[inline]
pub fn shuffle_chunk_u32(src: &[u32], chunk_len: usize, dst_u8: &mut [u8]) {
    let count = src.len().min(chunk_len);
    shuffle_tile_u32(&src[..count], chunk_len, 0, dst_u8);
    zero_pad_chunk_planes(count, chunk_len, dst_u8);
}

/// Unshuffle a span `[local_start .. local_start + count]` from a 4-plane shuffled chunk of stride `chunk_len`
/// into `dst_u32[0 .. count]`.
#[inline]
pub fn unshuffle_span_u32(
    raw_u8: &[u8],
    chunk_len: usize,
    local_start: usize,
    dst_u32: &mut [u32],
) {
    unshuffle_span_u32_opt_cumsum(raw_u8, chunk_len, local_start, dst_u32, false);
}

/// Unshuffle a span `[local_start .. local_start + count]` from a 4-plane shuffled chunk of stride `chunk_len`
/// into `dst_u32[0 .. count]`, optionally performing fused register-level prefix sum (`do_cumsum`).
#[inline]
pub fn unshuffle_span_u32_opt_cumsum(
    raw_u8: &[u8],
    chunk_len: usize,
    local_start: usize,
    dst_u32: &mut [u32],
    do_cumsum: bool,
) {
    let count = dst_u32.len();
    assert!(
        local_start
            .checked_add(count)
            .is_some_and(|end| end <= chunk_len),
        "span [local_start..local_start + count] exceeds chunk_len"
    );
    assert!(
        chunk_len.checked_mul(4).is_some_and(|b| b <= raw_u8.len()),
        "raw_u8 too small for 4 * chunk_len bytes"
    );
    if count == 0 {
        return;
    }
    unsafe {
        let s0 = raw_u8.as_ptr().add(local_start);
        let s1 = s0.add(chunk_len);
        let s2 = s0.add(chunk_len * 2);
        let s3 = s0.add(chunk_len * 3);
        let d_ptr = dst_u32.as_mut_ptr();

        let mut i = 0usize;
        #[cfg(target_arch = "x86_64")]
        {
            if !do_cumsum {
                while i + 16 <= count {
                    let b0 = _mm_loadu_si128(s0.add(i) as *const __m128i);
                    let b1 = _mm_loadu_si128(s1.add(i) as *const __m128i);
                    let b2 = _mm_loadu_si128(s2.add(i) as *const __m128i);
                    let b3 = _mm_loadu_si128(s3.add(i) as *const __m128i);

                    let t0_lo = _mm_unpacklo_epi8(b0, b1);
                    let t0_hi = _mm_unpackhi_epi8(b0, b1);
                    let t1_lo = _mm_unpacklo_epi8(b2, b3);
                    let t1_hi = _mm_unpackhi_epi8(b2, b3);

                    let u0 = _mm_unpacklo_epi16(t0_lo, t1_lo);
                    let u1 = _mm_unpackhi_epi16(t0_lo, t1_lo);
                    let u2 = _mm_unpacklo_epi16(t0_hi, t1_hi);
                    let u3 = _mm_unpackhi_epi16(t0_hi, t1_hi);

                    _mm_storeu_si128(d_ptr.add(i) as *mut __m128i, u0);
                    _mm_storeu_si128(d_ptr.add(i + 4) as *mut __m128i, u1);
                    _mm_storeu_si128(d_ptr.add(i + 8) as *mut __m128i, u2);
                    _mm_storeu_si128(d_ptr.add(i + 12) as *mut __m128i, u3);
                    i += 16;
                }
                while i < count {
                    *d_ptr.add(i) = (*s0.add(i) as u32)
                        | ((*s1.add(i) as u32) << 8)
                        | ((*s2.add(i) as u32) << 16)
                        | ((*s3.add(i) as u32) << 24);
                    i += 1;
                }
                return;
            } else {
                let mut acc128 = _mm_setzero_si128();
                while i + 16 <= count {
                    let b0 = _mm_loadu_si128(s0.add(i) as *const __m128i);
                    let b1 = _mm_loadu_si128(s1.add(i) as *const __m128i);
                    let b2 = _mm_loadu_si128(s2.add(i) as *const __m128i);
                    let b3 = _mm_loadu_si128(s3.add(i) as *const __m128i);

                    let t0_lo = _mm_unpacklo_epi8(b0, b1);
                    let t0_hi = _mm_unpackhi_epi8(b0, b1);
                    let t1_lo = _mm_unpacklo_epi8(b2, b3);
                    let t1_hi = _mm_unpackhi_epi8(b2, b3);

                    let mut u0 = _mm_unpacklo_epi16(t0_lo, t1_lo);
                    let mut u1 = _mm_unpackhi_epi16(t0_lo, t1_lo);
                    let mut u2 = _mm_unpacklo_epi16(t0_hi, t1_hi);
                    let mut u3 = _mm_unpackhi_epi16(t0_hi, t1_hi);

                    u0 = _mm_add_epi32(u0, _mm_slli_si128(u0, 4));
                    u0 = _mm_add_epi32(u0, _mm_slli_si128(u0, 8));
                    u0 = _mm_add_epi32(u0, acc128);
                    _mm_storeu_si128(d_ptr.add(i) as *mut __m128i, u0);
                    acc128 = _mm_shuffle_epi32(u0, 0xFF);

                    u1 = _mm_add_epi32(u1, _mm_slli_si128(u1, 4));
                    u1 = _mm_add_epi32(u1, _mm_slli_si128(u1, 8));
                    u1 = _mm_add_epi32(u1, acc128);
                    _mm_storeu_si128(d_ptr.add(i + 4) as *mut __m128i, u1);
                    acc128 = _mm_shuffle_epi32(u1, 0xFF);

                    u2 = _mm_add_epi32(u2, _mm_slli_si128(u2, 4));
                    u2 = _mm_add_epi32(u2, _mm_slli_si128(u2, 8));
                    u2 = _mm_add_epi32(u2, acc128);
                    _mm_storeu_si128(d_ptr.add(i + 8) as *mut __m128i, u2);
                    acc128 = _mm_shuffle_epi32(u2, 0xFF);

                    u3 = _mm_add_epi32(u3, _mm_slli_si128(u3, 4));
                    u3 = _mm_add_epi32(u3, _mm_slli_si128(u3, 8));
                    u3 = _mm_add_epi32(u3, acc128);
                    _mm_storeu_si128(d_ptr.add(i + 12) as *mut __m128i, u3);
                    acc128 = _mm_shuffle_epi32(u3, 0xFF);

                    i += 16;
                }
                let mut acc = _mm_cvtsi128_si32(acc128) as u32;
                while i < count {
                    let v = (*s0.add(i) as u32)
                        | ((*s1.add(i) as u32) << 8)
                        | ((*s2.add(i) as u32) << 16)
                        | ((*s3.add(i) as u32) << 24);
                    acc = acc.wrapping_add(v);
                    *d_ptr.add(i) = acc;
                    i += 1;
                }
                return;
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut acc = 0u32;
            while i < count {
                let v = (*s0.add(i) as u32)
                    | ((*s1.add(i) as u32) << 8)
                    | ((*s2.add(i) as u32) << 16)
                    | ((*s3.add(i) as u32) << 24);
                if do_cumsum {
                    acc = acc.wrapping_add(v);
                    *d_ptr.add(i) = acc;
                } else {
                    *d_ptr.add(i) = v;
                }
                i += 1;
            }
        }
    }
}

/// In-place 32-bit wrapping prefix sum (`cumsum`) using 8-wide SSE2 instructions.
#[inline]
pub fn cumsum_u32_inplace(slice: &mut [u32]) {
    let k = slice.len();
    if k <= 1 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let ptr = slice.as_mut_ptr();
        let mut i = 0usize;
        let mut running = _mm_setzero_si128();
        while i + 8 <= k {
            let mut v0 = _mm_loadu_si128(ptr.add(i) as *const __m128i);
            let mut v1 = _mm_loadu_si128(ptr.add(i + 4) as *const __m128i);

            v0 = _mm_add_epi32(v0, _mm_slli_si128(v0, 4));
            v0 = _mm_add_epi32(v0, _mm_slli_si128(v0, 8));
            v0 = _mm_add_epi32(v0, running);
            _mm_storeu_si128(ptr.add(i) as *mut __m128i, v0);
            running = _mm_shuffle_epi32(v0, 0xFF);

            v1 = _mm_add_epi32(v1, _mm_slli_si128(v1, 4));
            v1 = _mm_add_epi32(v1, _mm_slli_si128(v1, 8));
            v1 = _mm_add_epi32(v1, running);
            _mm_storeu_si128(ptr.add(i + 4) as *mut __m128i, v1);
            running = _mm_shuffle_epi32(v1, 0xFF);

            i += 8;
        }
        let mut acc = if i > 0 { *ptr.add(i - 1) } else { 0 };
        if i == 0 {
            acc = *ptr;
            i = 1;
        }
        while i < k {
            acc = acc.wrapping_add(*ptr.add(i));
            *ptr.add(i) = acc;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut acc = slice[0];
        for slot in slice.iter_mut().skip(1) {
            acc = acc.wrapping_add(*slot);
            *slot = acc;
        }
    }
}

#[inline]
pub fn decode_pattern_u32(
    page: &[u8],
    byte_off: usize,
    dst: &mut [u32],
    is_delta: bool,
    is_shuffled: bool,
) {
    let k = dst.len();
    assert!(
        k.checked_mul(4)
            .and_then(|b| byte_off.checked_add(b))
            .is_some_and(|end| end <= page.len()),
        "page slice out of bounds for decode_pattern_u32"
    );
    if k == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                decode_pattern_u32_sse2(page, byte_off, dst, is_delta, is_shuffled);
            }
            return;
        }
    }
    decode_pattern_u32_scalar(page, byte_off, dst, is_delta, is_shuffled);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn decode_pattern_u32_sse2(
    page: &[u8],
    byte_off: usize,
    dst: &mut [u32],
    is_delta: bool,
    is_shuffled: bool,
) {
    let k = dst.len();
    let dst_ptr = dst.as_mut_ptr();

    if is_shuffled {
        let b0_ptr = unsafe { page.as_ptr().add(byte_off) };
        let b1_ptr = unsafe { b0_ptr.add(k) };
        let b2_ptr = unsafe { b0_ptr.add(2 * k) };
        let b3_ptr = unsafe { b0_ptr.add(3 * k) };

        let mut i = 0usize;
        let mut acc128 = _mm_setzero_si128();

        while i + 8 <= k {
            unsafe {
                let v0 = _mm_loadl_epi64(b0_ptr.add(i) as *const __m128i);
                let v1 = _mm_loadl_epi64(b1_ptr.add(i) as *const __m128i);
                let v2 = _mm_loadl_epi64(b2_ptr.add(i) as *const __m128i);
                let v3 = _mm_loadl_epi64(b3_ptr.add(i) as *const __m128i);

                let v01 = _mm_unpacklo_epi8(v0, v1);
                let v23 = _mm_unpacklo_epi8(v2, v3);

                let lo = _mm_unpacklo_epi16(v01, v23);
                let hi = _mm_unpackhi_epi16(v01, v23);

                if is_delta {
                    let s1 = _mm_add_epi32(lo, _mm_slli_si128(lo, 4));
                    let s2 = _mm_add_epi32(s1, _mm_slli_si128(s1, 8));
                    let res_lo = _mm_add_epi32(s2, acc128);
                    _mm_storeu_si128(dst_ptr.add(i) as *mut __m128i, res_lo);
                    acc128 = _mm_shuffle_epi32(res_lo, 0xFF);

                    let t1 = _mm_add_epi32(hi, _mm_slli_si128(hi, 4));
                    let t2 = _mm_add_epi32(t1, _mm_slli_si128(t1, 8));
                    let res_hi = _mm_add_epi32(t2, acc128);
                    _mm_storeu_si128(dst_ptr.add(i + 4) as *mut __m128i, res_hi);
                    acc128 = _mm_shuffle_epi32(res_hi, 0xFF);
                } else {
                    _mm_storeu_si128(dst_ptr.add(i) as *mut __m128i, lo);
                    _mm_storeu_si128(dst_ptr.add(i + 4) as *mut __m128i, hi);
                }
            }
            i += 8;
        }

        let mut acc = _mm_cvtsi128_si32(acc128) as u32;
        while i < k {
            unsafe {
                let v = (*b0_ptr.add(i) as u32)
                    | ((*b1_ptr.add(i) as u32) << 8)
                    | ((*b2_ptr.add(i) as u32) << 16)
                    | ((*b3_ptr.add(i) as u32) << 24);
                if is_delta {
                    acc = acc.wrapping_add(v);
                    *dst_ptr.add(i) = acc;
                } else {
                    *dst_ptr.add(i) = v;
                }
            }
            i += 1;
        }
    } else {
        let src_ptr = unsafe { page.as_ptr().add(byte_off) as *const u32 };
        if is_delta {
            let mut i = 0usize;
            let mut acc128 = _mm_setzero_si128();
            while i + 4 <= k {
                unsafe {
                    let v = _mm_loadu_si128(src_ptr.add(i) as *const __m128i);
                    let s1 = _mm_add_epi32(v, _mm_slli_si128(v, 4));
                    let s2 = _mm_add_epi32(s1, _mm_slli_si128(s1, 8));
                    let res = _mm_add_epi32(s2, acc128);
                    _mm_storeu_si128(dst_ptr.add(i) as *mut __m128i, res);
                    acc128 = _mm_shuffle_epi32(res, 0xFF);
                }
                i += 4;
            }
            let mut acc = _mm_cvtsi128_si32(acc128) as u32;
            while i < k {
                unsafe {
                    let v = std::ptr::read_unaligned(src_ptr.add(i));
                    acc = acc.wrapping_add(v);
                    *dst_ptr.add(i) = acc;
                }
                i += 1;
            }
        } else {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr as *const u8, dst_ptr as *mut u8, k * 4);
            }
        }
    }
}

fn decode_pattern_u32_scalar(
    page: &[u8],
    byte_off: usize,
    dst: &mut [u32],
    is_delta: bool,
    is_shuffled: bool,
) {
    let k = dst.len();
    let mut acc = 0u32;
    if is_shuffled {
        for j in 0..k {
            let v = (page[byte_off + j] as u32)
                | ((page[byte_off + k + j] as u32) << 8)
                | ((page[byte_off + 2 * k + j] as u32) << 16)
                | ((page[byte_off + 3 * k + j] as u32) << 24);
            if is_delta {
                acc = acc.wrapping_add(v);
                dst[j] = acc;
            } else {
                dst[j] = v;
            }
        }
    } else {
        for (j, slot) in dst.iter_mut().enumerate() {
            let off = byte_off + j * 4;
            let v = u32::from_le_bytes(page[off..off + 4].try_into().unwrap());
            if is_delta {
                acc = acc.wrapping_add(v);
                *slot = acc;
            } else {
                *slot = v;
            }
        }
    }
}

#[inline]
pub fn encode_pattern_u32(
    src: &[u32],
    page: &mut [u8],
    byte_off: usize,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
) -> i32 {
    let k = src.len();
    assert!(
        k.checked_mul(4)
            .and_then(|b| byte_off.checked_add(b))
            .is_some_and(|end| end <= page.len()),
        "page slice out of bounds for encode_pattern_u32"
    );
    if k == 0 {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3") {
            unsafe {
                return encode_pattern_u32_sse41(
                    src,
                    page,
                    byte_off,
                    is_delta,
                    is_shuffled,
                    check_sorted,
                );
            }
        }
    }
    encode_pattern_u32_scalar(src, page, byte_off, is_delta, is_shuffled, check_sorted)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1,ssse3")]
unsafe fn encode_pattern_u32_sse41(
    src: &[u32],
    page: &mut [u8],
    byte_off: usize,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
) -> i32 {
    let k = src.len();
    let src_ptr = src.as_ptr();
    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    if is_shuffled {
        let b0_ptr = unsafe { page.as_mut_ptr().add(byte_off) };
        let b1_ptr = unsafe { b0_ptr.add(k) };
        let b2_ptr = unsafe { b0_ptr.add(2 * k) };
        let b3_ptr = unsafe { b0_ptr.add(3 * k) };

        let mut i = 0usize;
        let mut prev_vec = _mm_setzero_si128();

        while i + 8 <= k {
            unsafe {
                let lo = _mm_loadu_si128(src_ptr.add(i) as *const __m128i);
                let hi = _mm_loadu_si128(src_ptr.add(i + 4) as *const __m128i);

                let (d_lo, d_hi) = if is_delta || check_sorted {
                    let prev_lo = _mm_or_si128(_mm_slli_si128(lo, 4), prev_vec);
                    let prev_hi = _mm_or_si128(_mm_slli_si128(hi, 4), _mm_srli_si128(lo, 12));
                    prev_vec = _mm_srli_si128(hi, 12);

                    if check_sorted {
                        let cmp_lo = _mm_cmpeq_epi32(_mm_min_epu32(lo, prev_lo), prev_lo);
                        let cmp_hi = _mm_cmpeq_epi32(_mm_min_epu32(hi, prev_hi), prev_hi);
                        if _mm_movemask_epi8(_mm_and_si128(cmp_lo, cmp_hi)) != 0xFFFF {
                            return -2;
                        }
                    }
                    if is_delta {
                        (_mm_sub_epi32(lo, prev_lo), _mm_sub_epi32(hi, prev_hi))
                    } else {
                        (lo, hi)
                    }
                } else {
                    (lo, hi)
                };

                let s_lo = _mm_shuffle_epi8(d_lo, shuf);
                let s_hi = _mm_shuffle_epi8(d_hi, shuf);
                let b01 = _mm_unpacklo_epi32(s_lo, s_hi);
                let b23 = _mm_unpackhi_epi32(s_lo, s_hi);

                _mm_storel_epi64(b0_ptr.add(i) as *mut __m128i, b01);
                _mm_storel_epi64(b1_ptr.add(i) as *mut __m128i, _mm_srli_si128(b01, 8));
                _mm_storel_epi64(b2_ptr.add(i) as *mut __m128i, b23);
                _mm_storel_epi64(b3_ptr.add(i) as *mut __m128i, _mm_srli_si128(b23, 8));
            }
            i += 8;
        }

        let mut prev = if i == 0 {
            0u32
        } else {
            unsafe { *src_ptr.add(i - 1) }
        };
        while i < k {
            unsafe {
                let cur = *src_ptr.add(i);
                if check_sorted && cur < prev {
                    return -2;
                }
                let val = if is_delta { cur.wrapping_sub(prev) } else { cur };
                prev = cur;
                *b0_ptr.add(i) = (val & 0xFF) as u8;
                *b1_ptr.add(i) = ((val >> 8) & 0xFF) as u8;
                *b2_ptr.add(i) = ((val >> 16) & 0xFF) as u8;
                *b3_ptr.add(i) = ((val >> 24) & 0xFF) as u8;
            }
            i += 1;
        }
    } else {
        let dst_ptr = unsafe { page.as_mut_ptr().add(byte_off) as *mut u32 };
        if !is_delta && !check_sorted {
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr as *const u8, dst_ptr as *mut u8, k * 4);
            }
            return 0;
        }
        let mut prev = 0u32;
        for i in 0..k {
            unsafe {
                let cur = *src_ptr.add(i);
                if check_sorted && cur < prev {
                    return -2;
                }
                let val = if is_delta { cur.wrapping_sub(prev) } else { cur };
                prev = cur;
                std::ptr::write_unaligned(dst_ptr.add(i), val);
            }
        }
    }
    0
}

fn encode_pattern_u32_scalar(
    src: &[u32],
    page: &mut [u8],
    byte_off: usize,
    is_delta: bool,
    is_shuffled: bool,
    check_sorted: bool,
) -> i32 {
    let k = src.len();
    let mut prev = 0u32;
    if is_shuffled {
        for j in 0..k {
            let cur = src[j];
            if check_sorted && cur < prev {
                return -2;
            }
            let val = if is_delta { cur.wrapping_sub(prev) } else { cur };
            prev = cur;
            page[byte_off + j] = (val & 0xFF) as u8;
            page[byte_off + k + j] = ((val >> 8) & 0xFF) as u8;
            page[byte_off + 2 * k + j] = ((val >> 16) & 0xFF) as u8;
            page[byte_off + 3 * k + j] = ((val >> 24) & 0xFF) as u8;
        }
    } else {
        for (j, &cur) in src.iter().enumerate() {
            if check_sorted && cur < prev {
                return -2;
            }
            let val = if is_delta { cur.wrapping_sub(prev) } else { cur };
            prev = cur;
            let bytes = val.to_le_bytes();
            let off = byte_off + j * 4;
            page[off..off + 4].copy_from_slice(&bytes);
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_shuffle_and_unshuffle_roundtrip() {
        for k in 0..=37 {
            let chunk_len = k + 5;
            let local_off = 3;
            let src: Vec<u32> = (0..k).map(|i| (i as u32) * 1031 + 17).collect();
            let mut dst_u8 = vec![0xAAu8; chunk_len * 4];
            let rc = delta_and_shuffle_segment_u32(&src, 0, true, chunk_len, local_off, &mut dst_u8);
            assert_eq!(rc, 0);

            let mut out = vec![0u32; k];
            unshuffle_span_u32_opt_cumsum(&dst_u8, chunk_len, local_off, &mut out, true);
            assert_eq!(out, src);
        }
    }

    #[test]
    fn test_delta_shuffle_unsorted_detection() {
        for k in 2..=25 {
            for bad_pos in 1..k {
                let mut src: Vec<u32> = (0..k).map(|i| (i as u32) * 10 + 10).collect();
                src[bad_pos] = src[bad_pos - 1] - 1;
                let mut dst_u8 = vec![0u8; k * 4];
                assert_eq!(
                    delta_and_shuffle_segment_u32(&src, 0, true, k, 0, &mut dst_u8),
                    -2
                );
                assert_eq!(
                    delta_and_shuffle_segment_u32(&src, 0, false, k, 0, &mut dst_u8),
                    0
                );
            }
        }
    }

    #[test]
    fn test_shuffle_chunk_and_cumsum() {
        for k in 0..=35 {
            let chunk_len = (k + 4).max(1);
            let src: Vec<u32> = (0..k)
                .map(|i| 0xDEAD_0000u32.wrapping_add((i as u32) * 0x0102_0304))
                .collect();
            let mut buf = vec![0xFFu8; chunk_len * 4];
            shuffle_chunk_u32(&src, chunk_len, &mut buf);
            let mut out = vec![0u32; chunk_len];
            unshuffle_span_u32(&buf, chunk_len, 0, &mut out);
            assert_eq!(&out[..k], &src[..]);
            assert!(out[k..].iter().all(|&v| v == 0));

            let mut cs = src.clone();
            cumsum_u32_inplace(&mut cs);
            let mut expected = Vec::with_capacity(k);
            let mut acc = 0u32;
            for &v in &src {
                acc = acc.wrapping_add(v);
                expected.push(acc);
            }
            assert_eq!(cs, expected);
        }
    }

    #[test]
    fn test_encode_decode_pattern_all_modes() {
        for &is_delta in &[false, true] {
            for &is_shuffled in &[false, true] {
                for &byte_off in &[0usize, 1, 3, 7] {
                    for k in 0..=35 {
                        let src: Vec<u32> = (0..k).map(|i| (i as u32) * 257 + 5).collect();
                        let mut page = vec![0u8; byte_off + k * 4 + 8];
                        let rc = encode_pattern_u32(
                            &src,
                            &mut page,
                            byte_off,
                            is_delta,
                            is_shuffled,
                            true,
                        );
                        assert_eq!(rc, 0);
                        let mut dst = vec![0u32; k];
                        decode_pattern_u32(&page, byte_off, &mut dst, is_delta, is_shuffled);
                        assert_eq!(dst, src);
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "exceeds chunk_len")]
    fn test_unshuffle_oob_panics() {
        let raw = vec![0u8; 16];
        let mut dst = vec![0u32; 3];
        unshuffle_span_u32(&raw, 4, 2, &mut dst);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_decode_pattern_oob_panics() {
        let page = vec![0u8; 10];
        let mut dst = vec![0u32; 3];
        decode_pattern_u32(&page, 0, &mut dst, false, false);
    }
}
