#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub fn decode_pattern_u32(
    page: &[u8],
    byte_off: usize,
    dst: &mut [u32],
    is_delta: bool,
    is_shuffled: bool,
) {
    let k = dst.len();
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
                    let lo = _mm_loadu_si128(src_ptr.add(i) as *const __m128i);
                    let s1 = _mm_add_epi32(lo, _mm_slli_si128(lo, 4));
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
                std::ptr::copy_nonoverlapping(page.as_ptr().add(byte_off), dst_ptr as *mut u8, k * 4);
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
    if is_shuffled {
        let b0 = &page[byte_off..byte_off + k];
        let b1 = &page[byte_off + k..byte_off + 2 * k];
        let b2 = &page[byte_off + 2 * k..byte_off + 3 * k];
        let b3 = &page[byte_off + 3 * k..byte_off + 4 * k];
        if is_delta {
            let mut acc = 0u32;
            for j in 0..k {
                let v = (b0[j] as u32)
                    | ((b1[j] as u32) << 8)
                    | ((b2[j] as u32) << 16)
                    | ((b3[j] as u32) << 24);
                acc = acc.wrapping_add(v);
                dst[j] = acc;
            }
        } else {
            for j in 0..k {
                dst[j] = (b0[j] as u32)
                    | ((b1[j] as u32) << 8)
                    | ((b2[j] as u32) << 16)
                    | ((b3[j] as u32) << 24);
            }
        }
    } else if is_delta {
        let mut acc = 0u32;
        for j in 0..k {
            let off = byte_off + j * 4;
            let v = u32::from_le_bytes([page[off], page[off + 1], page[off + 2], page[off + 3]]);
            acc = acc.wrapping_add(v);
            dst[j] = acc;
        }
    } else {
        unsafe {
            std::ptr::copy_nonoverlapping(
                page.as_ptr().add(byte_off),
                dst.as_mut_ptr() as *mut u8,
                k * 4,
            );
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
    if k == 0 {
        return 0;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") && is_x86_feature_detected!("ssse3") {
            return unsafe {
                encode_pattern_u32_sse41(src, page, byte_off, is_delta, is_shuffled, check_sorted)
            };
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
        for j in 0..k {
            let cur = src[j];
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
