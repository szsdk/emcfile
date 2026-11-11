use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Source block of 65,536 pixels (`local_p: u16`, 256 KiB for `i32`, 64 OS pages -> L1 dTLB resident)
/// paired with destination tiles of 4,096 frame pixels (`local_d: u16`, 32 KiB of `f64` -> L1d resident,
/// `<= 4` ways per L1d cache set on 1024-wide detectors).
const SRC_BLK_SHIFT: usize = 16;
const SRC_BLK_LEN: usize = 1 << SRC_BLK_SHIFT;

const DST_TILE_SHIFT: usize = 12;
const DST_TILE_LEN: usize = 1 << DST_TILE_SHIFT;
const DST_TILE_MASK: usize = DST_TILE_LEN - 1;

trait ToF64: Element + Copy {
    fn to_f64(self) -> f64;
}

macro_rules! impl_to_f64 {
    ($($t:ty),*) => {
        $(
            impl ToF64 for $t {
                #[inline(always)]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_to_f64!(f64, f32, i64, u64, i32, u32, i16, u16, i8, u8);

trait MaskBits: Element + Copy {
    fn is_bad(self) -> bool;
}

macro_rules! impl_mask_bits {
    ($($t:ty),*) => {
        $(
            impl MaskBits for $t {
                #[inline(always)]
                fn is_bad(self) -> bool {
                    (self & 2) != 0
                }
            }
        )*
    };
}

impl_mask_bits!(u8, i8, u16, i16, u32, i32, u64, i64);

#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct MicroPair {
    pub local_p: u16,
    pub local_d: u16,
}

#[derive(Clone, Copy)]
pub(crate) struct MicroTileSpan {
    pub src_base: usize,
    pub src_len: usize,
    pub pair_beg: usize,
    pub pair_end: usize,
}

/// Immutable single-threaded execution plan for `DetectorRenderer`.
///
/// Invariants established at construction:
/// - `frame_len == width * height`
/// - `inv_count.len() == frame_len`
/// - `mask_bool.len() == frame_len` when `has_any_masked` is true
/// - `flat_u32.len() == num_pix` and every `idx` in `flat_u32` satisfies `(idx as usize) < frame_len`
/// - `tile_span_offsets.len() == num_dst_tiles + 1` when `!is_identity_like`
/// - Every `MicroTileSpan` for tile `t` (`dst_len = (t * DST_TILE_LEN + DST_TILE_LEN).min(frame_len) - t * DST_TILE_LEN`)
///   has `src_base + src_len <= num_pix`, `pair_beg <= pair_end <= pairs.len()`, and for every
///   `pair` in `pairs[pair_beg..pair_end]`, `(pair.local_p as usize) < src_len` and `(pair.local_d as usize) < dst_len`.
#[pyclass(frozen)]
pub struct RustRendererPlan {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) frame_len: usize,
    pub(crate) num_pix: usize,
    pub(crate) num_dst_tiles: usize,
    pub(crate) is_exact_identity: bool,
    pub(crate) is_identity_like: bool,
    pub(crate) all_unit_inv: bool,
    pub(crate) flat_u32: Vec<u32>,
    pub(crate) tile_span_offsets: Vec<usize>,
    pub(crate) spans: Vec<MicroTileSpan>,
    pub(crate) pairs: Vec<MicroPair>,
    pub(crate) inv_count: Vec<f64>,
    pub(crate) mask_bool: Vec<bool>,
    pub(crate) has_any_masked: bool,
}

impl RustRendererPlan {
    /// Overwrites every element of `out[0..self.frame_len]` with the rendered frame for `src`.
    #[inline]
    fn write_frame<T: ToF64>(&self, src: &[T], out: &mut [f64]) {
        assert_eq!(src.len(), self.num_pix);
        assert_eq!(out.len(), self.frame_len);

        if self.is_exact_identity {
            if self.all_unit_inv {
                for (o, &val) in out.iter_mut().zip(src.iter()) {
                    *o = val.to_f64();
                }
            } else {
                for ((o, &val), &inv) in out.iter_mut().zip(src.iter()).zip(self.inv_count.iter()) {
                    *o = val.to_f64() * inv;
                }
            }
            return;
        }

        if self.is_identity_like {
            out.fill(0.0);
            let out_ptr = out.as_mut_ptr();
            for (&idx, &val) in self.flat_u32.iter().zip(src.iter()) {
                // SAFETY: `idx as usize < self.frame_len == out.len()`.
                unsafe {
                    *out_ptr.add(idx as usize) += val.to_f64();
                }
            }
            if !self.all_unit_inv {
                for (o, &inv) in out.iter_mut().zip(self.inv_count.iter()) {
                    *o *= inv;
                }
            }
            return;
        }

        let mut buf = [0.0_f64; DST_TILE_LEN];
        for t in 0..self.num_dst_tiles {
            let d_base = t << DST_TILE_SHIFT;
            let d_end = (d_base + DST_TILE_LEN).min(self.frame_len);
            let d_len = d_end - d_base;
            let out_tile = &mut out[d_base..d_end];
            let span_beg = self.tile_span_offsets[t];
            let span_end = self.tile_span_offsets[t + 1];
            if span_beg == span_end {
                out_tile.fill(f64::NAN);
                continue;
            }
            let buf_sub = &mut buf[..d_len];
            buf_sub.fill(0.0);
            let dst_ptr = buf_sub.as_mut_ptr();

            for span in &self.spans[span_beg..span_end] {
                let src_sub = &src[span.src_base..span.src_base + span.src_len];
                let sub_pairs = &self.pairs[span.pair_beg..span.pair_end];
                let src_ptr = src_sub.as_ptr();

                for pair in sub_pairs {
                    let lp = pair.local_p as usize;
                    let ld = pair.local_d as usize;
                    debug_assert!(lp < span.src_len);
                    debug_assert!(ld < d_len);
                    // SAFETY: `lp < span.src_len == src_sub.len()` and `ld < d_len == buf_sub.len()`
                    // were validated when constructing `RustRendererPlan`.
                    unsafe {
                        *dst_ptr.add(ld) += (*src_ptr.add(lp)).to_f64();
                    }
                }
            }

            if self.all_unit_inv {
                out_tile.copy_from_slice(buf_sub);
            } else {
                let inv_tile = &self.inv_count[d_base..d_end];
                for ((o, &b), &inv) in out_tile.iter_mut().zip(buf_sub.iter()).zip(inv_tile.iter())
                {
                    *o = b * inv;
                }
            }
        }
    }

    /// Accumulates one sparse pattern directly into a zeroed `out[0..self.frame_len]` slice.
    #[inline]
    fn accumulate_sparse_zeroed(
        &self,
        place_ones: &[u32],
        place_multi: &[u32],
        count_multi: &[i32],
        out: &mut [f64],
    ) -> Result<(), &'static str> {
        assert_eq!(out.len(), self.frame_len);
        if place_multi.len() != count_multi.len() {
            return Err("place_multi and count_multi must have identical lengths");
        }
        let num_pix = self.num_pix;
        let flat_ptr = self.flat_u32.as_ptr();
        let out_ptr = out.as_mut_ptr();

        for &pix in place_ones {
            let p = pix as usize;
            if p >= num_pix {
                return Err("place_ones pixel index out of range");
            }
            // SAFETY: `p < num_pix == self.flat_u32.len()` and `f < self.frame_len == out.len()`.
            unsafe {
                let f = *flat_ptr.add(p) as usize;
                *out_ptr.add(f) += 1.0;
            }
        }

        for (&pix, &cnt) in place_multi.iter().zip(count_multi.iter()) {
            let p = pix as usize;
            if p >= num_pix {
                return Err("place_multi pixel index out of range");
            }
            // SAFETY: `p < num_pix == self.flat_u32.len()` and `f < self.frame_len == out.len()`.
            unsafe {
                let f = *flat_ptr.add(p) as usize;
                *out_ptr.add(f) += cnt as f64;
            }
        }

        if !self.all_unit_inv {
            for (o, &inv) in out.iter_mut().zip(self.inv_count.iter()) {
                *o *= inv;
            }
        }
        Ok(())
    }

    fn run_1d<'py, T: ToF64>(
        &self,
        py: Python<'py>,
        arr: PyReadonlyArray1<'py, T>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)>> {
        if !arr.is_c_contiguous() {
            return Ok(None);
        }
        let src = match arr.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        if src.len() != self.num_pix {
            return Err(PyValueError::new_err(format!(
                "raw_img has {} pixels; the detector requires {}",
                src.len(),
                self.num_pix
            )));
        }

        // SAFETY: `f64` is a primitive numeric type and `write_frame` overwrites all `height * width` elements.
        let rendered = unsafe { PyArray2::<f64>::new(py, [self.height, self.width], false) };
        // SAFETY: `rendered` was freshly allocated with exclusive access here.
        let out_slice = unsafe { rendered.as_slice_mut()? };
        py.detach(|| {
            self.write_frame(src, out_slice);
        });

        let mask = if self.has_any_masked {
            PyArray1::from_slice(py, &self.mask_bool).reshape([self.height, self.width])?
        } else {
            PyArray2::<bool>::zeros(py, [self.height, self.width], false)
        };
        Ok(Some((rendered, mask)))
    }

    fn run_2d<'py, T: ToF64>(
        &self,
        py: Python<'py>,
        arr: PyReadonlyArray2<'py, T>,
    ) -> PyResult<Option<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<bool>>)>> {
        if !arr.is_c_contiguous() {
            return Ok(None);
        }
        let shape = arr.shape();
        let n_images = shape[0];
        let cols = shape[1];
        if cols != self.num_pix {
            return Err(PyValueError::new_err(format!(
                "raw_img has {} pixels; the detector requires {}",
                cols, self.num_pix
            )));
        }
        let src = match arr.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };

        // SAFETY: `f64` is a primitive numeric type and `write_frame` overwrites all `n_images * height * width` elements.
        let rendered =
            unsafe { PyArray3::<f64>::new(py, [n_images, self.height, self.width], false) };
        // SAFETY: `rendered` was freshly allocated with exclusive access here.
        let out_slice = unsafe { rendered.as_slice_mut()? };

        let mask = if self.has_any_masked {
            // SAFETY: overwritten completely below when `has_any_masked` is true.
            unsafe { PyArray3::<bool>::new(py, [n_images, self.height, self.width], false) }
        } else {
            PyArray3::<bool>::zeros(py, [n_images, self.height, self.width], false)
        };
        // SAFETY: `mask` was freshly allocated with exclusive access here.
        let mask_out = unsafe { mask.as_slice_mut()? };

        let frame_len = self.frame_len;
        let num_pix = self.num_pix;
        let has_any_masked = self.has_any_masked;
        let mask_bool = &self.mask_bool;

        py.detach(|| {
            if frame_len > 0 {
                for (img_idx, out_frame) in out_slice.chunks_exact_mut(frame_len).enumerate() {
                    let in_frame = &src[img_idx * num_pix..(img_idx + 1) * num_pix];
                    self.write_frame(in_frame, out_frame);
                }
                if has_any_masked {
                    for mask_frame in mask_out.chunks_exact_mut(frame_len) {
                        mask_frame.copy_from_slice(mask_bool);
                    }
                }
            }
        });

        Ok(Some((rendered, mask)))
    }
}

#[pymethods]
impl RustRendererPlan {
    /// Render a C-contiguous 1D `(num_pix,)` numeric array on a single thread.
    /// Returns `None` if `raw_img` is non-contiguous or uses an unsupported dtype.
    fn render_dense_1d<'py>(
        &self,
        py: Python<'py>,
        raw_img: &Bound<'py, PyAny>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)>> {
        macro_rules! try_1d {
            ($($t:ty),*) => {
                $(
                    if let Ok(arr) = raw_img.extract::<PyReadonlyArray1<'py, $t>>() {
                        return self.run_1d(py, arr);
                    }
                )*
            };
        }
        try_1d!(i32, f64, f32, u32, i64, u64, i16, u16, i8, u8);
        Ok(None)
    }

    /// Render a C-contiguous 2D `(n_images, num_pix)` numeric array on a single thread.
    /// Returns `None` if `raw_img` is non-contiguous or uses an unsupported dtype.
    fn render_dense_2d<'py>(
        &self,
        py: Python<'py>,
        raw_img: &Bound<'py, PyAny>,
    ) -> PyResult<Option<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<bool>>)>> {
        macro_rules! try_2d {
            ($($t:ty),*) => {
                $(
                    if let Ok(arr) = raw_img.extract::<PyReadonlyArray2<'py, $t>>() {
                        return self.run_2d(py, arr);
                    }
                )*
            };
        }
        try_2d!(i32, f64, f32, u32, i64, u64, i16, u16, i8, u8);
        Ok(None)
    }

    /// Render a single `SPARSE_PATTERN` (`place_ones`, `place_multi`, `count_multi`) directly on a single thread.
    fn render_sparse_1d<'py>(
        &self,
        py: Python<'py>,
        place_ones: PyReadonlyArray1<'py, u32>,
        place_multi: PyReadonlyArray1<'py, u32>,
        count_multi: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>)>> {
        if !place_ones.is_c_contiguous()
            || !place_multi.is_c_contiguous()
            || !count_multi.is_c_contiguous()
        {
            return Ok(None);
        }
        let ones_s = match place_ones.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let multi_s = match place_multi.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let count_s = match count_multi.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };

        let rendered = PyArray2::<f64>::zeros(py, [self.height, self.width], false);
        // SAFETY: `rendered` was freshly allocated with exclusive access here.
        let out_slice = unsafe { rendered.as_slice_mut()? };

        py.detach(|| {
            self.accumulate_sparse_zeroed(ones_s, multi_s, count_s, out_slice)
        })
        .map_err(PyValueError::new_err)?;

        let mask = if self.has_any_masked {
            PyArray1::from_slice(py, &self.mask_bool).reshape([self.height, self.width])?
        } else {
            PyArray2::<bool>::zeros(py, [self.height, self.width], false)
        };
        Ok(Some((rendered, mask)))
    }

    /// Render a `PatternsSOne` batch directly from CSR-style sparse arrays on a single thread.
    fn render_sparse_2d<'py>(
        &self,
        py: Python<'py>,
        ones_idx: PyReadonlyArray1<'py, u64>,
        multi_idx: PyReadonlyArray1<'py, u64>,
        place_ones: PyReadonlyArray1<'py, u32>,
        place_multi: PyReadonlyArray1<'py, u32>,
        count_multi: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<Option<(Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<bool>>)>> {
        if !ones_idx.is_c_contiguous()
            || !multi_idx.is_c_contiguous()
            || !place_ones.is_c_contiguous()
            || !place_multi.is_c_contiguous()
            || !count_multi.is_c_contiguous()
        {
            return Ok(None);
        }
        let o_idx = match ones_idx.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let m_idx = match multi_idx.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let ones_s = match place_ones.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let multi_s = match place_multi.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };
        let count_s = match count_multi.as_slice() {
            Ok(s) => s,
            Err(_) => return Ok(None),
        };

        if o_idx.is_empty() || o_idx.len() != m_idx.len() {
            return Err(PyValueError::new_err(
                "ones_idx and multi_idx must have identical positive lengths",
            ));
        }
        if multi_s.len() != count_s.len() {
            return Err(PyValueError::new_err(
                "place_multi and count_multi must have identical lengths",
            ));
        }
        let n_images = o_idx.len() - 1;

        let rendered = PyArray3::<f64>::zeros(py, [n_images, self.height, self.width], false);
        // SAFETY: `rendered` was freshly allocated with exclusive access here.
        let out_slice = unsafe { rendered.as_slice_mut()? };

        let mask = if self.has_any_masked {
            // SAFETY: overwritten completely below when `has_any_masked` is true.
            unsafe { PyArray3::<bool>::new(py, [n_images, self.height, self.width], false) }
        } else {
            PyArray3::<bool>::zeros(py, [n_images, self.height, self.width], false)
        };
        // SAFETY: `mask` was freshly allocated with exclusive access here.
        let mask_out = unsafe { mask.as_slice_mut()? };

        let frame_len = self.frame_len;
        let has_any_masked = self.has_any_masked;
        let mask_bool = &self.mask_bool;

        py.detach(|| -> Result<(), &'static str> {
            if frame_len > 0 {
                for (img_idx, out_frame) in out_slice.chunks_exact_mut(frame_len).enumerate() {
                    let o_beg = o_idx[img_idx] as usize;
                    let o_end = o_idx[img_idx + 1] as usize;
                    let m_beg = m_idx[img_idx] as usize;
                    let m_end = m_idx[img_idx + 1] as usize;
                    if o_beg > o_end
                        || o_end > ones_s.len()
                        || m_beg > m_end
                        || m_end > multi_s.len()
                    {
                        return Err("sparse pattern offset out of bounds");
                    }
                    self.accumulate_sparse_zeroed(
                        &ones_s[o_beg..o_end],
                        &multi_s[m_beg..m_end],
                        &count_s[m_beg..m_end],
                        out_frame,
                    )?;
                }
                if has_any_masked {
                    for mask_frame in mask_out.chunks_exact_mut(frame_len) {
                        mask_frame.copy_from_slice(mask_bool);
                    }
                }
            }
            Ok(())
        })
        .map_err(PyValueError::new_err)?;

        Ok(Some((rendered, mask)))
    }
}

type PreparedRendererTuple<'py> = (
    i32,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<i32>>,
    usize,
    usize,
    Bound<'py, PyArray2<u8>>,
    Bound<'py, PyArray2<f64>>,
    Py<RustRendererPlan>,
);

struct CorePlanData {
    direction: i32,
    width: usize,
    height: usize,
    frame_len: usize,
    num_dst_tiles: usize,
    mask_u8: Vec<u8>,
    mask_bool: Vec<bool>,
    has_any_masked: bool,
    count_data: Vec<f64>,
    is_exact_identity: bool,
    is_identity_like: bool,
    all_unit_inv: bool,
    flat_u32: Vec<u32>,
    tile_span_offsets: Vec<usize>,
    spans: Vec<MicroTileSpan>,
    pairs: Vec<MicroPair>,
    inv_count: Vec<f64>,
}

fn build_renderer_core<M: MaskBits>(
    coor_s: &[f64],
    mask_s: &[M],
    detd: f64,
    ewald_rad: f64,
    cxy_out: &mut [f64],
    xy_out: &mut [i32],
) -> Result<CorePlanData, &'static str> {
    let num_pix = mask_s.len();
    if num_pix > (u32::MAX as usize) {
        return Err("num_pix exceeds u32 index range");
    }

    let mut z_sum = 0.0_f64;
    for i in 0..num_pix {
        z_sum += coor_s[3 * i + 2];
    }
    let direction: i32 = if z_sum > 0.0 {
        -1
    } else if z_sum < 0.0 {
        1
    } else {
        0
    };

    let is_inf = ewald_rad.is_infinite();
    let dir_f = direction as f64;
    let scale = detd / ewald_rad;

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;

    if is_inf {
        for i in 0..num_pix {
            let cx = coor_s[3 * i];
            let cy = coor_s[3 * i + 1];
            if !cx.is_finite() || !cy.is_finite() {
                return Err("non-finite projected coordinate");
            }
            cxy_out[2 * i] = cx;
            cxy_out[2 * i + 1] = cy;
            if cx < min_x {
                min_x = cx;
            }
            if cy < min_y {
                min_y = cy;
            }
        }
    } else {
        for i in 0..num_pix {
            let x = coor_s[3 * i];
            let y = coor_s[3 * i + 1];
            let z = coor_s[3 * i + 2];
            let denom = ewald_rad + dir_f * z;
            let cx = (x * ewald_rad / denom) * scale;
            let cy = (y * ewald_rad / denom) * scale;
            if !cx.is_finite() || !cy.is_finite() {
                return Err("non-finite projected coordinate");
            }
            cxy_out[2 * i] = cx;
            cxy_out[2 * i + 1] = cy;
            if cx < min_x {
                min_x = cx;
            }
            if cy < min_y {
                min_y = cy;
            }
        }
    }

    let mut max_x: i32 = 0;
    let mut max_y: i32 = 0;
    let i32_limit = (i32::MAX - 1) as f64;
    for i in 0..num_pix {
        let rx = (cxy_out[2 * i] - min_x).round_ties_even();
        let ry = (cxy_out[2 * i + 1] - min_y).round_ties_even();
        if !(0.0..=i32_limit).contains(&rx) || !(0.0..=i32_limit).contains(&ry) {
            return Err("projected pixel coordinate overflows int32");
        }
        let ix = rx as i32;
        let iy = ry as i32;
        xy_out[2 * i] = ix;
        xy_out[2 * i + 1] = iy;
        if ix > max_x {
            max_x = ix;
        }
        if iy > max_y {
            max_y = iy;
        }
    }

    let width = (max_x as usize) + 1;
    let height = (max_y as usize) + 1;
    let frame_len = width
        .checked_mul(height)
        .ok_or("frame dimensions overflow usize")?;
    if frame_len > (u32::MAX as usize) {
        return Err("frame dimensions exceed u32 flat index range");
    }

    let mut flat_u32 = Vec::with_capacity(num_pix);
    let mut mask_u8 = vec![0_u8; frame_len];
    let mut has_any_masked = false;

    // Detect if sequential source order is exact identity or row-major-like
    let mut large_jumps: usize = 0;
    let mut prev_f: usize = 0;
    let mut is_exact_identity = num_pix == frame_len;
    for (i, &m) in mask_s.iter().enumerate() {
        let ix = xy_out[2 * i] as usize;
        let iy = xy_out[2 * i + 1] as usize;
        let f = iy * width + ix;
        debug_assert!(f < frame_len);
        if f != i {
            is_exact_identity = false;
        }
        if i > 0 && f.abs_diff(prev_f) > 64 {
            large_jumps += 1;
        }
        prev_f = f;
        flat_u32.push(f as u32);
        if m.is_bad() {
            mask_u8[f] = 1;
            has_any_masked = true;
        }
    }

    let mask_bool = if has_any_masked {
        mask_u8.iter().map(|&b| b != 0).collect()
    } else {
        Vec::new()
    };

    // Row-major-like if < 5% of steps jump by > 64 elements (e.g. only at row boundaries).
    let is_identity_like = is_exact_identity || (large_jumps * 20 <= num_pix);
    let num_dst_tiles = frame_len.div_ceil(DST_TILE_LEN);

    let mut raw_counts = vec![0_u32; frame_len];
    let mut tile_span_offsets = Vec::new();
    let mut spans = Vec::new();
    let mut pairs = Vec::new();

    if is_exact_identity {
        raw_counts.fill(1);
    } else if is_identity_like {
        for &f32_idx in &flat_u32 {
            raw_counts[f32_idx as usize] += 1;
        }
    } else {
        // Build `dst_tile`-outer, `src_blk`-inner spans with L1d-sized destination tiles (`4,096` f64s = 32 KiB).
        let num_src_blks = num_pix.div_ceil(SRC_BLK_LEN);
        let mut tile_cnt = vec![0_usize; num_dst_tiles];
        for &f32_idx in &flat_u32 {
            let f = f32_idx as usize;
            if !has_any_masked || mask_u8[f] == 0 {
                tile_cnt[f >> DST_TILE_SHIFT] += 1;
            } else {
                raw_counts[f] += 1;
            }
        }

        let mut tile_pair_off = Vec::with_capacity(num_dst_tiles + 1);
        let mut total_pairs = 0_usize;
        tile_pair_off.push(0);
        for &c in &tile_cnt {
            total_pairs += c;
            tile_pair_off.push(total_pairs);
        }

        pairs.resize(
            total_pairs,
            MicroPair {
                local_p: 0,
                local_d: 0,
            },
        );
        let mut tile_cur = tile_pair_off[..num_dst_tiles].to_vec();
        let mut tile_prev = tile_cur.clone();
        let mut tmp_spans: Vec<(u32, MicroTileSpan)> = Vec::new();
        let mut tile_span_cnt = vec![0_usize; num_dst_tiles];

        for sb in 0..num_src_blks {
            let s_base = sb << SRC_BLK_SHIFT;
            let s_end = (s_base + SRC_BLK_LEN).min(num_pix);
            let s_len = s_end - s_base;
            let flat_sub = &flat_u32[s_base..s_end];

            for (lp, &f32_idx) in flat_sub.iter().enumerate() {
                let f = f32_idx as usize;
                if !has_any_masked || mask_u8[f] == 0 {
                    let t = f >> DST_TILE_SHIFT;
                    let pos = tile_cur[t];
                    tile_cur[t] = pos + 1;
                    pairs[pos] = MicroPair {
                        local_p: lp as u16,
                        local_d: (f & DST_TILE_MASK) as u16,
                    };
                }
            }

            for t in 0..num_dst_tiles {
                let beg = tile_prev[t];
                let end = tile_cur[t];
                if beg < end {
                    tile_prev[t] = end;
                    tile_span_cnt[t] += 1;
                    tmp_spans.push((
                        t as u32,
                        MicroTileSpan {
                            src_base: s_base,
                            src_len: s_len,
                            pair_beg: beg,
                            pair_end: end,
                        },
                    ));
                }
            }
        }

        // Accumulate `raw_counts` tile-by-tile so `rc_tile` (16 KiB) stays 100% in L1d!
        for t in 0..num_dst_tiles {
            let p_beg = tile_pair_off[t];
            let p_end = tile_pair_off[t + 1];
            if p_beg < p_end {
                let d_base = t << DST_TILE_SHIFT;
                let d_end = (d_base + DST_TILE_LEN).min(frame_len);
                let rc_tile = &mut raw_counts[d_base..d_end];
                for pair in &pairs[p_beg..p_end] {
                    rc_tile[pair.local_d as usize] += 1;
                }
            }
        }

        // Bucket-sort `tmp_spans` by destination tile `t` into `spans` and `tile_span_offsets`.
        tile_span_offsets.reserve(num_dst_tiles + 1);
        let mut total_spans = 0_usize;
        tile_span_offsets.push(0);
        for &sc in &tile_span_cnt {
            total_spans += sc;
            tile_span_offsets.push(total_spans);
        }
        spans.resize(
            total_spans,
            MicroTileSpan {
                src_base: 0,
                src_len: 0,
                pair_beg: 0,
                pair_end: 0,
            },
        );
        let mut span_cur = tile_span_offsets[..num_dst_tiles].to_vec();
        for (t_u32, sp) in tmp_spans {
            let t = t_u32 as usize;
            let pos = span_cur[t];
            span_cur[t] = pos + 1;
            spans[pos] = sp;
        }
    }

    let all_unit_inv =
        !has_any_masked && (is_exact_identity || raw_counts.iter().all(|&rc| rc == 1));

    let mut count_data = Vec::with_capacity(frame_len);
    let mut inv_count = Vec::with_capacity(frame_len);
    if all_unit_inv {
        count_data.resize(frame_len, 1.0_f64);
        inv_count.resize(frame_len, 1.0_f64);
    } else {
        let mut unmasked_sum_u64: u64 = 0;
        let mut unmasked_cnt: usize = 0;
        for f in 0..frame_len {
            if !has_any_masked || mask_u8[f] == 0 {
                unmasked_sum_u64 += raw_counts[f] as u64;
                unmasked_cnt += 1;
            }
        }
        if unmasked_cnt > 0 {
            let mean = (unmasked_sum_u64 as f64) / (unmasked_cnt as f64);
            if mean != 0.0 {
                let inv_mean = 1.0 / mean;
                // Precompute small raw_count LUT (`0..16`) to avoid per-pixel `f64` divisions!
                let mut c_lut = [0.0_f64; 16];
                let mut inv_lut = [f64::NAN; 16];
                for r in 1..16 {
                    let c = (r as f64) * inv_mean;
                    c_lut[r] = c;
                    inv_lut[r] = mean / (r as f64);
                }
                if !has_any_masked {
                    for &rc in &raw_counts {
                        let r = rc as usize;
                        if r < 16 {
                            count_data.push(c_lut[r]);
                            inv_count.push(inv_lut[r]);
                        } else {
                            let c = (r as f64) * inv_mean;
                            count_data.push(c);
                            inv_count.push(1.0 / c);
                        }
                    }
                } else {
                    for f in 0..frame_len {
                        let r = raw_counts[f] as usize;
                        if mask_u8[f] == 0 {
                            if r < 16 {
                                count_data.push(c_lut[r]);
                                inv_count.push(inv_lut[r]);
                            } else {
                                let c = (r as f64) * inv_mean;
                                count_data.push(c);
                                inv_count.push(1.0 / c);
                            }
                        } else {
                            count_data.push(r as f64);
                            inv_count.push(f64::NAN);
                        }
                    }
                }
            } else {
                for f in 0..frame_len {
                    if !has_any_masked || mask_u8[f] == 0 {
                        count_data.push((raw_counts[f] as f64) / 0.0);
                    } else {
                        count_data.push(raw_counts[f] as f64);
                    }
                    inv_count.push(f64::NAN);
                }
            }
        } else {
            for &rc in &raw_counts {
                count_data.push(rc as f64);
                inv_count.push(f64::NAN);
            }
        }
    }

    Ok(CorePlanData {
        direction,
        width,
        height,
        frame_len,
        num_dst_tiles,
        mask_u8,
        mask_bool,
        has_any_masked,
        count_data,
        is_exact_identity,
        is_identity_like,
        all_unit_inv,
        flat_u32,
        tile_span_offsets,
        spans,
        pairs,
        inv_count,
    })
}

/// Single-threaded initialization for `DetectorRenderer` (`DetRender`).
#[pyfunction]
pub fn prepare_detector_renderer<'py>(
    py: Python<'py>,
    coor: PyReadonlyArray2<'py, f64>,
    det_mask: &Bound<'py, PyAny>,
    detd: f64,
    ewald_rad: f64,
) -> PyResult<PreparedRendererTuple<'py>> {
    if !coor.is_c_contiguous() {
        return Err(PyValueError::new_err("coor must be C-contiguous"));
    }
    let shape = coor.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err("coor must have shape (num_pix, 3)"));
    }
    let num_pix = shape[0];
    if num_pix == 0 {
        return Err(PyValueError::new_err("num_pix must be positive"));
    }
    let coor_s = coor
        .as_slice()
        .map_err(|_| PyValueError::new_err("coor must be C-contiguous"))?;

    // SAFETY: `cxy_out` and `xy_out` are completely overwritten in `build_renderer_core` before returning.
    let cxy_arr = unsafe { PyArray2::<f64>::new(py, [num_pix, 2], false) };
    let xy_arr = unsafe { PyArray2::<i32>::new(py, [num_pix, 2], false) };
    // SAFETY: freshly allocated arrays with exclusive access while holding the GIL.
    let cxy_out = unsafe { cxy_arr.as_slice_mut()? };
    let xy_out = unsafe { xy_arr.as_slice_mut()? };

    macro_rules! dispatch_mask {
        ($($t:ty),*) => {
            $(
                if let Ok(m_arr) = det_mask.extract::<PyReadonlyArray1<'py, $t>>() {
                    if m_arr.is_c_contiguous() {
                        if let Ok(m_s) = m_arr.as_slice() {
                            if m_s.len() != num_pix {
                                return Err(PyValueError::new_err(
                                    "det_mask length must match coor rows",
                                ));
                            }
                            let computed = py.detach(|| {
                                build_renderer_core(coor_s, m_s, detd, ewald_rad, cxy_out, xy_out)
                            });
                            return finish_prepared_renderer(py, num_pix, cxy_arr, xy_arr, computed);
                        }
                    }
                }
            )*
        };
    }

    dispatch_mask!(i32, u8, i64, i16, u16, u32, u64, i8);
    Err(PyValueError::new_err(
        "det_mask must be a C-contiguous 1D integer array",
    ))
}

fn finish_prepared_renderer<'py>(
    py: Python<'py>,
    num_pix: usize,
    cxy_arr: Bound<'py, PyArray2<f64>>,
    xy_arr: Bound<'py, PyArray2<i32>>,
    computed: Result<CorePlanData, &'static str>,
) -> PyResult<PreparedRendererTuple<'py>> {
    let data = computed.map_err(PyValueError::new_err)?;

    let mask_arr = PyArray1::from_vec(py, data.mask_u8).reshape([data.height, data.width])?;
    let count_arr = PyArray1::from_vec(py, data.count_data).reshape([data.height, data.width])?;
    let plan = Py::new(
        py,
        RustRendererPlan {
            width: data.width,
            height: data.height,
            frame_len: data.frame_len,
            num_pix,
            num_dst_tiles: data.num_dst_tiles,
            is_exact_identity: data.is_exact_identity,
            is_identity_like: data.is_identity_like,
            all_unit_inv: data.all_unit_inv,
            flat_u32: data.flat_u32,
            tile_span_offsets: data.tile_span_offsets,
            spans: data.spans,
            pairs: data.pairs,
            inv_count: data.inv_count,
            mask_bool: data.mask_bool,
            has_any_masked: data.has_any_masked,
        },
    )?;

    Ok((
        data.direction,
        cxy_arr,
        xy_arr,
        data.width,
        data.height,
        mask_arr,
        count_arr,
        plan,
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustRendererPlan>()?;
    m.add_function(wrap_pyfunction!(prepare_detector_renderer, m)?)?;
    Ok(())
}
