use numpy::{
    Element, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type RenderedImage<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<bool>>);
type RenderedStack<'py> = (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<bool>>);

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
    ) -> PyResult<Option<RenderedImage<'py>>> {
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
    ) -> PyResult<Option<RenderedStack<'py>>> {
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
    ) -> PyResult<Option<RenderedImage<'py>>> {
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
    ) -> PyResult<Option<RenderedStack<'py>>> {
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
    ) -> PyResult<Option<RenderedImage<'py>>> {
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

        py.detach(|| self.accumulate_sparse_zeroed(ones_s, multi_s, count_s, out_slice))
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
    ) -> PyResult<Option<RenderedStack<'py>>> {
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

#[inline]
fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut pivot_row = col;
        let mut max_abs = a[col][col].abs();
        for (row, values) in a.iter().enumerate().skip(col + 1) {
            let v = values[col].abs();
            if v > max_abs {
                max_abs = v;
                pivot_row = row;
            }
        }
        if !max_abs.is_finite() || max_abs <= f64::MIN_POSITIVE {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        let inv_piv = 1.0 / a[col][col];
        for row in (col + 1)..3 {
            let factor = a[row][col] * inv_piv;
            let pivot = a[col];
            for (value, pivot_value) in a[row].iter_mut().zip(pivot).skip(col) {
                *value -= factor * pivot_value;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0_f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for j in (i + 1)..3 {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
        if !x[i].is_finite() {
            return None;
        }
    }
    Some(x)
}

#[inline]
fn radius_std_strided(coor: &[f64], n: usize, step: usize, r: [f64; 3]) -> f64 {
    let mut cnt = 0_usize;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut i = 0_usize;
    while i < n {
        let dx = coor[3 * i] - r[0];
        let dy = coor[3 * i + 1] - r[1];
        let dz = coor[3 * i + 2] - r[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        cnt += 1;
        let delta = d - mean;
        mean += delta / (cnt as f64);
        let delta2 = d - mean;
        m2 += delta * delta2;
        i += step;
    }
    if cnt == 0 {
        return 0.0;
    }
    (m2 / (cnt as f64)).max(0.0).sqrt()
}

fn kasa_sphere_init(coor: &[f64], n: usize, step: usize) -> Option<[f64; 3]> {
    let mut mx = 0.0_f64;
    let mut my = 0.0_f64;
    let mut mz = 0.0_f64;
    let mut cnt = 0_usize;
    let mut i = 0_usize;
    while i < n {
        mx += coor[3 * i];
        my += coor[3 * i + 1];
        mz += coor[3 * i + 2];
        cnt += 1;
        i += step;
    }
    if cnt < 4 {
        return None;
    }
    let inv_n = 1.0 / (cnt as f64);
    mx *= inv_n;
    my *= inv_n;
    mz *= inv_n;

    let mut q_mean = 0.0_f64;
    i = 0;
    while i < n {
        let ux = coor[3 * i] - mx;
        let uy = coor[3 * i + 1] - my;
        let uz = coor[3 * i + 2] - mz;
        q_mean += ux * ux + uy * uy + uz * uz;
        i += step;
    }
    q_mean *= inv_n;

    let mut a = [[0.0_f64; 3]; 3];
    let mut b = [0.0_f64; 3];
    i = 0;
    while i < n {
        let ux = coor[3 * i] - mx;
        let uy = coor[3 * i + 1] - my;
        let uz = coor[3 * i + 2] - mz;
        let q = (ux * ux + uy * uy + uz * uz) - q_mean;
        let u = [ux, uy, uz];
        for r_idx in 0..3 {
            b[r_idx] += 0.5 * q * u[r_idx];
            for c_idx in 0..3 {
                a[r_idx][c_idx] += u[r_idx] * u[c_idx];
            }
        }
        i += step;
    }
    let dc = solve_3x3(a, b)?;
    Some([mx + dc[0], my + dc[1], mz + dc[2]])
}

fn nelder_mead_sphere(
    coor: &[f64],
    n: usize,
    step: usize,
    x0: [f64; 3],
    xatol: f64,
    max_iter: usize,
) -> ([f64; 3], f64) {
    let mut s = [x0; 4];
    for k in 0..3 {
        let h = if x0[k] != 0.0 { 0.05 * x0[k] } else { 0.00025 };
        s[k + 1][k] += h;
    }
    let mut f = [
        radius_std_strided(coor, n, step, s[0]),
        radius_std_strided(coor, n, step, s[1]),
        radius_std_strided(coor, n, step, s[2]),
        radius_std_strided(coor, n, step, s[3]),
    ];

    for _ in 0..max_iter {
        // Sort 4 vertices by objective value
        for i in 0..3 {
            for j in (i + 1)..4 {
                if f[j] < f[i] {
                    f.swap(i, j);
                    s.swap(i, j);
                }
            }
        }
        let mut max_dx = 0.0_f64;
        for i in 1..4 {
            for (value, best_value) in s[i].iter().zip(s[0]) {
                let d = (value - best_value).abs();
                if d > max_dx {
                    max_dx = d;
                }
            }
        }
        if max_dx <= xatol {
            break;
        }

        let cbar = [
            (s[0][0] + s[1][0] + s[2][0]) / 3.0,
            (s[0][1] + s[1][1] + s[2][1]) / 3.0,
            (s[0][2] + s[1][2] + s[2][2]) / 3.0,
        ];
        let xr = [
            2.0 * cbar[0] - s[3][0],
            2.0 * cbar[1] - s[3][1],
            2.0 * cbar[2] - s[3][2],
        ];
        let fr = radius_std_strided(coor, n, step, xr);

        if fr < f[0] {
            let xe = [
                3.0 * cbar[0] - 2.0 * s[3][0],
                3.0 * cbar[1] - 2.0 * s[3][1],
                3.0 * cbar[2] - 2.0 * s[3][2],
            ];
            let fe = radius_std_strided(coor, n, step, xe);
            if fe < fr {
                s[3] = xe;
                f[3] = fe;
            } else {
                s[3] = xr;
                f[3] = fr;
            }
        } else if fr < f[2] {
            s[3] = xr;
            f[3] = fr;
        } else {
            let xc = if fr < f[3] {
                [
                    0.5 * (cbar[0] + xr[0]),
                    0.5 * (cbar[1] + xr[1]),
                    0.5 * (cbar[2] + xr[2]),
                ]
            } else {
                [
                    0.5 * (cbar[0] + s[3][0]),
                    0.5 * (cbar[1] + s[3][1]),
                    0.5 * (cbar[2] + s[3][2]),
                ]
            };
            let fc = radius_std_strided(coor, n, step, xc);
            if fc < fr.min(f[3]) {
                s[3] = xc;
                f[3] = fc;
            } else {
                for i in 1..4 {
                    let best = s[0];
                    for (value, best_value) in s[i].iter_mut().zip(best) {
                        *value = 0.5 * (best_value + *value);
                    }
                    f[i] = radius_std_strided(coor, n, step, s[i]);
                }
            }
        }
    }
    // The last iteration may have improved a vertex after the sort.
    let best = (0..4).min_by(|&i, &j| f[i].total_cmp(&f[j])).unwrap();
    (s[best], f[best])
}

/// Gauss-Newton Gauss-Newton / Levenberg-Marquardt refinement minimizing `std(||x - r||)`.
fn gauss_newton_sphere(coor: &[f64], n: usize, step: usize, mut r: [f64; 3]) -> ([f64; 3], f64) {
    let mut lambda = 1e-12_f64;
    let mut cur_std = radius_std_strided(coor, n, step, r);

    for _ in 0..25 {
        let mut cnt = 0_usize;
        let mut sum_d = 0.0_f64;
        let mut sum_u = [0.0_f64; 3];
        let mut sum_du = [0.0_f64; 3];
        let mut sum_uu = [[0.0_f64; 3]; 3];

        let mut i = 0_usize;
        while i < n {
            let dx = coor[3 * i] - r[0];
            let dy = coor[3 * i + 1] - r[1];
            let dz = coor[3 * i + 2] - r[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            if d > 1e-15 {
                let inv_d = 1.0 / d;
                let u = [dx * inv_d, dy * inv_d, dz * inv_d];
                cnt += 1;
                sum_d += d;
                sum_du[0] += dx;
                sum_du[1] += dy;
                sum_du[2] += dz;
                for a in 0..3 {
                    sum_u[a] += u[a];
                    for b in a..3 {
                        sum_uu[a][b] += u[a] * u[b];
                    }
                }
            }
            i += step;
        }
        if cnt < 4 {
            break;
        }
        let inv_n = 1.0 / (cnt as f64);
        let d_mean = sum_d * inv_n;
        let u_mean = [sum_u[0] * inv_n, sum_u[1] * inv_n, sum_u[2] * inv_n];

        let mut h = [[0.0_f64; 3]; 3];
        let mut g = [0.0_f64; 3];
        for a in 0..3 {
            g[a] = sum_du[a] - (cnt as f64) * d_mean * u_mean[a];
            for b in a..3 {
                let val = sum_uu[a][b] - (cnt as f64) * u_mean[a] * u_mean[b];
                h[a][b] = val;
                h[b][a] = val;
            }
            h[a][a] += lambda * h[a][a].abs().max(1e-12);
        }

        if cur_std <= 1e-12 * d_mean.abs().max(1.0) {
            break;
        }

        let Some(delta) = solve_3x3(h, g) else {
            lambda *= 10.0;
            continue;
        };
        let step_norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        let r_norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let cand = [r[0] + delta[0], r[1] + delta[1], r[2] + delta[2]];
        let cand_std = radius_std_strided(coor, n, step, cand);
        if cand_std <= cur_std {
            r = cand;
            cur_std = cand_std;
            lambda = (lambda * 0.25).max(1e-15);
            if step_norm <= 1e-9 * (1.0 + r_norm) {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    (r, cur_std)
}

fn fit_ewald_sphere_center_core(coor: &[f64], n: usize) -> Result<[f64; 4], &'static str> {
    if coor.iter().any(|value| !value.is_finite()) {
        return Err("coordinates must contain only finite values");
    }
    if n < 3 {
        return Err("The input coor is degenerated to a line.");
    }

    let mut found_ca = 0_usize;
    let mut n0 = [0.0_f64; 3];
    let mut found = false;

    'outer: for ca in 0..n {
        let ax = coor[3 * ca];
        let ay = coor[3 * ca + 1];
        let az = coor[3 * ca + 2];
        for cb in (ca + 1)..n {
            let dbx = coor[3 * cb] - ax;
            let dby = coor[3 * cb + 1] - ay;
            let dbz = coor[3 * cb + 2] - az;
            let nb = (dbx * dbx + dby * dby + dbz * dbz).sqrt();
            if !nb.is_finite() || nb <= 0.0 {
                continue;
            }
            let inv_nb = 1.0 / nb;
            let ubx = dbx * inv_nb;
            let uby = dby * inv_nb;
            let ubz = dbz * inv_nb;

            for cc in (cb + 1)..n {
                let dcx = coor[3 * cc] - ax;
                let dcy = coor[3 * cc + 1] - ay;
                let dcz = coor[3 * cc + 2] - az;
                let nc = (dcx * dcx + dcy * dcy + dcz * dcz).sqrt();
                if !nc.is_finite() || nc <= 0.0 {
                    continue;
                }
                let inv_nc = 1.0 / nc;
                let ucx = dcx * inv_nc;
                let ucy = dcy * inv_nc;
                let ucz = dcz * inv_nc;

                let cx = uby * ucz - ubz * ucy;
                let cy = ubz * ucx - ubx * ucz;
                let cz = ubx * ucy - uby * ucx;
                let n_norm_sq = cx * cx + cy * cy + cz * cz;
                if n_norm_sq > 1e-2 {
                    found_ca = ca;
                    n0 = [cx, cy, cz];
                    found = true;
                    break 'outer;
                }
            }
        }
    }

    if !found {
        return Err("The input coor is degenerated to a line.");
    }

    // Check 2D coplanarity: `np.all(np.abs(np.dot(coor_shift, n0)) < 1e-3)`
    let ax = coor[3 * found_ca];
    let ay = coor[3 * found_ca + 1];
    let az = coor[3 * found_ca + 2];
    let mut is_2d = true;
    for i in 0..n {
        if i == found_ca {
            continue;
        }
        let dx = coor[3 * i] - ax;
        let dy = coor[3 * i + 1] - ay;
        let dz = coor[3 * i + 2] - az;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if !len.is_finite() || len <= 0.0 {
            is_2d = false;
            break;
        }
        let dot = (dx * n0[0] + dy * n0[1] + dz * n0[2]) / len;
        if !dot.is_finite() || dot.abs() >= 1e-3 {
            is_2d = false;
            break;
        }
    }
    if is_2d {
        return Ok([n0[0], n0[1], n0[2], 0.0]);
    }

    // Keep the Python path's sampling and planar thresholds. Optimize the
    // sample first so only the final refinement scans the full detector.
    let sub_step = if n < 32 { 1 } else { n / 32 };
    let (mut best_r, mut best_std) =
        nelder_mead_sphere(coor, n, sub_step, [0.0, 0.0, -1.0], 1e-8, 600);

    // An algebraic sphere fit provides a second starting point when the
    // simplex from [0, 0, -1] stalls. Singular samples use the simplex alone.
    if let Some(r_kasa) = kasa_sphere_init(coor, n, sub_step) {
        let (r_gn, std_gn) = gauss_newton_sphere(coor, n, sub_step, r_kasa);
        if std_gn < best_std {
            best_r = r_gn;
            best_std = std_gn;
        }
    }

    let (r_gn_sub, std_gn_sub) = gauss_newton_sphere(coor, n, sub_step, best_r);
    if std_gn_sub < best_std {
        best_r = r_gn_sub;
    }

    let (mut final_r, final_std) = gauss_newton_sphere(coor, n, 1, best_r);
    if n <= 1024 {
        let (r_nm, std_nm) = nelder_mead_sphere(coor, n, 1, final_r, 1e-8, 400);
        if std_nm < final_std {
            final_r = r_nm;
        }
    }

    Ok([final_r[0], final_r[1], final_r[2], 1.0])
}

/// Single-threaded Rust implementation of `fit_ewald_sphere_center`.
#[pyfunction]
pub fn fit_ewald_sphere_center_rs<'py>(
    py: Python<'py>,
    coordinates: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if !coordinates.is_c_contiguous() {
        return Err(PyValueError::new_err("coordinates must be C-contiguous"));
    }
    let shape = coordinates.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(
            "coordinates must have shape (num_points, 3)",
        ));
    }
    let n = shape[0];
    let coor_s = coordinates
        .as_slice()
        .map_err(|_| PyValueError::new_err("coordinates must be C-contiguous"))?;

    let vec4 = py
        .detach(|| fit_ewald_sphere_center_core(coor_s, n))
        .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_slice(py, &vec4))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustRendererPlan>()?;
    m.add_function(wrap_pyfunction!(prepare_detector_renderer, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ewald_sphere_center_rs, m)?)?;
    Ok(())
}
