"""Parity, edge-case, caching, and parallel-threshold tests for EMC Rust acceleration."""

from __future__ import annotations

import importlib.util
import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import coo_array, coo_matrix, csc_array, csr_array, csr_matrix

import emcfile as ef
from emcfile._emc_accel import get_emc_rs
from emcfile._pattern_files import _PatternsSOneBytes

HAS_EMC_FAST = get_emc_rs() is not None


@contextmanager
def _emc_accel_mode(enabled: bool, workers: int | None = None) -> Iterator[None]:
    prev_disable = os.environ.get("EMCFILE_DISABLE_EMC_ACCEL")
    prev_workers = os.environ.get("EMCFILE_EMC_WORKERS")
    try:
        os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = "0" if enabled else "1"
        if workers is not None:
            os.environ["EMCFILE_EMC_WORKERS"] = str(workers)
        yield
    finally:
        if prev_disable is None:
            os.environ.pop("EMCFILE_DISABLE_EMC_ACCEL", None)
        else:
            os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = prev_disable
        if prev_workers is None:
            os.environ.pop("EMCFILE_EMC_WORKERS", None)
        else:
            os.environ["EMCFILE_EMC_WORKERS"] = prev_workers


def _assert_patterns_identical(a: ef.EMCPatternArray, b: ef.EMCPatternArray) -> None:
    assert a.shape == b.shape
    assert a.ones.dtype == np.uint32 and b.ones.dtype == np.uint32
    assert a.multi.dtype == np.uint32 and b.multi.dtype == np.uint32
    assert a.place_ones.dtype == np.uint32 and b.place_ones.dtype == np.uint32
    assert a.place_multi.dtype == np.uint32 and b.place_multi.dtype == np.uint32
    assert a.count_multi.dtype == np.int32 and b.count_multi.dtype == np.int32
    assert not a.ones_offsets.flags.writeable
    assert not a.multi_offsets.flags.writeable
    np.testing.assert_array_equal(a.ones, b.ones)
    np.testing.assert_array_equal(a.multi, b.multi)
    np.testing.assert_array_equal(a.ones_offsets, b.ones_offsets)
    np.testing.assert_array_equal(a.multi_offsets, b.multi_offsets)
    np.testing.assert_array_equal(a.place_ones, b.place_ones)
    np.testing.assert_array_equal(a.place_multi, b.place_multi)
    np.testing.assert_array_equal(a.count_multi, b.count_multi)


@pytest.fixture
def synthetic_dense() -> np.ndarray:
    rng = np.random.default_rng(2026)
    # Mix empty rows, 1-only rows, multi-only rows, and dense bursts across >64 columns
    arr = np.zeros((48, 197), dtype=np.int32)
    mask1 = rng.random(arr.shape) < 0.08
    maskm = rng.random(arr.shape) < 0.04
    arr[mask1] = 1
    arr[maskm] = rng.integers(2, 25, size=int(maskm.sum()), dtype=np.int32)
    arr[0, :] = 0
    arr[7, :] = 0
    arr[-1, :] = 0
    arr[3, ::5] = 1
    arr[5, ::7] = 9
    return arr


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_find_spec_monkeypatch_disables_emc_accel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orig = importlib.util.find_spec
    assert get_emc_rs() is not None
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            None if name == "emcfile_fast" else orig(name, *args, **kwargs)
        ),
    )
    assert get_emc_rs() is None


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
)
def test_dense_conversion_parity_all_integer_dtypes(
    synthetic_dense: np.ndarray, dtype: Any
) -> None:
    src = synthetic_dense.astype(dtype)
    if np.issubdtype(dtype, np.signedinteger):
        src = src.copy()
        src[1, :10] = -3  # Negative entries must be ignored identically

    with _emc_accel_mode(False):
        py_pat = ef.patterns(src)
        py_f_order = ef.patterns(np.asfortranarray(src))
        py_sliced = ef.patterns(src, start=4, end=19)
        py_1d = ef.patterns(src[3])
    with _emc_accel_mode(True, workers=4):
        rs_pat = ef.patterns(src)
        rs_f_order = ef.patterns(np.asfortranarray(src))
        rs_sliced = ef.patterns(src, start=4, end=19)
        rs_1d = ef.patterns(src[3])

    _assert_patterns_identical(rs_pat, py_pat)
    _assert_patterns_identical(rs_f_order, py_f_order)
    _assert_patterns_identical(rs_sliced, py_sliced)
    _assert_patterns_identical(rs_1d, py_1d)


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_dense_coo_csr_overflow_and_edge_cases() -> None:
    # Empty 0x0, 0xN, Nx0
    for shape in [(0, 0), (0, 70), (12, 0)]:
        z = np.zeros(shape, dtype=np.int32)
        with _emc_accel_mode(False):
            p_py = ef.patterns(z)
        with _emc_accel_mode(True):
            p_rs = ef.patterns(z)
        _assert_patterns_identical(p_rs, p_py)

    # Overflow > int32::MAX raises ValueError in both modes
    too_big_i64 = np.array([[0, 1, int(np.iinfo(np.int32).max) + 5]], dtype=np.int64)
    too_big_u64 = np.array([[0, 1, int(np.iinfo(np.int32).max) + 5]], dtype=np.uint64)
    for arr in (too_big_i64, too_big_u64):
        for mode in (False, True):
            with _emc_accel_mode(mode), pytest.raises(ValueError, match="int32"):
                ef.patterns(arr)

    # COO with unsorted coordinates, duplicates summing to <=0, 1, and >1
    rows = np.array([2, 0, 2, 1, 0, 1, 2], dtype=np.int32)
    cols = np.array([5, 3, 5, 4, 3, 4, 1], dtype=np.int32)
    data = np.array([2, 4, -1, 3, -4, 2, -5], dtype=np.int32)
    # (2,5)->1 (ones), (0,3)->0 (dropped), (1,4)->5 (multi), (2,1)->-5 (dropped)
    for cls in (coo_array, coo_matrix):
        sp_coo = cls((data, (rows, cols)), shape=(4, 10))
        with _emc_accel_mode(False):
            py_coo = ef.patterns(sp_coo)
        with _emc_accel_mode(True):
            rs_coo = ef.patterns(sp_coo)
        _assert_patterns_identical(rs_coo, py_coo)

    # CSR canonical, CSR non-canonical (unsorted/duplicate), and CSC
    dense_mix = np.array([[0, 1, 4, 0], [2, 0, -3, 1], [0, 0, 0, 0]], dtype=np.int32)
    for cls in (csr_array, csr_matrix, csc_array):
        sp = cls(dense_mix)
        with _emc_accel_mode(False):
            py_sp = ef.patterns(sp)
            py_sp_sub = ef.patterns(sp, start=1, end=3)
        with _emc_accel_mode(True):
            rs_sp = ef.patterns(sp)
            rs_sp_sub = ef.patterns(sp, start=1, end=3)
        _assert_patterns_identical(rs_sp, py_sp)
        _assert_patterns_identical(rs_sp_sub, py_sp_sub)


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_in_memory_reductions_todense_and_matmul_parity(
    synthetic_dense: np.ndarray,
) -> None:
    pat = ef.patterns(synthetic_dense)

    for axis in (None, 0, 1, -1, -2):
        for keepdims in (False, True):
            for dt in (None, np.int32, np.int64, np.float32, np.float64):
                with _emc_accel_mode(False):
                    exp = pat.sum(axis=axis, keepdims=keepdims, dtype=dt)
                with _emc_accel_mode(True, workers=4):
                    got = pat.sum(axis=axis, keepdims=keepdims, dtype=dt)
                assert np.asarray(got).dtype == np.asarray(exp).dtype
                assert np.asarray(got).shape == np.asarray(exp).shape
                if dt == np.float32:
                    np.testing.assert_allclose(got, exp, rtol=1e-6)
                else:
                    np.testing.assert_array_equal(got, exp)

    with _emc_accel_mode(False):
        d_py = pat.todense()
    with _emc_accel_mode(True, workers=4):
        d_rs = pat.todense()
    assert type(d_rs) is type(d_py)
    assert d_rs.dtype == d_py.dtype
    np.testing.assert_array_equal(d_rs, d_py)

    rng = np.random.default_rng(99)
    all_dtypes = [
        np.float64,
        np.float32,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.bool_,
    ]
    for dt in all_dtypes:
        if dt == np.bool_:
            vec = rng.integers(0, 2, size=pat.num_pixels).astype(np.bool_)
            mat = rng.integers(0, 2, size=(pat.num_pixels, 7)).astype(np.bool_)
        elif np.issubdtype(dt, np.floating):
            vec = rng.standard_normal(pat.num_pixels).astype(dt)
            mat = rng.standard_normal((pat.num_pixels, 7)).astype(dt)
        else:
            vec = rng.integers(0, 11, size=pat.num_pixels).astype(dt)
            mat = rng.integers(0, 11, size=(pat.num_pixels, 7)).astype(dt)

        for rhs in (vec, mat, np.asfortranarray(mat)):
            with _emc_accel_mode(False):
                exp_mm = pat @ rhs
            with _emc_accel_mode(True, workers=4):
                got_mm = pat @ rhs
            assert got_mm.dtype == exp_mm.dtype, f"dtype mismatch for {dt}"
            assert got_mm.shape == exp_mm.shape
            if np.issubdtype(got_mm.dtype, np.floating):
                np.testing.assert_allclose(got_mm, exp_mm, rtol=1e-6, atol=1e-6)
            else:
                np.testing.assert_array_equal(got_mm, exp_mm)


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_in_memory_row_and_column_indexing_parity(
    synthetic_dense: np.ndarray,
) -> None:
    pat = ef.patterns(synthetic_dense)
    rng = np.random.default_rng(123)

    row_selectors: list[Any] = [
        0,
        -1,
        slice(None),
        slice(5, 31),
        slice(3, 40, 3),
        slice(41, 4, -2),
        slice(None, None, -1),
        slice(20, 10, 1),  # empty slice
        rng.random(pat.num_patterns) < 0.35,
        np.zeros(pat.num_patterns, dtype=bool),
        np.array([10, 0, 10, -1, 5, 0], dtype=np.int64),
        np.array([], dtype=np.int32),
    ]
    for rsel in row_selectors:
        with _emc_accel_mode(False):
            exp = pat[rsel]
        with _emc_accel_mode(True, workers=4):
            got = pat[rsel]
        if isinstance(exp, np.ndarray):
            np.testing.assert_array_equal(got, exp)
        else:
            _assert_patterns_identical(got, exp)

    # Column selectors (including unsorted source indices and duplicate fancy columns)
    perm_cols = np.arange(pat.num_pixels)
    perm_cols[0], perm_cols[-1] = perm_cols[-1], perm_cols[0]
    unsorted_pat = pat[:, perm_cols]
    assert not unsorted_pat.has_sorted_indices()

    col_selectors: list[Any] = [
        0,
        -1,
        slice(None),
        slice(10, 140),
        slice(5, 190, 4),
        slice(180, 12, -3),
        slice(None, None, -1),
        slice(50, 20, 1),
        rng.random(pat.num_pixels) < 0.4,
        np.zeros(pat.num_pixels, dtype=bool),
        np.array([15, 2, 15, 0, -1, 2, 99], dtype=np.int64),
        np.array([], dtype=np.int32),
    ]
    for src_pat in (pat, unsorted_pat):
        for rsel in (slice(None), slice(2, 35, 2), np.array([7, 3, 7, 12])):
            for csel in col_selectors:
                if isinstance(rsel, np.ndarray) and isinstance(csel, np.ndarray):
                    # Paired fancy indexing is delegated to SciPy in both modes
                    continue
                with _emc_accel_mode(False):
                    exp2 = src_pat[rsel, csel]
                with _emc_accel_mode(True, workers=4):
                    got2 = src_pat[rsel, csel]
                if isinstance(exp2, np.ndarray) or np.isscalar(exp2):
                    np.testing.assert_array_equal(got2, exp2)
                else:
                    _assert_patterns_identical(got2, exp2)

    # Wide pixel count > int32::MAX column selection & single-column extraction
    wide_pix = (1 << 31) + 17
    wide_pat = ef.PatternsSOne(
        wide_pix,
        np.array([2, 1], dtype=np.uint32),
        np.array([1, 1], dtype=np.uint32),
        np.array([3, wide_pix - 2, wide_pix - 1], dtype=np.uint32),
        np.array([wide_pix - 1, 5], dtype=np.uint32),
        np.array([7, 4], dtype=np.int32),
    )
    wide_cols = np.array([wide_pix - 1, 3, wide_pix - 1, 5], dtype=np.int64)
    with _emc_accel_mode(False):
        # Fallback for >i32 uses Python path for single col
        exp_wcol = wide_pat[:, wide_pix - 1]
        exp_wslice = wide_pat[:, wide_pix - 5 :]
    with _emc_accel_mode(True):
        got_wcol = wide_pat[:, wide_pix - 1]
        got_wslice = wide_pat[:, wide_pix - 5 :]
        got_wfancy = wide_pat[:, wide_cols]
    np.testing.assert_array_equal(got_wcol, exp_wcol)
    _assert_patterns_identical(got_wslice, exp_wslice)
    np.testing.assert_array_equal(
        got_wfancy.todense(),
        np.array([[7, 1, 7, 0], [1, 0, 1, 4]], dtype=np.int32),
    )

    # uint64 max index must raise IndexError, never wrap to -1
    u64_oob = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(IndexError):
        _ = pat[u64_oob]
    with pytest.raises(IndexError):
        _ = pat[:, u64_oob]
    with pytest.raises(IndexError):
        _ = pat[u64_oob, 0]


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_sort_indices_and_has_sorted_indices_parity() -> None:
    # Unsorted with duplicate check + read-only buffers
    p_unsorted = ef.PatternsSOne(
        20,
        np.array([0, 3, 1], dtype=np.uint32),
        np.array([0, 3, 2], dtype=np.uint32),
        np.array([9, 2, 5, 4], dtype=np.uint32),
        np.array([12, 3, 8, 15, 1], dtype=np.uint32),
        np.array([20, 30, 40, 50, 60], dtype=np.int32),
    )
    p_py = ef.PatternsSOne(
        p_unsorted.num_pix,
        p_unsorted.ones.copy(),
        p_unsorted.multi.copy(),
        p_unsorted.place_ones.copy(),
        p_unsorted.place_multi.copy(),
        p_unsorted.count_multi.copy(),
    )
    p_rs = ef.PatternsSOne(
        p_unsorted.num_pix,
        p_unsorted.ones.copy(),
        p_unsorted.multi.copy(),
        p_unsorted.place_ones.copy(),
        p_unsorted.place_multi.copy(),
        p_unsorted.count_multi.copy(),
    )

    with _emc_accel_mode(False):
        assert p_py.has_sorted_indices() is False
        p_py.sort_indices()
        assert p_py.has_sorted_indices() is True
    with _emc_accel_mode(True):
        assert p_rs.has_sorted_indices() is False
        p_rs.sort_indices()
        assert p_rs.has_sorted_indices() is True

    _assert_patterns_identical(p_rs, p_py)

    # Read-only sorted buffer is a no-op; read-only unsorted buffer raises ValueError
    p_ro_sorted = ef.PatternsSOne(
        10,
        np.array([2], dtype=np.uint32),
        np.array([2], dtype=np.uint32),
        np.array([1, 5], dtype=np.uint32),
        np.array([2, 7], dtype=np.uint32),
        np.array([9, 3], dtype=np.int32),
    )
    p_ro_sorted.place_ones.flags.writeable = False
    p_ro_sorted.place_multi.flags.writeable = False
    p_ro_sorted.count_multi.flags.writeable = False
    p_ro_sorted.sort_indices()
    assert p_ro_sorted.has_sorted_indices() is True

    p_ro_unsorted = ef.PatternsSOne(
        10,
        np.array([2], dtype=np.uint32),
        np.array([2], dtype=np.uint32),
        np.array([5, 1], dtype=np.uint32),
        np.array([7, 2], dtype=np.uint32),
        np.array([3, 9], dtype=np.int32),
    )
    p_ro_unsorted.place_ones.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        p_ro_unsorted.sort_indices()

    # Duplicate positions in place_multi preserve stable count_multi ordering
    pm_ties = np.array([5] * 20 + [1, 5], dtype=np.uint32)
    cm_ties = np.arange(2, 24, dtype=np.int32)
    p_tie_py = ef.PatternsSOne(
        10,
        np.array([0], dtype=np.uint32),
        np.array([len(pm_ties)], dtype=np.uint32),
        np.empty(0, dtype=np.uint32),
        pm_ties.copy(),
        cm_ties.copy(),
    )
    p_tie_rs = ef.PatternsSOne(
        10,
        np.array([0], dtype=np.uint32),
        np.array([len(pm_ties)], dtype=np.uint32),
        np.empty(0, dtype=np.uint32),
        pm_ties.copy(),
        cm_ties.copy(),
    )
    with _emc_accel_mode(False):
        p_tie_py.sort_indices()
    with _emc_accel_mode(True):
        p_tie_rs.sort_indices()
    _assert_patterns_identical(p_tie_rs, p_tie_py)


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_binary_emc_file_and_bytesio_full_parity(
    tmp_path: Path, synthetic_dense: np.ndarray
) -> None:
    pat = ef.patterns(synthetic_dense)
    f_py = tmp_path / "py.emc"
    f_rs = tmp_path / "rs.emc"

    # Single write & multi-batch write_patterns
    with _emc_accel_mode(False):
        pat.write(f_py, overwrite=True)
    with _emc_accel_mode(True):
        pat.write(f_rs, overwrite=True)
    assert f_rs.read_bytes() == f_py.read_bytes()

    f_multi_py = tmp_path / "multi_py.emc"
    f_multi_rs = tmp_path / "multi_rs.emc"
    batches = [pat[:15], pat[15:33], pat[33:]]
    with _emc_accel_mode(False):
        ef.write_patterns(batches, f_multi_py, overwrite=True)
    with _emc_accel_mode(True):
        ef.write_patterns(batches, f_multi_rs, overwrite=True)
    assert f_multi_rs.read_bytes() == f_py.read_bytes()
    assert f_multi_py.read_bytes() == f_py.read_bytes()

    # BytesIO roundtrip AND verify BytesIO remains writable after reading (no lingered export lock)
    bio = io.BytesIO()
    with _emc_accel_mode(True):
        pat.write(bio)
        b_reader = _PatternsSOneBytes(bio)
        _assert_patterns_identical(b_reader[:], pat)
        _assert_patterns_identical(b_reader[3:30:4], pat[3:30:4])
        np.testing.assert_array_equal(b_reader[5], pat[5])
        np.testing.assert_array_equal(b_reader[:, 12], pat[:, 12])
        assert b_reader.sum() == pat.sum()
        np.testing.assert_array_equal(b_reader.sum(axis=0), pat.sum(axis=0))
        np.testing.assert_array_equal(b_reader.sum(axis=1), pat.sum(axis=1))
        sp_b = b_reader.sparse_pattern(4)
    sp_ref = pat.sparse_pattern(4)
    np.testing.assert_array_equal(sp_b.place_ones, sp_ref.place_ones)
    np.testing.assert_array_equal(sp_b.place_multi, sp_ref.place_multi)
    np.testing.assert_array_equal(sp_b.count_multi, sp_ref.count_multi)
    # Resizing bio must succeed (BufferError would happen if getbuffer() were leaked)
    bio.seek(0, io.SEEK_END)
    bio.write(b"extra")

    # Compare PatternsSOneEMC and PatternsSOneEMCReadBuffer across all indexing & reductions
    rng = np.random.default_rng(55)
    mask = rng.random(pat.num_patterns) < 0.4
    fancy = np.array([12, 0, 12, -2, 7], dtype=np.int64)

    with _emc_accel_mode(False):
        src_py = ef.open_patterns(f_rs)
        exp_full = src_py[:]
        exp_slice = src_py[6:29]
        exp_stride = src_py[40:3:-3]
        exp_mask = src_py[mask]
        exp_fancy = src_py[fancy]
        exp_frame = src_py[-3]
        exp_col_all = src_py[:, 15]
        exp_col_sub = src_py[fancy, -4]
        exp_col_mask = src_py[mask, 0]
        exp_2d = src_py[4:20:2, 10:80:3]
        exp_sum_none = src_py.sum()
        exp_sum_0 = src_py.sum(axis=0, keepdims=True, dtype=np.float64)
        exp_sum_1 = src_py.sum(axis=1, chunk_size=7)

    with _emc_accel_mode(True, workers=4):
        src_rs = ef.open_patterns(f_rs)
        with src_rs.open() as rb_rs:
            for reader in (src_rs, rb_rs):
                _assert_patterns_identical(reader[:], exp_full)
                _assert_patterns_identical(reader[6:29], exp_slice)
                _assert_patterns_identical(reader[40:3:-3], exp_stride)
                _assert_patterns_identical(reader[mask], exp_mask)
                _assert_patterns_identical(reader[fancy], exp_fancy)
                np.testing.assert_array_equal(reader[-3], exp_frame)
                np.testing.assert_array_equal(reader[:, 15], exp_col_all)
                np.testing.assert_array_equal(reader[fancy, -4], exp_col_sub)
                np.testing.assert_array_equal(reader[mask, 0], exp_col_mask)
                _assert_patterns_identical(reader[4:20:2, 10:80:3], exp_2d)
                assert reader.sum() == exp_sum_none
                np.testing.assert_array_equal(
                    reader.sum(axis=0, keepdims=True, dtype=np.float64), exp_sum_0
                )
                np.testing.assert_array_equal(
                    reader.sum(axis=1, chunk_size=7), exp_sum_1
                )
                np.testing.assert_array_equal(reader.place_ones, pat.place_ones)
                np.testing.assert_array_equal(reader.place_multi, pat.place_multi)
                np.testing.assert_array_equal(reader.count_multi, pat.count_multi)

        # Closed reader must raise ValueError on subsequent I/O
        with pytest.raises(ValueError, match="closed"):
            _ = rb_rs[0]
        with pytest.raises(ValueError, match="closed"):
            _ = rb_rs.sparse_pattern(0)
        with pytest.raises(ValueError, match="closed"):
            _ = rb_rs.sum()
        with pytest.raises(ValueError, match="closed"):
            _ = rb_rs.place_ones


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_emc_atomic_overwrite_cache_invalidation_and_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "mutable.emc"
    p1 = ef.patterns(np.array([[0, 1, 3, 0], [2, 0, 1, 0]], dtype=np.int32))
    p2 = ef.patterns(
        np.array([[5, 0, 0, 1], [0, 4, 1, 2], [1, 1, 0, 0]], dtype=np.int32)
    )

    p1.write(path, overwrite=True)
    s1 = ef.open_patterns(path)
    with s1.open() as pinned_reader:
        assert pinned_reader.sum() == p1.sum()
        # Overwrite path atomically while pinned_reader holds old inode open
        p2.write(path, overwrite=True)
        assert pinned_reader.shape == p1.shape
        _assert_patterns_identical(pinned_reader[:], p1)
        # Newly opened reader must see p2, never stale p1 cache
        s2 = ef.open_patterns(path)
        assert s2.shape == p2.shape
        _assert_patterns_identical(s2[:], p2)

    # Empty 0-pattern .emc file roundtrip and sum
    empty_path = tmp_path / "empty.emc"
    p_empty = ef.patterns(16)
    p_empty.write(empty_path, overwrite=True)
    s_empty = ef.open_patterns(empty_path)
    assert s_empty.shape == (0, 16)
    _assert_patterns_identical(s_empty[:], p_empty)
    assert s_empty.sum() == 0
    np.testing.assert_array_equal(s_empty.sum(axis=0), np.zeros(16, dtype=np.int64))
    np.testing.assert_array_equal(s_empty.sum(axis=1), np.zeros(0, dtype=np.int64))

    # Trailing byte corruption must raise ValueError("Error when parsing")
    trailing = tmp_path / "trailing.emc"
    trailing.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="Error when parsing"):
        _ = ef.open_patterns(trailing)[:]
    with pytest.raises(ValueError, match="Error when parsing"):
        _ = ef.open_patterns(trailing)[0]
    with pytest.raises(ValueError, match="Error when parsing"):
        _ = ef.open_patterns(trailing).sum()


@pytest.mark.skipif(
    not HAS_EMC_FAST or not hasattr(os, "fork"),
    reason="requires emcfile_fast and os.fork",
)
def test_emc_mmap_cache_fork_safety(tmp_path: Path) -> None:
    path = tmp_path / "fork.emc"
    p = ef.patterns(np.array([[1, 0, 4, 2], [0, 1, 0, 3]], dtype=np.int32))
    p.write(path, overwrite=True)
    # Populate parent mmap cache before fork
    assert ef.open_patterns(path).sum() == 11

    pid = os.fork()
    if pid == 0:
        try:
            ok = ef.open_patterns(path).sum() == 11 and (ef.patterns(path) == p)
            os._exit(0 if ok else 2)
        except BaseException:
            os._exit(3)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


@pytest.mark.skipif(not HAS_EMC_FAST, reason="emcfile_fast.emc not installed")
def test_large_parallel_thresholds_parity(tmp_path: Path) -> None:
    """Exercise multi-threaded Rayon paths above size thresholds."""
    rng = np.random.default_rng(777)
    # 1200 rows x 600 cols = 720,000 elements (>262,144 dense parallel threshold, >512 sort rows)
    dense = np.zeros((1200, 600), dtype=np.int32)
    ones_pos = rng.integers(0, 600, size=(1200, 60))
    multi_pos = rng.integers(0, 600, size=(1200, 25))
    rows_grid = np.arange(1200)[:, None]
    dense[rows_grid, ones_pos] = 1
    dense[rows_grid, multi_pos] = rng.integers(2, 15, size=(1200, 25), dtype=np.int32)

    with _emc_accel_mode(True, workers=1):
        p_serial = ef.patterns(dense)
    with _emc_accel_mode(True, workers=4):
        p_par = ef.patterns(dense)
    _assert_patterns_identical(p_par, p_serial)

    # Permute columns to trigger parallel sort_indices (>512 rows & >65,536 events)
    rev_cols = np.arange(599, -1, -1, dtype=np.int64)
    with _emc_accel_mode(True, workers=4):
        p_rev = p_par[:, rev_cols]
        assert p_rev.has_sorted_indices() is False
        p_rev.sort_indices()
        assert p_rev.has_sorted_indices() is True
    np.testing.assert_array_equal(p_rev.todense(), dense[:, ::-1])

    # Parallel SpMM & file-backed parallel gather
    rhs = rng.standard_normal((600, 8))
    with _emc_accel_mode(False):
        exp_mm = p_par @ rhs
    with _emc_accel_mode(True, workers=4):
        got_mm = p_par @ rhs
    np.testing.assert_allclose(got_mm, exp_mm, rtol=1e-10, atol=1e-10)

    emc_file = tmp_path / "large.emc"
    p_par.write(emc_file, overwrite=True)
    src = ef.open_patterns(emc_file)
    ids = rng.integers(0, 1200, size=600, dtype=np.int64)
    with _emc_accel_mode(False):
        exp_gather = src[ids]
    with _emc_accel_mode(True, workers=4):
        got_gather = src[ids]
    _assert_patterns_identical(got_gather, exp_gather)
