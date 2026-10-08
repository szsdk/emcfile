from pathlib import Path

import h5py
import numpy as np
import pytest

import emcfile as ef
import emcfile._h5_indexed as indexed
from emcfile._h5_full_scan import _decode_chunk
from tests.test_hdf5_v2_codecs import _patterns


@pytest.fixture(autouse=True)
def enable_indexed(monkeypatch):
    monkeypatch.setenv("EMCFILE_H5_INDEXED_WORKERS", "4")


@pytest.mark.usefixtures("hdf5_fast")
def test_multi_frame_indexed_is_opt_in(tmp_path, monkeypatch):
    expected, path = _patterns(), tmp_path / "opt-in.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    source = ef.open_patterns(path)
    source.init_idx()
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)
    with h5py.File(path) as group:
        assert (
            indexed.indexed_read(
                group, np.array([[0, 2]]), source.ones_idx, source.multi_idx
            )
            is None
        )


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.usefixtures("hdf5_fast")
def test_indexed_public_api(tmp_path: Path, monkeypatch, workers, persistent):
    monkeypatch.setenv("EMCFILE_H5_DIRECT_CHUNK_BYTES", "20")
    monkeypatch.setenv("EMCFILE_H5_INDEXED_WORKERS", str(workers))
    expected = _patterns(7)
    path = tmp_path / "indexed.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    calls = []
    original = indexed.indexed_read

    def recorded(*args):
        result = original(*args)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(indexed, "indexed_read", recorded)
    source = ef.open_patterns(path)
    if persistent:
        source = source.open()
    try:
        for selection in (
            np.array([27, 2, 2, 0, 7, 1]),
            np.array([], dtype=int),
            np.arange(1, 28, 3),
            slice(2, 23),
            slice(0, 28, 3),
            np.arange(28) % 3 == 0,
        ):
            assert source[selection] == expected[selection]
        np.testing.assert_array_equal(source[3], expected[3])
    finally:
        if persistent:
            source.close()
    assert all(calls) and len(calls) == 7


@pytest.mark.parametrize("codec", ["zstd", "gzip", "lzf", None])
def test_fallbacks(tmp_path, monkeypatch, codec):
    if codec == "zstd":
        pytest.importorskip("hdf5plugin")
    expected = _patterns(3)
    path = tmp_path / "fallback.h5"
    expected.write(path, position_encoding="delta", compression=codec, shuffle=True)
    ids = np.array([5, 2, 2, 0])
    monkeypatch.setenv("EMCFILE_H5_INDEXED_WORKERS", "0")
    assert ef.open_patterns(path)[ids] == expected[ids]
    monkeypatch.setenv("EMCFILE_H5_INDEXED_WORKERS", "2")
    assert ef.open_patterns(path)[ids] == expected[ids]
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)
    np.testing.assert_array_equal(ef.open_patterns(path)[2], expected[2])


def test_plan_unique_chunks_and_destination_order():
    offsets = np.array([0, 3, 3, 8, 9], "u8")
    chunks, boundaries, size = indexed._plan(
        np.array([[2, 4], [0, 1], [2, 3]]), offsets, 4
    )
    assert sorted(chunks) == [0, 4, 8]
    np.testing.assert_array_equal(boundaries, [0, 5, 6, 9, 14])
    source = np.arange(9)
    output = np.empty(size, dtype=int)
    for start, spans in chunks.items():
        for destination, local, length in spans:
            output[destination : destination + length] = source[
                start + local : start + local + length
            ]
    np.testing.assert_array_equal(output, np.r_[source[3:9], source[:3], source[3:8]])


@pytest.mark.parametrize("mask", [0, 1, 2, 3])
@pytest.mark.usefixtures("hdf5_fast")
def test_chunk_filter_masks(mask):
    import zstandard

    values = np.array([0, 1, 65536, 2**32 - 1], "u4")
    raw = (
        values.tobytes()
        if mask & 1
        else values.view("u1").reshape(-1, 4).T.copy().tobytes()
    )
    payload = raw if mask & 2 else zstandard.ZstdCompressor().compress(raw)
    np.testing.assert_array_equal(_decode_chunk(mask, payload, 4), values)
    for selection in (slice(1, 3), slice(0, 1), slice(3, 4), slice(0, 0)):
        actual = _decode_chunk(mask, payload, 4, selection=selection)
        np.testing.assert_array_equal(actual, values[selection])
        assert actual.dtype == np.dtype("u4")
        assert actual.flags.writeable


def test_chunk_errors():
    pytest.importorskip("zstandard")
    with pytest.raises(ValueError, match="filter mask"):
        _decode_chunk(4, b"", 4)
    with pytest.raises(ValueError, match="length"):
        _decode_chunk(3, b"bad", 4)


@pytest.mark.usefixtures("hdf5_fast")
def test_missing_optional_dependency_and_invalid_budget(tmp_path, monkeypatch):
    expected, path = _patterns(), tmp_path / "optional.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    monkeypatch.setattr(indexed.importlib.util, "find_spec", lambda name: None)
    assert ef.open_patterns(path)[np.array([2, 0])] == expected[np.array([2, 0])]
    monkeypatch.setenv("EMCFILE_H5_INDEXED_WORKERS", "bad")
    with pytest.raises(ValueError, match="integer"):
        ef.open_patterns(path)[np.array([2, 0])]


@pytest.mark.usefixtures("hdf5_fast")
def test_worker_failure_propagates(tmp_path, monkeypatch):
    expected, path = _patterns(), tmp_path / "error.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)

    def fail(*args, **kwargs):
        raise RuntimeError("worker decode failure")

    monkeypatch.setattr(indexed, "_decode_chunk", fail)
    with pytest.raises(RuntimeError, match="worker decode failure"):
        ef.open_patterns(path)[np.array([2, 0])]


@pytest.mark.usefixtures("hdf5_fast")
def test_out_of_bounds_ranges(tmp_path):
    expected, path = _patterns(), tmp_path / "bounds.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    source = ef.open_patterns(path)
    source.init_idx()
    with h5py.File(path) as group:
        with pytest.raises(IndexError):
            indexed.indexed_read(
                group, np.array([[0, 5]]), source.ones_idx, source.multi_idx
            )


@pytest.mark.usefixtures("hdf5_fast")
def test_vds_and_absolute_layout_fallback(tmp_path):
    expected = _patterns()
    first, second, virtual = (tmp_path / name for name in ("a.h5", "b.h5", "vds.h5"))
    expected[:2].write(
        first, position_encoding="delta", compression="zstd", shuffle=True
    )
    expected[2:].write(
        second, position_encoding="delta", compression="zstd", shuffle=True
    )
    ef.create_vds(virtual, [first, second])
    ids = np.array([3, 1, 1])
    assert ef.open_patterns(virtual)[ids] == expected[ids]
    absolute = tmp_path / "absolute.h5"
    expected.write(absolute, compression="zstd", shuffle=True)
    assert ef.open_patterns(absolute)[ids] == expected[ids]
    np.testing.assert_array_equal(ef.open_patterns(virtual)[1], expected[1])
    np.testing.assert_array_equal(ef.open_patterns(absolute)[1], expected[1])


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("chunk_bytes", [4, 20, 16 << 20])
@pytest.mark.usefixtures("hdf5_fast")
def test_single_frame_default_without_workers_or_numba(
    tmp_path, monkeypatch, persistent, chunk_bytes
):
    monkeypatch.setenv("EMCFILE_H5_DIRECT_CHUNK_BYTES", str(chunk_bytes))
    expected, path = _patterns(3), tmp_path / "single.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)

    def unexpected(*args, **kwargs):
        pytest.fail("single frames must not use workers, JIT, or the generic reader")

    original_find_spec = indexed.importlib.util.find_spec
    monkeypatch.setattr(
        indexed.importlib.util,
        "find_spec",
        lambda name: None if name == "numba" else original_find_spec(name),
    )
    monkeypatch.setattr(indexed, "ThreadPoolExecutor", unexpected)
    monkeypatch.setattr(indexed, "_unshuffle_u32", unexpected)
    monkeypatch.setattr(indexed, "decode_segmented_delta_inplace", unexpected)
    monkeypatch.setattr("emcfile._pattern_files.read_indexed_array_h5", unexpected)
    source = ef.open_patterns(path)
    if persistent:
        source = source.open()
    try:
        for frame in range(expected.num_data):
            np.testing.assert_array_equal(source[frame], expected[frame])
            np.testing.assert_array_equal(source[np.int64(frame)], expected[frame])
            assert source[frame : frame + 1] == expected[frame : frame + 1]
            assert source[np.array([frame])] == expected[frame : frame + 1]
            actual_sparse = source.sparse_pattern(frame)
            expected_sparse = expected.sparse_pattern(frame)
            for name in ("place_ones", "place_multi", "count_multi"):
                np.testing.assert_array_equal(
                    getattr(actual_sparse, name), getattr(expected_sparse, name)
                )
    finally:
        if persistent:
            source.close()


@pytest.mark.parametrize("disabled", ["budget", "dependency"])
@pytest.mark.usefixtures("hdf5_fast")
def test_single_frame_generic_fallback(tmp_path, monkeypatch, disabled):
    expected, path = _patterns(), tmp_path / "fallback-single.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)
    if disabled == "budget":
        monkeypatch.setenv("EMCFILE_H5_INDEXED_WORKERS", "0")
    else:
        monkeypatch.setattr(indexed.importlib.util, "find_spec", lambda name: None)

    def unexpected(*args, **kwargs):
        pytest.fail("the direct reader must be disabled")

    monkeypatch.setattr(indexed, "_single_frame", unexpected)
    np.testing.assert_array_equal(ef.open_patterns(path)[2], expected[2])


@pytest.mark.usefixtures("hdf5_fast")
def test_single_frame_decode_error_propagates(tmp_path, monkeypatch):
    expected, path = _patterns(), tmp_path / "single-error.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)

    def fail(*args, **kwargs):
        raise ValueError("bad chunk")

    monkeypatch.setattr(indexed, "_decode_chunk", fail)
    with pytest.raises(ValueError, match="bad chunk"):
        ef.open_patterns(path)[2]


@pytest.mark.usefixtures("hdf5_fast")
def test_single_frame_does_not_change_bulk_dispatch(tmp_path, monkeypatch):
    import emcfile._h5_full_scan as full

    expected, path = _patterns(), tmp_path / "dispatch.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)
    monkeypatch.setattr(full, "MIN_FULL_SCAN_BYTES", 0)

    def unexpected(*args, **kwargs):
        pytest.fail("bulk selections must not use the single-frame path")

    monkeypatch.setattr(indexed, "_single_frame", unexpected)
    source = ef.open_patterns(path)
    assert source[:] == expected
    assert source[np.array([3, 1, 1])] == expected[np.array([3, 1, 1])]


@pytest.mark.parametrize("ones,multi", [([0, 2, 0], [0, 0, 0]), ([0, 0, 0], [0, 2, 0])])
@pytest.mark.usefixtures("hdf5_fast")
def test_single_frame_empty_payloads(tmp_path, monkeypatch, ones, multi):
    expected = ef.PatternsSOne(
        32,
        np.array(ones, "u4"),
        np.array(multi, "u4"),
        np.arange(sum(ones), dtype="u4"),
        np.arange(sum(multi), dtype="u4"),
        np.full(sum(multi), 2, "i4"),
    )
    path = tmp_path / "empty.h5"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    monkeypatch.delenv("EMCFILE_H5_INDEXED_WORKERS", raising=False)
    with ef.open_patterns(path).open() as source:
        for frame in range(3):
            assert source[frame : frame + 1] == expected[frame : frame + 1]
            np.testing.assert_array_equal(source[frame], expected[frame])
