from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import emcfile as ef
import emcfile._parquet as pq_mod

pytest.importorskip("emcfile_fast")


def _patterns(repetitions: int = 1) -> ef.PatternsSOne:
    ones = np.tile(np.array([0, 3, 1, 2], "u4"), repetitions)
    multi = np.tile(np.array([2, 0, 2, 1], "u4"), repetitions)
    one_values = np.concatenate([np.arange(count, dtype="u4") * 6 for count in ones])
    multi_values = np.concatenate(
        [np.arange(count, dtype="u4") * 6 + 1 for count in multi]
    )
    return ef.PatternsSOne(
        32,
        ones,
        multi,
        one_values,
        multi_values,
        np.arange(multi_values.size, dtype="i4") + 2,
    )


def test_plain_import_does_not_load_fast_extra():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, emcfile; assert not any(x in sys.modules for x in ('emcfile_fast', 'hdf5plugin'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_parquet_aliases_and_metadata(tmp_path: Path):
    assert ef.ParquetPatternFile is ef.PatternsSOneParquet
    expected = _patterns(2)
    path = tmp_path / "patterns.parquet"
    expected.write(
        path,
        position_encoding="delta",
        compression="zstd",
        compression_opts=3,
        shuffle=True,
    )
    source = ef.open_patterns(path)
    assert isinstance(source, ef.ParquetPatternFile)
    assert source.num_patterns == expected.num_patterns
    assert source.num_pixels == expected.num_pixels
    assert source.shape == expected.shape
    assert len(source) == len(expected)
    assert source.nbytes == expected.nbytes
    assert np.isclose(source.sparsity(), expected.sparsity())
    np.testing.assert_array_equal(source.ones, expected.ones)
    np.testing.assert_array_equal(source.multi, expected.multi)
    np.testing.assert_array_equal(source.ones_offsets, expected.ones_offsets)
    np.testing.assert_array_equal(source.multi_offsets, expected.multi_offsets)
    np.testing.assert_array_equal(source.place_ones, expected.place_ones)
    np.testing.assert_array_equal(source.place_multi, expected.place_multi)
    np.testing.assert_array_equal(source.count_multi, expected.count_multi)
    assert ef.patterns(path) == expected
    assert ef.patterns(path, start=1, end=5) == expected[1:5]

    fresh = ef.open_patterns(path)
    assert not fresh._init_idx
    assert fresh.nbytes == expected.nbytes
    assert np.isclose(fresh.sparsity(), expected.sparsity())
    np.testing.assert_array_equal(fresh[2], expected[2])
    assert fresh[1:5] == expected[1:5]
    assert fresh[:] == expected
    np.testing.assert_array_equal(fresh.place_ones, expected.place_ones)
    assert not fresh._init_idx


@pytest.mark.parametrize("suffix", [".parquet", ".pq", ".PARQUET"])
def test_parquet_suffixes_and_magic_detection(tmp_path: Path, suffix: str):
    expected = _patterns(1)
    path = tmp_path / f"sample{suffix}"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    assert ef.open_patterns(path)[:] == expected

    no_suffix = tmp_path / f"raw_file_{suffix.strip('.')}"
    no_suffix.write_bytes(path.read_bytes())
    reader = ef.open_patterns(no_suffix)
    assert isinstance(reader, ef.PatternsSOneParquet)
    assert reader[:] == expected


@pytest.mark.parametrize(
    "compression,options,shuffle",
    [
        (None, None, False),
        (None, None, True),
        ("zstd", 1, False),
        ("zstd", 3, True),
    ],
)
@pytest.mark.parametrize("encoding", ["absolute", "delta"])
def test_parquet_codec_roundtrip(
    tmp_path: Path, compression, options, shuffle: bool, encoding: str
):
    expected = _patterns(3)
    path = tmp_path / f"{encoding}-{compression}-{shuffle}.parquet"
    expected.write(
        path,
        position_encoding=encoding,
        compression=compression,
        compression_opts=options,
        shuffle=shuffle,
    )
    source = ef.open_patterns(path)
    assert source[:] == expected
    np.testing.assert_array_equal(source.place_ones, expected.place_ones)
    np.testing.assert_array_equal(source.place_multi, expected.place_multi)
    np.testing.assert_array_equal(source.count_multi, expected.count_multi)


def test_parquet_sortedness_check_is_opt_in(tmp_path: Path):
    invalid = ef.PatternsSOne(
        8,
        np.array([2], "u4"),
        np.array([0], "u4"),
        np.array([4, 3], "u4"),
        np.array([], "u4"),
        np.array([], "i4"),
    )
    invalid.write(tmp_path / "unchecked.parquet", position_encoding="delta")
    with pytest.raises(ValueError, match="sorted"):
        invalid.write(
            tmp_path / "checked.parquet", position_encoding="delta", check_sorted=True
        )


@pytest.mark.parametrize("workers", [0, 1, 2, 4])
@pytest.mark.parametrize("persistent", [False, True])
def test_parquet_indexed_and_full_scan_public_api(
    tmp_path: Path, monkeypatch, workers: int, persistent: bool
):
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "20")
    monkeypatch.setenv("EMCFILE_PARQUET_FULL_SCAN_WORKERS", str(workers))
    monkeypatch.setenv("EMCFILE_PARQUET_INDEXED_WORKERS", str(workers))
    expected = _patterns(7)
    path = tmp_path / "indexed.parquet"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    source = ef.open_patterns(path)
    if persistent:
        source = source.open()
    try:
        assert source[:] == expected
        for selection in (
            np.array([27, 2, 2, 0, 7, 1]),
            np.array([-1, 0, -3, 2]),
            np.array([], dtype=int),
            np.arange(1, 28, 3),
            slice(2, 23),
            slice(-6, -1),
            slice(None, None, -1),
            slice(None, 99),
            slice(99, 120),
            slice(5, 2),
            slice(0, 0, 2),
            slice(0, 28, 3),
            np.arange(28) % 3 == 0,
            (slice(2, 15), slice(4, 20)),
        ):
            assert source[selection] == expected[selection]
        for frame in (0, 3, 15, 27, -1, -5):
            np.testing.assert_array_equal(source[frame], expected[frame])
            actual_sparse = source.sparse_pattern(frame)
            expected_sparse = expected.sparse_pattern(frame)
            for name in ("place_ones", "place_multi", "count_multi"):
                np.testing.assert_array_equal(
                    getattr(actual_sparse, name), getattr(expected_sparse, name)
                )
    finally:
        if persistent:
            source.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_parquet_single_frame_without_workers_or_numba(
    tmp_path: Path, monkeypatch, persistent: bool
):
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "20")
    expected = _patterns(3)
    path = tmp_path / "single.parquet"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)

    def unexpected(*args, **kwargs):
        pytest.fail("single frames must not use workers or Numba JIT")

    monkeypatch.setattr(pq_mod, "ThreadPoolExecutor", unexpected)
    monkeypatch.setattr(pq_mod, "_ensure_decode_kernel", unexpected)
    source = ef.open_patterns(path)
    if persistent:
        source = source.open()
    try:
        for frame in range(expected.num_data):
            np.testing.assert_array_equal(source[frame], expected[frame])
            np.testing.assert_array_equal(source[np.int64(frame)], expected[frame])
            assert source[frame : frame + 1] == expected[frame : frame + 1]
            assert source[np.array([frame])] == expected[frame : frame + 1]
    finally:
        if persistent:
            source.close()


def test_parquet_batched_multi_source_and_collector_write(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "32")
    p1 = _patterns(2)
    p2 = _patterns(3)
    combined = ef.PatternsSOneList([p1, p2])[:]
    path = tmp_path / "multi.parquet"
    ef.write_patterns(
        [p1, p2],
        path,
        buffer_size=48,
        position_encoding="delta",
        compression="zstd",
        shuffle=True,
    )
    assert ef.open_patterns(path)[:] == combined

    coll_path = tmp_path / "collection.parquet"
    ef.EMCPatternCollection([p1, p2]).write(
        coll_path, position_encoding="delta", compression="zstd", shuffle=True
    )
    assert ef.open_patterns(coll_path)[:] == combined

    collector = ef.EMCPatternCollector(batch_size=2)
    collector.extend(combined.todense())
    collector_path = tmp_path / "collector.parquet"
    collector.write(
        collector_path, position_encoding="delta", compression="zstd", shuffle=True
    )
    assert ef.open_patterns(collector_path)[:] == combined


def test_parquet_empty_datasets_and_payloads(tmp_path: Path):
    empty = ef.patterns(16)
    empty_path = tmp_path / "empty.parquet"
    empty.write(empty_path, position_encoding="delta", compression="zstd", shuffle=True)
    source = ef.open_patterns(empty_path)
    assert source.shape == (0, 16)
    assert source[:] == empty
    assert source.place_ones.size == 0
    assert source.place_multi.size == 0
    assert source.count_multi.size == 0

    sparse_empty = ef.PatternsSOne(
        16,
        np.array([0, 2, 0], "u4"),
        np.array([0, 0, 0], "u4"),
        np.array([3, 9], "u4"),
        np.array([], "u4"),
        np.array([], "i4"),
    )
    sparse_path = tmp_path / "sparse_empty.parquet"
    sparse_empty.write(
        sparse_path, position_encoding="delta", compression="zstd", shuffle=True
    )
    with ef.open_patterns(sparse_path).open() as reader:
        assert reader[:] == sparse_empty
        for i in range(3):
            np.testing.assert_array_equal(reader[i], sparse_empty[i])


def test_parquet_missing_dependency_error(tmp_path: Path, monkeypatch):
    expected = _patterns(1)
    path = tmp_path / "no_fast.parquet"
    monkeypatch.setattr(pq_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="fast"):
        expected.write(
            path, position_encoding="delta", compression="zstd", shuffle=True
        )


def test_parquet_errors(tmp_path: Path):
    expected = _patterns(1)
    path = tmp_path / "exists.parquet"
    expected.write(path)
    with pytest.raises(FileExistsError):
        expected.write(path, overwrite=False)
    expected.write(path, overwrite=True)

    with pytest.raises(ValueError, match="position_encoding"):
        expected.write(tmp_path / "bad_enc.parquet", position_encoding="invalid")
    with pytest.raises(ValueError, match="compression"):
        expected.write(tmp_path / "bad_comp.parquet", compression="unknown_codec")
    with pytest.raises(ValueError, match="at least one"):
        ef.write_patterns([], tmp_path / "empty_list.parquet")
    with pytest.raises(ValueError, match="same number of pixels"):
        ef.write_patterns([expected, ef.patterns(64)], tmp_path / "mismatch.parquet")

    source = ef.open_patterns(path)
    source.init_idx()
    with pytest.raises(IndexError):
        pq_mod.read_parquet_patterns(
            path,
            np.array([[0, 99]]),
            source.ones_idx,
            source.multi_idx,
            num_data=source.num_data,
            rg_pats=source.rg_pats,
            position_encoding=source.position_encoding,
            shuffle=source.shuffle,
            payload_codec=source.payload_codec,
        )
