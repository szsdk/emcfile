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


@pytest.mark.parametrize(
    "compression,options,shuffle,encoding",
    [
        (None, None, False, "absolute"),
        ("zstd", 1, False, "absolute"),
        ("zstd", 3, True, "delta"),
    ],
)
def test_parquet_readable_by_polars(
    tmp_path: Path,
    monkeypatch,
    compression,
    options,
    shuffle: bool,
    encoding: str,
):
    pl = pytest.importorskip("polars")
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "24")
    expected = _patterns(3)
    path = tmp_path / f"polars-{encoding}-{compression}-{shuffle}.parquet"
    expected.write(
        path,
        position_encoding=encoding,
        compression=compression,
        compression_opts=options,
        shuffle=shuffle,
    )

    schema = pl.read_parquet_schema(path)
    assert schema == {
        "place_ones": pl.Binary,
        "place_multi": pl.Binary,
        "count_multi": pl.Binary,
    }

    df = pl.read_parquet(path)
    assert df.shape == (expected.num_data, 3)
    assert df.columns == ["place_ones", "place_multi", "count_multi"]

    for i in range(expected.num_data):
        row = df.row(i)
        assert len(row[0]) == int(expected.ones[i]) * 4
        assert len(row[1]) == int(expected.multi[i]) * 4
        assert len(row[2]) == int(expected.multi[i]) * 4
        if encoding == "absolute" and not shuffle:
            sp = expected.sparse_pattern(i)
            np.testing.assert_array_equal(
                np.frombuffer(row[0], dtype=np.uint32), sp.place_ones
            )
            np.testing.assert_array_equal(
                np.frombuffer(row[1], dtype=np.uint32), sp.place_multi
            )
            np.testing.assert_array_equal(
                np.frombuffer(row[2], dtype=np.int32), sp.count_multi
            )
        elif encoding == "delta" and shuffle:
            sp = expected.sparse_pattern(i)
            po_raw = (
                np.frombuffer(row[0], dtype=np.uint8)
                .reshape(4, -1)
                .T.copy()
                .view(np.uint32)
                .ravel()
            )
            pm_raw = (
                np.frombuffer(row[1], dtype=np.uint8)
                .reshape(4, -1)
                .T.copy()
                .view(np.uint32)
                .ravel()
            )
            cm_raw = (
                np.frombuffer(row[2], dtype=np.uint8)
                .reshape(4, -1)
                .T.copy()
                .view(np.int32)
                .ravel()
            )
            np.testing.assert_array_equal(
                np.cumsum(po_raw, dtype=np.uint32), sp.place_ones
            )
            np.testing.assert_array_equal(
                np.cumsum(pm_raw, dtype=np.uint32), sp.place_multi
            )
            np.testing.assert_array_equal(cm_raw, sp.count_multi)


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


def test_parquet_concurrent_writes_no_deadlock(tmp_path: Path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("EMCFILE_H5_WORKERS", "2")
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "64")
    expected = _patterns(24)

    def _write_and_verify(idx: int) -> bool:
        out = tmp_path / f"concurrent_{idx}.parquet"
        expected.write(out, position_encoding="delta", compression="zstd", shuffle=True)
        return ef.open_patterns(out)[:] == expected

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(_write_and_verify, range(12)))
    assert all(results)


def _pq_fork_child(src_path: str, dst_path: str, conn) -> None:
    try:
        got = ef.open_patterns(src_path)[:]
        got.write(dst_path, position_encoding="delta", compression="zstd", shuffle=True)
        reopened = ef.open_patterns(dst_path)[:]
        conn.send((got == reopened, len(reopened)))
    except BaseException as exc:  # pragma: no cover - diagnostic path
        conn.send((False, repr(exc)))
    finally:
        conn.close()


def test_parquet_fork_safety(tmp_path: Path, monkeypatch):
    import multiprocessing as mp

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method unavailable on this platform")

    monkeypatch.setenv("EMCFILE_H5_WORKERS", "2")
    expected = _patterns(24)
    src = tmp_path / "parent.parquet"
    dst = tmp_path / "child.parquet"
    expected.write(src, position_encoding="delta", compression="zstd", shuffle=True)
    assert ef.open_patterns(src)[:] == expected

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_pq_fork_child, args=(str(src), str(dst), child_conn))
    proc.start()
    child_conn.close()
    proc.join(timeout=15)
    if proc.is_alive():  # pragma: no cover
        proc.kill()
        pytest.fail("Forked Parquet worker hung (Rayon threadpool / mmap deadlock)")
    assert proc.exitcode == 0
    ok, payload = parent_conn.recv()
    parent_conn.close()
    assert ok, payload
    assert payload == len(expected)
    assert ef.open_patterns(dst)[:] == expected


def test_parquet_validates_corrupt_offsets_and_rust_bounds(tmp_path: Path):
    import emcfile_fast

    expected = _patterns(4)
    path = tmp_path / "bounds.parquet"
    expected.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    source = ef.open_patterns(path)
    source.init_idx()

    bad_ones_idx = source.ones_idx.copy()
    bad_ones_idx[-1] += 5
    with pytest.raises(ValueError, match="Pattern counts do not match"):
        pq_mod.read_parquet_patterns_with_counts(
            path,
            np.array([[0, source.num_data]], dtype=np.int64),
            bad_ones_idx,
            source.multi_idx,
            num_data=source.num_data,
            rg_pats=source.rg_pats,
            position_encoding=source.position_encoding,
            shuffle=source.shuffle,
            payload_codec=source.payload_codec,
        )

    with pytest.raises(ValueError, match="exceed src length"):
        emcfile_fast.parquet.encode_rg_u8(
            np.array([1, 2], dtype=np.uint32),
            np.array([0, 99], dtype=np.uint64),
            0,
            1,
            True,
            True,
            False,
        )
    with pytest.raises(ValueError, match="out of bounds"):
        emcfile_fast.parquet.encode_rg_u8(
            np.array([1, 2], dtype=np.uint32),
            np.array([0, 2], dtype=np.uint64),
            0,
            5,
            True,
            True,
            False,
        )


def test_parquet_negative_zstd_level_and_multipage_stats(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "128")
    monkeypatch.setenv("EMCFILE_PARQUET_PAGE_BYTES", "16")
    expected = _patterns(12)
    path = tmp_path / "fast_neg_zstd.parquet"
    expected.write(
        path,
        position_encoding="delta",
        compression="zstd",
        compression_opts=-3,
        shuffle=True,
    )
    meta = pq_mod.read_parquet_metadata(path)
    assert meta["compression"] == "zstd"
    meta["compression"] = "mutated"
    assert pq_mod.read_parquet_metadata(path)["compression"] == "zstd"

    assert ef.open_patterns(path)[:] == expected

    try:
        import polars as pl
    except ImportError:
        pass
    else:
        df = pl.read_parquet(path)
        assert len(df) == len(expected)


def test_parquet_atomic_overwrite_while_mmap_open(tmp_path: Path):
    import sys

    p_large = _patterns(20)
    p_mid = _patterns(9)
    p_small = _patterns(3)
    path = tmp_path / "live_overwrite.parquet"
    p_large.write(path, position_encoding="delta", compression="zstd", shuffle=True)
    assert ef.open_patterns(path)[:] == p_large

    p_mid.write(
        path,
        overwrite=True,
        position_encoding="delta",
        compression="zstd",
        shuffle=True,
    )
    assert ef.open_patterns(path)[:] == p_mid

    if sys.platform != "win32":
        src = ef.open_patterns(path)
        with src.open() as reader:
            first_before = reader[0].copy()
            p_small.write(
                path,
                overwrite=True,
                position_encoding="delta",
                compression="zstd",
                shuffle=True,
            )
            np.testing.assert_array_equal(reader[0], first_before)
            assert ef.open_patterns(path)[:] == p_small


def test_parquet_footer_lru_skips_reparse_and_single_worker_pipeline(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("EMCFILE_PARQUET_CHUNK_BYTES", "64")
    monkeypatch.setenv("EMCFILE_PARQUET_WRITE_WORKERS", "1")
    expected = _patterns(16)
    path = tmp_path / "pipeline_lru.parquet"
    expected.write(
        path,
        position_encoding="delta",
        compression="zstd",
        compression_opts=1,
        shuffle=True,
    )
    pq_mod._invalidate_parquet_cache(path)
    np.testing.assert_array_equal(ef.open_patterns(path)[0], expected[0])

    rs = pq_mod._require_parquet_rs()
    orig_parse = rs.parse_parquet_footer

    def fail_if_reparsed(_data):
        pytest.fail("warm _file_info_lru hit must not re-parse footer")

    monkeypatch.setattr(rs, "parse_parquet_footer", fail_if_reparsed)
    try:
        np.testing.assert_array_equal(ef.open_patterns(path)[5], expected[5])
    finally:
        monkeypatch.setattr(rs, "parse_parquet_footer", orig_parse)
    assert ef.open_patterns(path)[:] == expected
