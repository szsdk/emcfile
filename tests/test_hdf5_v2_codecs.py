from pathlib import Path

import h5py
import numpy as np
import pytest

import emcfile as ef
import emcfile._emc_patterns as implementation
import emcfile._h5_direct_write as direct
import emcfile._h5_filters as filters
import emcfile._h5_full_scan as fast
from emcfile._delta import decode_pattern_local_delta, encode_pattern_local_delta
from emcfile._h5_workers import effective_workers


def _patterns(repetitions: int = 1) -> ef.PatternsSOne:
    ones = np.tile(np.array([0, 3, 1, 2], "u4"), repetitions)
    multi = np.tile(np.array([2, 0, 2, 1], "u4"), repetitions)
    one_values = np.concatenate([np.arange(count, dtype="u4") * 7 for count in ones])
    multi_values = np.concatenate([np.arange(count, dtype="u4") * 9 for count in multi])
    return ef.PatternsSOne(
        32,
        ones,
        multi,
        one_values,
        multi_values,
        np.arange(multi_values.size, dtype="i4"),
    )


def test_delta_roundtrip_and_explicit_sortedness_check():
    values = np.array([1, 9, 30, 4], "u4")
    counts = np.array([3, 1], "u4")
    encoded = encode_pattern_local_delta(values, counts)
    np.testing.assert_array_equal(decode_pattern_local_delta(encoded, counts), values)
    with pytest.raises(ValueError, match="sorted"):
        encode_pattern_local_delta(np.array([2, 1], "u4"), [2], check_sorted=True)


def test_writer_sortedness_check_is_opt_in(tmp_path: Path):
    invalid = ef.PatternsSOne(
        8,
        np.array([2], "u4"),
        np.array([0], "u4"),
        np.array([4, 3], "u4"),
        np.array([], "u4"),
        np.array([], "i4"),
    )
    invalid.write(tmp_path / "unchecked.h5", position_encoding="delta")
    with pytest.raises(ValueError, match="sorted"):
        invalid.write(
            tmp_path / "checked.h5", position_encoding="delta", check_sorted=True
        )


@pytest.mark.parametrize(
    "compression,options,shuffle",
    [(None, None, False), ("lzf", None, True), ("gzip", 1, True), ("zstd", 1, True)],
)
@pytest.mark.parametrize("encoding", ["absolute", "delta"])
def test_v2_codec_roundtrip(tmp_path: Path, compression, options, shuffle, encoding):
    if compression == "zstd":
        pytest.importorskip("hdf5plugin")
    expected, path = _patterns(), tmp_path / f"{encoding}-{compression}.h5"
    expected.write(
        path,
        position_encoding=encoding,
        compression=compression,
        compression_opts=options,
        shuffle=shuffle,
    )
    assert ef.open_patterns(path)[:] == expected
    with h5py.File(path, "r") as file:
        assert str(file.attrs["version"]) == "2"
        assert int(file.attrs["num_pix"]) == expected.num_pix
        assert int(file.attrs["num_data"]) == expected.num_data
        assert file.attrs["position_encoding"] == encoding
        np.testing.assert_array_equal(file["ones"][:], expected.ones)
        np.testing.assert_array_equal(file["multi"][:], expected.multi)
        np.testing.assert_array_equal(file["count_multi"][:], expected.count_multi)
        if encoding == "absolute":
            np.testing.assert_array_equal(file["place_ones"][:], expected.place_ones)
            np.testing.assert_array_equal(file["place_multi"][:], expected.place_multi)
        else:
            np.testing.assert_array_equal(
                decode_pattern_local_delta(file["place_ones"][:], expected.ones),
                expected.place_ones,
            )
            np.testing.assert_array_equal(
                decode_pattern_local_delta(file["place_multi"][:], expected.multi),
                expected.place_multi,
            )


def test_direct_layout_is_standard_and_full_scan_falls_back_cleanly(
    tmp_path: Path, monkeypatch, hdf5_fast
):
    monkeypatch.setattr(fast, "MIN_FULL_SCAN_BYTES", 0)
    expected, path = _patterns(), tmp_path / "direct.h5"
    expected.write(
        path,
        position_encoding="delta",
        compression="zstd",
        compression_opts=4,
        shuffle=True,
    )
    with h5py.File(path) as file:
        assert fast.eligible(file)
        assert [
            file["place_ones"].id.get_create_plist().get_filter(i)[0] for i in range(2)
        ] == [2, 32015]
    monkeypatch.setenv("EMCFILE_H5_FULL_SCAN_WORKERS", "1")
    assert ef.open_patterns(path)[:] == expected
    monkeypatch.setenv("EMCFILE_H5_FULL_SCAN_WORKERS", "0")
    assert ef.open_patterns(path)[:] == expected


def test_direct_writer_uses_one_shared_pool_with_default_budget(
    tmp_path: Path, monkeypatch, hdf5_fast
):
    workers, pools = [], []
    original = implementation.PrefilteredDatasetWriter

    def writer(*args, **kwargs):
        workers.append(args[2])
        pools.append(kwargs["pool"])
        return original(*args, **kwargs)

    monkeypatch.delenv("EMCFILE_H5_WRITE_WORKERS", raising=False)
    monkeypatch.setattr(
        "emcfile._h5_workers.os.sched_getaffinity",
        lambda _pid: set(range(8)),
        raising=False,
    )
    monkeypatch.setattr(implementation, "PrefilteredDatasetWriter", writer)
    _patterns().write(
        tmp_path / "defaults.h5",
        position_encoding="delta",
        compression="zstd",
        shuffle=True,
    )
    assert workers == [4] * 5
    assert len({id(pool) for pool in pools}) == 1


def test_effective_workers_honors_affinity(monkeypatch):
    monkeypatch.setattr(
        "emcfile._h5_workers.os.sched_getaffinity", lambda _pid: {1, 2}, raising=False
    )
    assert effective_workers(8) == 2
    assert effective_workers(0, allow_zero=True) == 0


@pytest.mark.parametrize("affinity_missing", [True, False])
@pytest.mark.parametrize("cpu_count,expected", [(2, 2), (None, 1)])
def test_effective_workers_without_affinity(
    monkeypatch, affinity_missing, cpu_count, expected
):
    if affinity_missing:
        monkeypatch.delattr("emcfile._h5_workers.os.sched_getaffinity", raising=False)
    else:

        def unavailable(_pid):
            raise OSError("CPU affinity unavailable")

        monkeypatch.setattr(
            "emcfile._h5_workers.os.sched_getaffinity", unavailable, raising=False
        )
    monkeypatch.setattr("emcfile._h5_workers.os.cpu_count", lambda: cpu_count)
    assert effective_workers(8) == expected


def test_plain_import_is_lazy():
    import subprocess
    import sys

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


def test_direct_writer_does_not_select_big_endian(monkeypatch):
    monkeypatch.setattr(direct.sys, "byteorder", "big")
    assert not direct.available()


def test_zstd_missing_plugin_has_actionable_error(tmp_path: Path, monkeypatch):
    def unavailable():
        raise RuntimeError(filters.FAST_EXTRA_MESSAGE)

    monkeypatch.setattr(implementation, "hdf5plugin", unavailable)
    with pytest.raises(RuntimeError, match="fast"):
        _patterns().write(tmp_path / "no-plugin.h5", compression="zstd")


def test_vds_preserves_delta_and_uses_generic_reading(
    tmp_path: Path, monkeypatch, hdf5_fast
):
    monkeypatch.setattr(fast, "MIN_FULL_SCAN_BYTES", 0)
    expected = _patterns()
    first, second, output = (
        tmp_path / "first.h5",
        tmp_path / "second.h5",
        tmp_path / "all.h5",
    )
    expected[:2].write(
        first, position_encoding="delta", compression="zstd", shuffle=True
    )
    expected[2:].write(
        second, position_encoding="delta", compression="zstd", shuffle=True
    )
    ef.create_vds(output, [first, second])
    with h5py.File(output) as file:
        assert not fast.eligible(file)
    assert ef.open_patterns(output)[:] == expected


def test_direct_zstd_writer_coerces_int64_arrays(tmp_path: Path, hdf5_fast):
    path = tmp_path / "int64_direct.h5"
    f_kwargs = implementation._h5_filter_kwargs("zstd", 1, True)
    with h5py.File(path, "w") as fp:
        ds_plain = fp.create_dataset(
            "count_multi", shape=(5,), chunks=(3,), dtype=np.int32, **f_kwargs
        )
        w_plain = direct.PrefilteredDatasetWriter(ds_plain, level=1, workers=2)
        w_plain.write(np.array([10, 20, 30, 40, 50], dtype=np.int64))
        w_plain.close()

        ds_delta = fp.create_dataset(
            "place_ones", shape=(4,), chunks=(3,), dtype=np.uint32, **f_kwargs
        )
        w_delta = direct.PrefilteredDatasetWriter(ds_delta, level=1, workers=2)
        w_delta.write_delta(
            np.array([1, 5, 2, 9], dtype=np.int64),
            np.array([0, 2, 4], dtype=np.uint64),
            np.array([2, 2], dtype=np.int64),
            check_sorted=True,
        )
        w_delta.close()

        ds_bad = fp.create_dataset(
            "bad", shape=(1,), chunks=(1,), dtype=np.int32, **f_kwargs
        )
        bad = direct.PrefilteredDatasetWriter(ds_bad, level=1, workers=1)
        try:
            with pytest.raises(ValueError, match="exceed"):
                bad.write(np.array([1 << 40], dtype=np.int64))
        finally:
            bad.close()

    with h5py.File(path, "r") as fp:
        filters.ensure_group_filters(fp)
        np.testing.assert_array_equal(
            fp["count_multi"][...], np.array([10, 20, 30, 40, 50], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            decode_pattern_local_delta(fp["place_ones"][...], np.array([2, 2], "u4")),
            np.array([1, 5, 2, 9], dtype=np.uint32),
        )


def test_rust_h5_rejects_fortran_and_invalid_offsets(tmp_path: Path, hdf5_fast):
    import emcfile_fast

    p = _patterns(4)
    path = tmp_path / "bounds.h5"
    p.write(path, position_encoding="delta", compression="zstd", shuffle=True)

    raw = path.read_bytes()
    f_ranges = np.asfortranarray(np.array([[0, 2], [4, 6]], dtype=np.int64))
    dummy_table = np.array([[0, 0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="C-contiguous"):
        emcfile_fast.h5.decode_indexed_mmap(
            raw,
            dummy_table,
            1024,
            f_ranges,
            p.ones_offsets,
            True,
            2,
        )

    bad_offsets = np.array([5, 2, 5], dtype=np.uint64)
    with pytest.raises(ValueError, match="monotonic|Pattern counts"):
        emcfile_fast.h5.encode_chunks_batch(
            np.array([1, 2, 3, 4, 5], dtype=np.uint32),
            bad_offsets,
            0,
            1,
            4,
            1,
            True,
            False,
            2,
        )


def _h5_fork_child(src_path: str, dst_path: str, conn) -> None:
    try:
        got = ef.open_patterns(src_path)[:]
        got.write(
            dst_path, position_encoding="delta", compression="zstd", shuffle=True
        )
        reopened = ef.open_patterns(dst_path)[:]
        conn.send((got == reopened, len(reopened)))
    except BaseException as exc:  # pragma: no cover - diagnostic path
        conn.send((False, repr(exc)))
    finally:
        conn.close()


def test_h5_zstd_fork_safety(tmp_path: Path, monkeypatch, hdf5_fast):
    import multiprocessing as mp

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method unavailable on this platform")

    monkeypatch.setenv("EMCFILE_H5_WORKERS", "2")
    monkeypatch.setattr(fast, "MIN_FULL_SCAN_BYTES", 0)
    expected = _patterns(32)
    src = tmp_path / "parent.h5"
    dst = tmp_path / "child.h5"
    expected.write(src, position_encoding="delta", compression="zstd", shuffle=True)
    assert ef.open_patterns(src)[:] == expected

    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_h5_fork_child, args=(str(src), str(dst), child_conn))
    proc.start()
    child_conn.close()
    proc.join(timeout=15)
    if proc.is_alive():  # pragma: no cover
        proc.kill()
        pytest.fail("Forked HDF5 worker hung (Rayon threadpool / cache deadlock)")
    assert proc.exitcode == 0
    ok, payload = parent_conn.recv()
    parent_conn.close()
    assert ok, payload
    assert payload == len(expected)
    assert ef.open_patterns(dst)[:] == expected


def test_vds_zstd_fresh_process_and_subgroup(tmp_path: Path, hdf5_fast):
    import subprocess
    import sys

    p0 = _patterns(5)
    p1 = _patterns(7)
    s0 = tmp_path / "part0.h5"
    s1 = tmp_path / "part1.h5"
    p0.write(s0, position_encoding="delta", compression="zstd", shuffle=True)
    p1.write(s1, position_encoding="delta", compression="zstd", shuffle=True)

    vds_root = tmp_path / "vds_root.h5"
    ef.create_vds(vds_root, [s0, s1])

    multi_h5 = tmp_path / "multi_group.h5"
    ef.write_array(f"{multi_h5}::/other", np.arange(4, dtype=np.int32))
    ef.create_vds(f"{multi_h5}::/vds_sub", [s0, s1], overwrite=False)
    np.testing.assert_array_equal(
        ef.read_array(f"{multi_h5}::/other"), np.arange(4, dtype=np.int32)
    )

    script = """
import sys
import emcfile as ef

assert "hdf5plugin" not in sys.modules
r_root = ef.open_patterns(sys.argv[1])[:]
r_sub = ef.patterns(sys.argv[2])
assert r_root == r_sub
assert r_root.shape == (48, 32)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(vds_root), f"{multi_h5}::/vds_sub"],
        check=True,
    )


def test_h5_atomic_overwrite_while_mmap_open_and_clean_failures(
    tmp_path: Path, hdf5_fast
):
    import sys

    p_large = _patterns(24)
    p_mid = _patterns(10)
    p_small = _patterns(3)
    target = tmp_path / "live_overwrite.h5"
    p_large.write(target, position_encoding="delta", compression="zstd", shuffle=True)
    assert ef.open_patterns(target)[:] == p_large

    # Cached mmap invalidation before overwrite works across all OSes (including Windows).
    p_mid.write(
        target,
        overwrite=True,
        position_encoding="delta",
        compression="zstd",
        shuffle=True,
    )
    assert ef.open_patterns(target)[:] == p_mid

    if sys.platform != "win32":
        src = ef.open_patterns(target)
        with src.open() as reader:
            first_before = reader[0].copy()
            p_small.write(
                target,
                overwrite=True,
                position_encoding="delta",
                compression="zstd",
                shuffle=True,
            )
            np.testing.assert_array_equal(reader[0], first_before)
            assert ef.open_patterns(target)[:] == p_small
    else:
        p_small.write(
            target,
            overwrite=True,
            position_encoding="delta",
            compression="zstd",
            shuffle=True,
        )
        assert ef.open_patterns(target)[:] == p_small

    with pytest.raises(FileExistsError):
        p_small.write(target, overwrite=False)
    assert ef.open_patterns(target)[:] == p_small

    unsorted = ef.PatternsSOne(
        8,
        np.array([2], dtype=np.uint32),
        np.array([0], dtype=np.uint32),
        np.array([5, 1], dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.int32),
    )
    failed_new = tmp_path / "should_not_exist.h5"
    with pytest.raises(ValueError, match="sorted"):
        unsorted.write(
            failed_new,
            position_encoding="delta",
            compression="zstd",
            shuffle=True,
            check_sorted=True,
        )
    assert not failed_new.exists()


