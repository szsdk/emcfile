from pathlib import Path

import h5py
import numpy as np
import pytest

import emcfile as ef
import emcfile._emc_patterns as implementation
import emcfile._h5_full_scan as fast
import emcfile._h5_direct_write as direct
import emcfile._h5_filters as filters
from emcfile._h5_workers import effective_workers
from emcfile._delta import decode_pattern_local_delta, encode_pattern_local_delta


def _patterns(repetitions: int = 1) -> ef.PatternsSOne:
    ones = np.tile(np.array([0, 3, 1, 2], "u4"), repetitions)
    multi = np.tile(np.array([2, 0, 2, 1], "u4"), repetitions)
    one_values = np.concatenate([np.arange(count, dtype="u4") * 7 for count in ones])
    multi_values = np.concatenate([np.arange(count, dtype="u4") * 9 for count in multi])
    return ef.PatternsSOne(32, ones, multi, one_values, multi_values, np.arange(multi_values.size, dtype="i4"))


def test_delta_roundtrip_and_explicit_sortedness_check():
    values = np.array([1, 9, 30, 4], "u4")
    counts = np.array([3, 1], "u4")
    encoded = encode_pattern_local_delta(values, counts)
    np.testing.assert_array_equal(decode_pattern_local_delta(encoded, counts), values)
    with pytest.raises(ValueError, match="sorted"):
        encode_pattern_local_delta(np.array([2, 1], "u4"), [2], check_sorted=True)


def test_writer_sortedness_check_is_opt_in(tmp_path: Path):
    invalid = ef.PatternsSOne(8, np.array([2], "u4"), np.array([0], "u4"), np.array([4, 3], "u4"), np.array([], "u4"), np.array([], "i4"))
    invalid.write(tmp_path / "unchecked.h5", position_encoding="delta")
    with pytest.raises(ValueError, match="sorted"):
        invalid.write(tmp_path / "checked.h5", position_encoding="delta", check_sorted=True)


@pytest.mark.parametrize("compression,options,shuffle", [(None, None, False), ("lzf", None, True), ("gzip", 1, True), ("zstd", 1, True)])
@pytest.mark.parametrize("encoding", ["absolute", "delta"])
def test_v2_codec_roundtrip(tmp_path: Path, compression, options, shuffle, encoding):
    expected, path = _patterns(), tmp_path / f"{encoding}-{compression}.h5"
    expected.write(path, position_encoding=encoding, compression=compression, compression_opts=options, shuffle=shuffle)
    assert ef.open_patterns(path)[:] == expected
    with h5py.File(path) as file:
        assert file.attrs["position_encoding"] == encoding
        np.testing.assert_array_equal(file["count_multi"][:], expected.count_multi)


def test_direct_layout_is_standard_and_full_scan_falls_back_cleanly(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fast, "MIN_FULL_SCAN_BYTES", 0)
    expected, path = _patterns(), tmp_path / "direct.h5"
    expected.write(path, position_encoding="delta", compression="zstd", compression_opts=4, shuffle=True)
    with h5py.File(path) as file:
        assert fast.eligible(file)
        assert [file["place_ones"].id.get_create_plist().get_filter(i)[0] for i in range(2)] == [2, 32015]
    monkeypatch.setenv("EMCFILE_H5_FULL_SCAN_WORKERS", "1")
    assert ef.open_patterns(path)[:] == expected
    monkeypatch.setenv("EMCFILE_H5_FULL_SCAN_WORKERS", "0")
    assert ef.open_patterns(path)[:] == expected


def test_direct_writer_uses_one_shared_pool_with_default_budget(tmp_path: Path, monkeypatch):
    workers, pools = [], []
    original = implementation.PrefilteredDatasetWriter

    def writer(*args, **kwargs):
        workers.append(args[2])
        pools.append(kwargs["pool"])
        return original(*args, **kwargs)

    monkeypatch.delenv("EMCFILE_H5_WRITE_WORKERS", raising=False)
    monkeypatch.setattr(implementation, "PrefilteredDatasetWriter", writer)
    _patterns().write(
        tmp_path / "defaults.h5", position_encoding="delta", compression="zstd", shuffle=True
    )
    assert workers == [4] * 5
    assert len({id(pool) for pool in pools}) == 1


def test_effective_workers_honors_affinity(monkeypatch):
    monkeypatch.setattr("emcfile._h5_workers.os.sched_getaffinity", lambda _pid: {1, 2})
    assert effective_workers(8) == 2
    assert effective_workers(0, allow_zero=True) == 0


def test_plain_import_is_lazy():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, emcfile; assert not any(x in sys.modules for x in ('numba', 'zstandard', 'hdf5plugin'))"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_numba_thread_mask_is_restored():
    numba = pytest.importorskip("numba")
    from emcfile._delta import encode_pattern_local_delta_parallel

    before = numba.get_num_threads()
    encode_pattern_local_delta_parallel(np.arange(100, dtype="u4"), [100], max(2, before))
    assert numba.get_num_threads() == before


def test_direct_writer_does_not_select_big_endian(monkeypatch):
    monkeypatch.setattr(direct.sys, "byteorder", "big")
    assert not direct.available()


def test_zstd_missing_plugin_has_actionable_error(tmp_path: Path, monkeypatch):
    def unavailable():
        raise RuntimeError(filters.FAST_EXTRA_MESSAGE)

    monkeypatch.setattr(implementation, "hdf5plugin", unavailable)
    with pytest.raises(RuntimeError, match="hdf5-fast"):
        _patterns().write(tmp_path / "no-plugin.h5", compression="zstd")


def test_vds_preserves_delta_and_uses_generic_reading(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(fast, "MIN_FULL_SCAN_BYTES", 0)
    expected = _patterns()
    first, second, output = tmp_path / "first.h5", tmp_path / "second.h5", tmp_path / "all.h5"
    expected[:2].write(first, position_encoding="delta", compression="zstd", shuffle=True)
    expected[2:].write(second, position_encoding="delta", compression="zstd", shuffle=True)
    ef.create_vds(output, [first, second])
    with h5py.File(output) as file:
        assert not fast.eligible(file)
    assert ef.open_patterns(output)[:] == expected
