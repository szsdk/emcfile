#!/usr/bin/env python3
"""Reproducible SZ-15 HDF5 filter benchmark for emcfile v2 patterns."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import emcfile as ef
import h5py
import hdf5plugin
import numpy as np


ATTRS = ("ones", "multi", "place_ones", "place_multi", "count_multi")
RNG_SEED = 20260911


def median_runs(fn: Any, repeats: int) -> tuple[float, list[float]]:
    values = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), values


def source_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    src = ef.file_patterns(path)
    arrays = {name: np.asarray(getattr(src, name)) for name in ATTRS}
    meta = {"num_data": int(src.num_data), "num_pix": int(src.num_pix)}
    return arrays, meta


def hdf5_kwargs(config: str) -> dict[str, Any]:
    """The requested pipeline. h5py orders scaleoffset, shuffle, compressor."""
    if config == "baseline":
        return {}
    if config == "lzf":
        return {"compression": "lzf"}
    if config == "shuffle_lzf":
        return {"shuffle": True, "compression": "lzf"}
    if config == "scaleoffset":
        return {"scaleoffset": 0}
    if config == "scaleoffset_lzf":
        return {"scaleoffset": 0, "compression": "lzf"}
    if config == "lz4":
        return dict(hdf5plugin.LZ4())
    if config == "shuffle_lz4":
        return {"shuffle": True, **dict(hdf5plugin.LZ4())}
    if config == "scaleoffset_lz4":
        return {"scaleoffset": 0, **dict(hdf5plugin.LZ4())}
    if config == "scaleoffset_shuffle_lzf":
        return {"scaleoffset": 0, "shuffle": True, "compression": "lzf"}
    if config == "scaleoffset_shuffle_lz4":
        return {"scaleoffset": 0, "shuffle": True, **dict(hdf5plugin.LZ4())}
    raise ValueError(config)


def source_chunking(arrays: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    # emcfile's writer passes compression to create_dataset and lets h5py choose
    # chunks.  This is h5py's exact auto-chunking heuristic, frozen once here
    # so every configuration has an identical layout (including baseline).
    from h5py._hl.filters import guess_chunk

    return {
        name: guess_chunk(arrays[name].shape, None, arrays[name].dtype.itemsize)
        for name in ATTRS
    }


def write_file(
    output: Path,
    arrays: dict[str, np.ndarray],
    meta: dict[str, int],
    chunks: dict[str, tuple[int, ...]],
    config: str,
) -> None:
    kwargs = hdf5_kwargs(config)
    with h5py.File(output, "w") as fp:
        group = fp.create_group("patterns")
        group.attrs.update(num_data=meta["num_data"], num_pix=meta["num_pix"], version="2")
        for name in ATTRS:
            group.create_dataset(name, data=arrays[name], chunks=chunks[name], **kwargs)


def filter_pipeline(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with h5py.File(path, "r") as fp:
        for name in ATTRS:
            ds = fp["patterns"][name]
            assert isinstance(ds, h5py.Dataset)
            plist = ds.id.get_create_plist()
            filters = []
            for i in range(plist.get_nfilters()):
                fid, flags, values, label = plist.get_filter(i)
                filters.append(
                    {"id": fid, "name": label.decode() if isinstance(label, bytes) else label,
                     "flags": flags, "values": list(values)}
                )
            result[name] = {"dtype": str(ds.dtype), "shape": list(ds.shape),
                            "chunks": list(ds.chunks or ()), "filters": filters}
    return result


def read_all(path: Path, meta: dict[str, int]) -> None:
    src = ef.file_patterns(f"{path}::/patterns")
    # These are the relevant v2 reader accesses and materialize every integer
    # dataset, then reconstruct the normal in-memory representation.
    obj = ef.PatternsSOne(
        meta["num_pix"], np.asarray(src.ones), np.asarray(src.multi),
        np.asarray(src.place_ones), np.asarray(src.place_multi), np.asarray(src.count_multi),
    )
    assert obj.num_data == meta["num_data"]


def random_access(path: Path, ids: np.ndarray) -> tuple[float, list[float], float, list[float]]:
    def batch() -> None:
        # Keeping the reader open is emcfile's normal repeated-access pattern.
        with ef.file_patterns(f"{path}::/patterns").open() as src:
            for idx in ids:
                src.sparse_pattern(int(idx))

    batch_median, batch_values = median_runs(batch, 3)
    latency = []
    with ef.file_patterns(f"{path}::/patterns").open() as src:
        src.sparse_pattern(int(ids[0]))  # initialise offsets outside timing
        for idx in ids:
            start = time.perf_counter()
            src.sparse_pattern(int(idx))
            latency.append(time.perf_counter() - start)
    return batch_median, batch_values, float(np.median(latency)), latency


def verify(path: Path, arrays: dict[str, np.ndarray], meta: dict[str, int]) -> None:
    src = ef.file_patterns(f"{path}::/patterns")
    assert src.num_data == meta["num_data"] and src.num_pix == meta["num_pix"]
    for name in ATTRS:
        got = np.asarray(getattr(src, name))
        assert got.dtype == arrays[name].dtype and np.array_equal(got, arrays[name]), name
    # Also validate the normal sparse-pattern access at first/interior/last.
    starts_one = np.r_[0, np.cumsum(arrays["ones"], dtype=np.uint64)]
    starts_multi = np.r_[0, np.cumsum(arrays["multi"], dtype=np.uint64)]
    for i in sorted(set((0, meta["num_data"] // 2, meta["num_data"] - 1))):
        pattern = src.sparse_pattern(i)
        a, b = starts_one[i : i + 2]
        c, d = starts_multi[i : i + 2]
        assert np.array_equal(pattern.place_ones, arrays["place_ones"][a:b])
        assert np.array_equal(pattern.place_multi, arrays["place_multi"][c:d])
        assert np.array_equal(pattern.count_multi, arrays["count_multi"][c:d])


def one_config(
    config: str, outdir: Path, arrays: dict[str, np.ndarray], meta: dict[str, int],
    chunks: dict[str, tuple[int, ...]], ids: np.ndarray, repeats: int,
) -> dict[str, Any]:
    output = outdir / f"{config}.h5"
    try:
        write_median, writes = median_runs(
            lambda: (output.unlink(missing_ok=True), write_file(output, arrays, meta, chunks, config)), repeats
        )
        # Keep the final repetition's file; all writes have identical content.
        verify(output, arrays, meta)
        read_median, reads = median_runs(lambda: read_all(output, meta), repeats)
        random_median, random_runs, latency_median, latency = random_access(output, ids)
        size = output.stat().st_size
        return {
            "status": "ok", "path": str(output), "file_bytes": size,
            "write_median_s": write_median, "write_runs_s": writes,
            "write_mib_s": sum(a.nbytes for a in arrays.values()) / write_median / 2**20,
            "full_read_median_s": read_median, "full_read_runs_s": reads,
            "full_read_mib_s": sum(a.nbytes for a in arrays.values()) / read_median / 2**20,
            "random_1000_median_s": random_median, "random_1000_runs_s": random_runs,
            "random_patterns_s": len(ids) / random_median,
            "single_pattern_median_ms": latency_median * 1000,
            "single_pattern_p99_ms": float(np.percentile(latency, 99) * 1000),
            "filter_pipeline": filter_pipeline(output),
        }
    except Exception as exc:
        return {"status": "unsupported_or_failed", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="EMC input or deterministic EMC subset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--random-patterns", type=int, default=1000)
    parser.add_argument("--configs", nargs="+", default=[
        "baseline", "lzf", "shuffle_lzf", "scaleoffset", "scaleoffset_lzf",
        "lz4", "shuffle_lz4", "scaleoffset_lz4",
        "scaleoffset_shuffle_lzf", "scaleoffset_shuffle_lz4",
    ])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, meta = source_arrays(args.input)
    chunks = source_chunking(arrays)
    rng = np.random.default_rng(RNG_SEED)
    ids = rng.integers(0, meta["num_data"], size=args.random_patterns, dtype=np.int64)
    raw = sum(a.nbytes for a in arrays.values())
    report: dict[str, Any] = {
        "input": str(args.input), "raw_integer_bytes": raw, "raw_integer_mib": raw / 2**20,
        "metadata": meta, "chunks": {k: list(v) for k, v in chunks.items()},
        "settings": {"rng_seed": RNG_SEED, "random_patterns": len(ids),
                     "repeats": args.repeats, "cache": "warm OS cache; all configurations timed equally"},
        "versions": {"python": platform.python_version(), "h5py": h5py.version.version,
                     "hdf5": h5py.version.hdf5_version, "hdf5plugin": hdf5plugin.version},
        "results": {},
    }
    for config in args.configs:
        print(f"[{config}]", flush=True)
        result = one_config(config, args.output_dir, arrays, meta, chunks, ids, args.repeats)
        if result["status"] == "ok":
            result["ratio_vs_raw"] = raw / result["file_bytes"]
        report["results"][config] = result
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
