"""Benchmark Parquet(zstd) against HDF5(zstd) across write and read workloads."""

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import emcfile as ef


def _generate_patterns(
    num_patterns: int,
    num_pixels: int,
    ones_per_pattern: int,
    multi_per_pattern: int,
    seed: int = 42,
) -> ef.PatternsSOne:
    rng = np.random.default_rng(seed)
    ones = np.full(num_patterns, ones_per_pattern, dtype=np.uint32)
    multi = np.full(num_patterns, multi_per_pattern, dtype=np.uint32)

    max_one_gap = max(2, num_pixels // max(1, ones_per_pattern))
    one_gaps = rng.integers(
        1, max_one_gap, size=(num_patterns, ones_per_pattern), dtype=np.uint32
    )
    place_ones = np.cumsum(one_gaps, axis=1, dtype=np.uint32).reshape(-1)

    max_multi_gap = max(2, num_pixels // max(1, multi_per_pattern))
    multi_gaps = rng.integers(
        1, max_multi_gap, size=(num_patterns, multi_per_pattern), dtype=np.uint32
    )
    place_multi = np.cumsum(multi_gaps, axis=1, dtype=np.uint32).reshape(-1)
    count_multi = rng.integers(
        2, 8, size=num_patterns * multi_per_pattern, dtype=np.int32
    )
    return ef.PatternsSOne(
        num_pixels, ones, multi, place_ones, place_multi, count_multi
    )


def _median_seconds(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def run_benchmark(
    num_patterns: int = 20_000,
    num_pixels: int = 1_000_000,
    ones_per_pattern: int = 2_000,
    multi_per_pattern: int = 10,
    level: int = 3,
    workers: int = 4,
    repeats: int = 3,
) -> None:
    os.environ["EMCFILE_H5_WRITE_WORKERS"] = str(workers)
    os.environ["EMCFILE_H5_FULL_SCAN_WORKERS"] = str(workers)
    os.environ["EMCFILE_H5_INDEXED_WORKERS"] = str(workers)
    os.environ["EMCFILE_PARQUET_WRITE_WORKERS"] = str(workers)
    os.environ["EMCFILE_PARQUET_FULL_SCAN_WORKERS"] = str(workers)
    os.environ["EMCFILE_PARQUET_INDEXED_WORKERS"] = str(workers)

    patterns = _generate_patterns(
        num_patterns, num_pixels, ones_per_pattern, multi_per_pattern
    )
    raw_mb = patterns.nbytes / (1024**2)

    rng = np.random.default_rng(123)
    rand_indices = rng.choice(num_patterns, size=min(1000, num_patterns), replace=False)
    sort_indices = np.sort(rand_indices)
    single_frames = [(k * 997) % num_patterns for k in range(min(25, num_patterns))]

    print(
        f"Dataset: {num_patterns:,} patterns, {num_pixels:,} pixels, "
        f"{ones_per_pattern} ones/pat, {multi_per_pattern} multi/pat "
        f"({raw_mb:.2f} MiB raw), workers={workers}, zstd level={level}"
    )
    print("-" * 96)
    print(
        f"{'Format':<15} | {'Size (MiB)':>10} | {'Write (s)':>9} | "
        f"{'Full Read (s)':>13} | {'Single Warm':>11} | "
        f"{'Rand 1k (s)':>11} | {'Sort 1k (s)':>11}"
    )
    print("-" * 96)

    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        tmp = Path(tmp_dir)
        for label, filename in [
            ("HDF5(zstd)", "patterns.h5"),
            ("Parquet(zstd)", "patterns.parquet"),
        ]:
            path = tmp / filename
            # Warmup write + read
            patterns.write(
                path,
                overwrite=True,
                position_encoding="delta",
                compression="zstd",
                compression_opts=level,
                shuffle=True,
            )
            loaded = ef.patterns(path)
            assert loaded == patterns

            write_s = _median_seconds(
                lambda: patterns.write(
                    path,
                    overwrite=True,
                    position_encoding="delta",
                    compression="zstd",
                    compression_opts=level,
                    shuffle=True,
                ),
                repeats,
            )
            size_mb = path.stat().st_size / (1024**2)

            full_s = _median_seconds(lambda: ef.patterns(path), repeats)

            source = ef.open_patterns(path)
            with source.open() as reader:
                _ = reader.sparse_pattern(0)
                t0 = time.perf_counter()
                for idx in single_frames:
                    _ = reader.sparse_pattern(idx)
                single_ms = (time.perf_counter() - t0) / len(single_frames) * 1000.0

                _ = reader[rand_indices]
                rand_s = _median_seconds(lambda: reader[rand_indices], repeats)
                sort_s = _median_seconds(lambda: reader[sort_indices], repeats)

            print(
                f"{label:<15} | {size_mb:>10.2f} | {write_s:>9.3f} | "
                f"{full_s:>13.3f} | {single_ms:>9.2f}ms | "
                f"{rand_s:>11.3f} | {sort_s:>11.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", type=int, default=20_000)
    parser.add_argument("--pixels", type=int, default=1_000_000)
    parser.add_argument("--ones", type=int, default=2_000)
    parser.add_argument("--multi", type=int, default=10)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(
        num_patterns=args.patterns,
        num_pixels=args.pixels,
        ones_per_pattern=args.ones,
        multi_per_pattern=args.multi,
        level=args.level,
        workers=args.workers,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
