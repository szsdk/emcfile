"""Paired legacy/current in-memory selection benchmark on a real EMC file.

The old CSR slice implementation remains available as
``PatternsSOne._get_subdataset((slice,))``. The former index-array path is
reproduced below so this script stays useful after the row-gather optimization.

Example:
    PYTHONPATH=src python benchmarks/sz65_real_slicing.py DATA.emc
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc

import numpy as np

from emcfile import PatternsSOne, patterns
from emcfile._indexing import contiguous_ranges


def legacy_indexed(source: PatternsSOne, ids: np.ndarray) -> PatternsSOne:
    if len(ids) == 0:
        return patterns(source.num_pix)
    ranges = contiguous_ranges(ids)
    ones_ranges = source.ones_idx[ranges]
    multi_ranges = source.multi_idx[ranges]
    return PatternsSOne(
        source.num_pix,
        source.ones[ids],
        source.multi[ids],
        np.concatenate([source.place_ones[s:e] for s, e in ones_ranges]),
        np.concatenate([source.place_multi[s:e] for s, e in multi_ranges]),
        np.concatenate([source.count_multi[s:e] for s, e in multi_ranges]),
    )


def legacy_select(source: PatternsSOne, selector):
    if isinstance(selector, slice):
        return source._get_subdataset((selector,))
    if isinstance(selector, np.ndarray):
        return legacy_indexed(
            source, np.flatnonzero(selector) if selector.dtype == bool else selector
        )
    return source[selector]


def peak_allocation(action) -> int:
    gc.collect()
    tracemalloc.start()
    result = action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return peak


def paired_times(old, new, repetitions: int) -> tuple[list[float], list[float]]:
    old()
    new()
    times = ([], [])
    for trial in range(repetitions):
        for method in (0, 1) if trial % 2 == 0 else (1, 0):
            gc.collect()
            start = time.perf_counter_ns()
            result = (old, new)[method]()
            times[method].append((time.perf_counter_ns() - start) / 1e6)
            del result
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="raw EMC source; loading is excluded from timing")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cases", nargs="+", help="run only named cases")
    parser.add_argument("--disable-numba", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3")
    if args.disable_numba:
        sys.modules["emcfile._row_gather_numba"] = None

    load_start = time.perf_counter()
    source = patterns(args.path)
    load_seconds = time.perf_counter() - load_start
    rows = len(source)
    middle = rows // 2
    random = np.random.default_rng(65)
    ids = random.choice(rows, size=min(256, rows), replace=False)
    ids_1000 = random.choice(rows, size=min(1000, rows), replace=False)
    ids_20pct = np.sort(random.choice(rows, size=max(1, rows // 5), replace=False))
    mask = np.zeros(rows, dtype=bool)
    mask[ids] = True
    contiguous_rows = min(1024, rows // 2)
    cases = {
        "contiguous_16": slice(middle, middle + 16),
        "contiguous_1024": slice(middle, middle + contiguous_rows),
        "stride_2_256": slice(middle, middle + 512, 2),
        "stride_2_as_ids": np.arange(middle, middle + 512, 2),
        "sorted_ids_256": np.sort(ids),
        "random_ids_256": ids,
        "mask_256": mask,
        "sorted_ids_1000": np.sort(ids_1000),
        "random_ids_1000": ids_1000,
        "sorted_ids_20pct": ids_20pct,
        "row_and_columns": (slice(middle, middle + 16), slice(0, 1024)),
    }
    if args.cases:
        unknown = set(args.cases) - cases.keys()
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = {name: cases[name] for name in args.cases}
    print(
        json.dumps(
            {
                "path": args.path,
                "shape": source.shape,
                "nbytes": source.nbytes,
                "load_seconds_excluded": load_seconds,
                "repetitions": args.repetitions,
            }
        ),
        flush=True,
    )
    print(
        "case,rows_selected,old_median_ms,new_median_ms,speedup,"
        "old_peak_bytes,new_peak_bytes,old_range_ms,new_range_ms",
        flush=True,
    )
    for name, selector in cases.items():

        def old():
            return legacy_select(source, selector)

        def new():
            return source[selector]

        old_result = old()
        new_result = new()
        assert old_result == new_result, name
        selected_rows = len(new_result)
        del old_result, new_result
        old_times, new_times = paired_times(old, new, args.repetitions)
        old_peak = peak_allocation(old)
        new_peak = peak_allocation(new)
        old_median = statistics.median(old_times)
        new_median = statistics.median(new_times)
        print(
            f"{name},{selected_rows},{old_median:.6f},{new_median:.6f},"
            f"{old_median / new_median:.3f},{old_peak},{new_peak},"
            f"{min(old_times):.6f}-{max(old_times):.6f},"
            f"{min(new_times):.6f}-{max(new_times):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
