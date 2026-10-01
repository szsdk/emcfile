"""Compare old and signed-buffer CSR paths for a column slice.

Run with ``PYTHONPATH=src python benchmarks/sz65_csr_columns.py DATA.emc``.
The source file is loaded before timing.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np

from emcfile import PatternsSOne, patterns

from sz65_csr_buffers import construct


def old_column_selection(source: PatternsSOne, stop: int) -> PatternsSOne:
    ones = construct(source, "ones", "u32/u64")[0][:, :stop]
    multi = construct(source, "multi", "u32/u64")[0][:, :stop]
    return PatternsSOne(
        stop,
        np.diff(ones.indptr).astype(np.uint32),
        np.diff(multi.indptr).astype(np.uint32),
        ones.indices.astype(np.uint32),
        multi.indices.astype(np.uint32),
        multi.data,
    )


def measure(action, repetitions: int) -> tuple[float, float, float]:
    action()
    times = []
    for _ in range(repetitions):
        gc.collect()
        before = time.perf_counter_ns()
        result = action()
        times.append((time.perf_counter_ns() - before) / 1e6)
        del result
    gc.collect()
    tracemalloc.start()
    result = action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del result
    return statistics.median(times), min(times), peak / (1 << 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="time one old/new pass and check equality for full datasets",
    )
    args = parser.parse_args()
    source = patterns(args.path)
    stop = min(args.columns, source.num_pix)

    def old():
        return old_column_selection(source, stop)

    def new():
        return source[:, :stop]

    if args.single_pass:
        gc.collect()
        before = time.perf_counter_ns()
        old_result = old()
        old_ms = (time.perf_counter_ns() - before) / 1e6
        gc.collect()
        before = time.perf_counter_ns()
        new_result = new()
        new_ms = (time.perf_counter_ns() - before) / 1e6
        assert old_result == new_result
        print(
            f"shape={source.shape},columns={stop},selected_mib="
            f"{new_result.nbytes / (1 << 20):.2f},old_ms={old_ms:.3f},"
            f"new_ms={new_ms:.3f},exact_equal=True",
            flush=True,
        )
        return

    assert old() == new()
    print(
        f"shape={source.shape},source_mib={source.nbytes / (1 << 20):.1f},"
        f"selected_columns={stop}",
        flush=True,
    )
    print("path,median_ms,min_ms,python_alloc_peak_mib", flush=True)
    for name, action in (("old", old), ("new", new)):
        median, minimum, peak = measure(action, args.repetitions)
        print(f"{name},{median:.3f},{minimum:.3f},{peak:.2f}", flush=True)


if __name__ == "__main__":
    main()
