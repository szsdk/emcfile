"""Compare native contiguous row slicing with the former CSR-backed path.

Run with ``PYTHONPATH=src python benchmarks/contiguous_slice.py``.
The selected row count stays fixed while the source grows.
"""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc

import numpy as np

from emcfile import PatternsSOne


def make_patterns(rows: int) -> PatternsSOne:
    ones = np.full(rows, 8, dtype=np.uint32)
    multi = np.full(rows, 2, dtype=np.uint32)
    return PatternsSOne(
        1024,
        ones,
        multi,
        np.tile(np.arange(8, dtype=np.uint32), rows),
        np.tile(np.arange(8, 10, dtype=np.uint32), rows),
        np.full(rows * 2, 2, dtype=np.int32),
    )


def measure(action, repetitions: int) -> tuple[float, int]:
    action()  # warmup
    times = []
    for _ in range(repetitions):
        before = time.perf_counter_ns()
        action()
        times.append((time.perf_counter_ns() - before) / 1e6)
    tracemalloc.start()
    action()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(times), peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1_000, 100_000, 1_000_000]
    )
    parser.add_argument("--selected", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=15)
    args = parser.parse_args()
    print("rows,method,median_ms,python_peak_bytes")
    for rows in args.rows:
        if rows < args.selected:
            parser.error("each --rows value must be >= --selected")
        patterns = make_patterns(rows)
        start = rows // 2
        selection = slice(start, start + args.selected)

        def direct():
            return patterns[selection]

        def csr():
            return patterns._get_subdataset((selection,))

        assert direct() == csr()
        for name, action in (("direct", direct), ("csr", csr)):
            median, peak = measure(action, args.repetitions)
            print(f"{rows},{name},{median:.6f},{peak}", flush=True)


if __name__ == "__main__":
    main()
