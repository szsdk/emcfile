"""Measure SciPy CSR construction, dtypes, allocation, and buffer sharing.

Run with ``PYTHONPATH=src python benchmarks/sz65_csr_buffers.py DATA.emc``.
The source file is loaded before timing. Results are in-memory measurements.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc

import numpy as np
from scipy.sparse import csr_array

from emcfile import patterns


def construct(source, kind: str, representation: str):
    place = source.place_ones if kind == "ones" else source.place_multi
    offsets = source.ones_idx if kind == "ones" else source.multi_idx
    if representation == "u32/u64":
        indices = place
        indptr = offsets
    elif representation == "i32/i32":
        indices = place.view(np.int32)
        indptr = offsets.astype(np.int32)
    elif representation == "i32/i64":
        indices = place.view(np.int32)
        indptr = offsets.view(np.int64)
    else:
        raise ValueError(representation)
    if kind == "ones":
        seed = np.ones(1, dtype=np.int32)
        data = np.lib.stride_tricks.as_strided(seed, shape=(place.size,), strides=(0,))
    else:
        data = source.count_multi
    csr = csr_array((data, indices, indptr), shape=source.shape, copy=False)
    return csr, data, indices, indptr


def measure(source, kind: str, representation: str, repetitions: int) -> None:
    offsets = source.ones_idx if kind == "ones" else source.multi_idx
    place = source.place_ones if kind == "ones" else source.place_multi
    if representation == "i32/i32" and int(offsets[-1]) > np.iinfo(np.int32).max:
        print(f"{kind},{representation},SKIP: offsets exceed int32", flush=True)
        return
    if representation != "u32/u64" and (
        place.size and int(place.max()) > np.iinfo(np.int32).max
    ):
        print(f"{kind},{representation},SKIP: pixel index exceeds int32", flush=True)
        return

    times = []
    for _ in range(repetitions):
        gc.collect()
        before = time.perf_counter_ns()
        csr, data, indices, indptr = construct(source, kind, representation)
        times.append((time.perf_counter_ns() - before) / 1e6)
        del csr, data, indices, indptr
    gc.collect()
    tracemalloc.start()
    csr, data, indices, indptr = construct(source, kind, representation)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    source_data = source.count_multi if kind == "multi" else data
    print(
        f"{kind},{representation},{csr.indices.dtype},{csr.indptr.dtype},"
        f"{csr.data.dtype},{np.shares_memory(csr.indices, place)},"
        f"{np.shares_memory(csr.indptr, indptr)},"
        f"{np.shares_memory(csr.indptr, offsets)},"
        f"{np.shares_memory(csr.data, source_data)},"
        f"{csr.data.strides[0] if csr.data.size else 0},"
        f"{statistics.median(times):.3f},{peak / (1 << 20):.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--kinds", nargs="+", choices=("multi", "ones"), default=("multi", "ones")
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=("u32/u64", "i32/i32", "i32/i64"),
        default=("u32/u64", "i32/i32", "i32/i64"),
    )
    args = parser.parse_args()
    source = patterns(args.path)
    print(
        f"shape={source.shape},source_mib={source.nbytes / (1 << 20):.1f},"
        f"ones_events={len(source.place_ones)},multi_events={len(source.place_multi)}",
        flush=True,
    )
    print(
        "kind,input_types,csr_indices,csr_indptr,csr_data,share_indices,"
        "share_passed_indptr,share_source_offsets,share_data,data_stride,"
        "median_ms,python_alloc_peak_mib",
        flush=True,
    )
    for kind in args.kinds:
        for representation in args.representations:
            measure(source, kind, representation, args.repetitions)


if __name__ == "__main__":
    main()
