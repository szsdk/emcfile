"""Full-file column-selection benchmark for chunked int32 CSR construction.

Each invocation measures exactly one path, so peak RSS is process-isolated.
Run with ``PYTHONPATH=src python benchmarks/sz65_chunked_columns.py FILE --mode chunked``.
"""

from __future__ import annotations

import argparse
import gc
import resource
import time

import numpy as np
from scipy.sparse import csr_array

from emcfile import PatternsSOne, patterns
from sz65_csr_columns import old_column_selection


def _rss_mib() -> float:
    with open("/proc/self/statm", encoding="ascii") as file:
        pages = int(file.read().split()[1])
    return pages * resource.getpagesize() / (1 << 20)


def _peak_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _chunk_stop(source: PatternsSOne, start: int, budget: int, max_rows: int) -> int:
    limit = min(source.num_data, start + max_rows)
    for offsets in (source.ones_idx, source.multi_idx):
        last = int(offsets[start]) + budget
        limit = min(limit, int(np.searchsorted(offsets, last, side="right") - 1))
    if limit <= start:
        raise ValueError("A single pattern exceeds the int32 event budget")
    return limit


def _chunk_csr(source: PatternsSOne, kind: str, start: int, stop: int) -> csr_array:
    if kind == "ones":
        offsets = source.ones_idx
        place = source.place_ones
    else:
        offsets = source.multi_idx
        place = source.place_multi
    event_start, event_stop = int(offsets[start]), int(offsets[stop])
    # A regular NumPy slice retains the full event array as its base. SciPy's
    # _prune_array copies views smaller than half their base even with
    # copy=False. A bounded memoryview gives this CSR a chunk-sized base.
    indices = np.frombuffer(memoryview(place)[event_start:event_stop], dtype=np.int32)
    indptr = (offsets[start : stop + 1] - event_start).astype(np.int32)
    if kind == "ones":
        seed = np.ones(1, dtype=np.int32)
        data = np.lib.stride_tricks.as_strided(
            seed, shape=(event_stop - event_start,), strides=(0,)
        )
    else:
        data = source.count_multi[event_start:event_stop]
    csr = csr_array(
        (data, indices, indptr), shape=(stop - start, source.num_pix), copy=False
    )
    # SciPy may normalize a one-element index array; that is not a large copy.
    if indices.size > 1 and not np.shares_memory(csr.indices, place):
        raise AssertionError(
            f"SciPy copied {kind} chunk event indices: rows={start}:{stop}, "
            f"events={indices.size}, dtype={csr.indices.dtype}, "
            f"input_address={indices.ctypes.data}, "
            f"csr_address={csr.indices.ctypes.data}"
        )
    if csr.indices.dtype != np.int32 or csr.indptr.dtype != np.int32:
        raise AssertionError("SciPy promoted chunk CSR indices")
    return csr


def chunked_column_selection(
    source: PatternsSOne, columns: int, budget: int, max_rows: int
) -> tuple[PatternsSOne, int]:
    if source.num_pix > np.iinfo(np.int32).max:
        raise ValueError("Pixel indices cannot be viewed as signed int32")
    if budget > np.iinfo(np.int32).max or budget < 1:
        raise ValueError("Event budget must fit signed int32")
    ones_counts = []
    multi_counts = []
    ones_places = []
    multi_places = []
    multi_values = []
    start = 0
    nchunks = 0
    while start < source.num_data:
        stop = _chunk_stop(source, start, budget, max_rows)
        ones = _chunk_csr(source, "ones", start, stop)[:, :columns]
        multi = _chunk_csr(source, "multi", start, stop)[:, :columns]
        ones_counts.append(np.diff(ones.indptr).astype(np.uint32))
        multi_counts.append(np.diff(multi.indptr).astype(np.uint32))
        ones_places.append(ones.indices.astype(np.uint32))
        multi_places.append(multi.indices.astype(np.uint32))
        multi_values.append(multi.data)
        start = stop
        nchunks += 1
    result = PatternsSOne(
        columns,
        np.concatenate(ones_counts),
        np.concatenate(multi_counts),
        np.concatenate(ones_places),
        np.concatenate(multi_places),
        np.concatenate(multi_values),
    )
    return result, nchunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument(
        "--mode", choices=("old", "adaptive", "chunked", "integrated"), required=True
    )
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=1_000_000_000)
    parser.add_argument("--max-rows", type=int, default=2_000_000_000)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    source = patterns(args.path)
    columns = min(args.columns, source.num_pix)
    gc.collect()
    baseline_mib = _rss_mib()
    start = time.perf_counter()
    if args.mode == "old":
        result = old_column_selection(source, columns)
        nchunks = 1
    elif args.mode == "adaptive":
        result = source._get_subdataset((slice(None), slice(None, columns)))
        nchunks = 1
    elif args.mode == "integrated":
        result = source[:, :columns]
        nchunks = -1
    else:
        result, nchunks = chunked_column_selection(
            source, columns, args.budget, args.max_rows
        )
    elapsed = time.perf_counter() - start
    peak_mib = _peak_mib()
    print(
        f"mode={args.mode},shape={source.shape},columns={columns},"
        f"budget={args.budget},max_rows={args.max_rows},chunks={nchunks},"
        f"seconds={elapsed:.3f},baseline_rss_mib={baseline_mib:.1f},"
        f"peak_rss_mib={peak_mib:.1f},growth_mib={peak_mib - baseline_mib:.1f},"
        f"selected_mib={result.nbytes / (1 << 20):.2f}",
        flush=True,
    )
    if args.verify:
        reference = source._get_subdataset((slice(None), slice(None, columns)))
        print(f"exact_equal={result == reference}", flush=True)
        if result != reference:
            raise AssertionError("Chunked selection differs from adaptive selection")


if __name__ == "__main__":
    main()
