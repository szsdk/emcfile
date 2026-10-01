"""Compare Numba gather, NumPy fallback, and bounded CSR row selection.

The source is loaded once before measurements. Numba compilation is warmed
before trials. Every timing includes construction of the returned
PatternsSOne. RSS is sampled during each action; timing order is paired and
rotated to reduce shared-node drift.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib
import statistics
import sys
import threading
import time

import numpy as np
import psutil
from scipy.sparse import csr_array

from emcfile import PatternsSOne, patterns

_NUMBA_MODULE = "emcfile._row_gather_numba"


def make_cases(rows: int, seed: int = 65) -> dict[str, slice | np.ndarray]:
    rng = np.random.default_rng(seed)
    random_256 = rng.choice(rows, min(256, rows), replace=False)
    random_1000 = rng.choice(rows, min(1000, rows), replace=False)
    repeated = np.resize(random_256[:64], 256).astype(np.int64)
    repeated[1::2] -= rows
    return {
        "stride_2": slice(None, None, 2),
        "stride_4": slice(None, None, 4),
        "stride_16": slice(None, None, 16),
        "random_256": random_256,
        "sorted_256": np.sort(random_256),
        "random_1000": random_1000,
        "sorted_1000": np.sort(random_1000),
        "repeated_negative_256": repeated,
        "sorted_20pct": np.sort(rng.choice(rows, max(1, rows // 5), replace=False)),
    }


def normalize_ids(selector: slice | np.ndarray, rows: int) -> np.ndarray:
    if isinstance(selector, slice):
        ids = np.arange(*selector.indices(rows), dtype=np.intp)
    elif selector.dtype == bool:
        ids = np.flatnonzero(selector)
    else:
        ids = selector.astype(np.intp, copy=True)
        ids[ids < 0] += rows
    if ids.size and (np.any(ids < 0) or np.any(ids >= rows)):
        raise IndexError("row selection index out of range")
    return ids


def _bounded_stop(source: PatternsSOne, start: int, stop: int, budget: int) -> int:
    chunk_stop = min(stop, start + np.iinfo(np.int32).max)
    for offsets in (source.ones_idx, source.multi_idx):
        last = int(offsets[start]) + budget
        chunk_stop = min(
            chunk_stop, int(np.searchsorted(offsets, last, side="right") - 1)
        )
    if chunk_stop <= start:
        raise ValueError("A single row exceeds the CSR event budget")
    return chunk_stop


def _csr_for_rows(source: PatternsSOne, kind: str, start: int, stop: int) -> csr_array:
    if kind == "ones":
        place, offsets, data = source.place_ones, source.ones_idx, None
    else:
        place, offsets, data = source.place_multi, source.multi_idx, source.count_multi
    event_start, event_stop = int(offsets[start]), int(offsets[stop])
    # SciPy copies a view when it is less than half the size of its base.
    indices = np.frombuffer(memoryview(place)[event_start:event_stop], dtype=np.int32)
    indptr = (offsets[start : stop + 1] - event_start).astype(np.int32)
    if data is None:
        seed = np.ones(1, dtype=np.int32)
        values = np.lib.stride_tricks.as_strided(
            seed, shape=(event_stop - event_start,), strides=(0,)
        )
    else:
        values = data[event_start:event_stop]
    result = csr_array(
        (values, indices, indptr), shape=(stop - start, source.num_pix), copy=False
    )
    if result.indices.dtype != np.int32 or result.indptr.dtype != np.int32:
        raise AssertionError("SciPy promoted bounded CSR indices")
    if indices.size > 1 and not np.shares_memory(result.indices, place):
        raise AssertionError("SciPy copied a bounded CSR event-index chunk")
    return result


def _from_sparse(ones: csr_array, multi: csr_array, num_pix: int) -> PatternsSOne:
    return PatternsSOne(
        num_pix,
        np.diff(ones.indptr).astype(np.uint32),
        np.diff(multi.indptr).astype(np.uint32),
        ones.indices.astype(np.uint32),
        multi.indices.astype(np.uint32),
        multi.data,
    )


def _join(parts: list[PatternsSOne], num_pix: int) -> PatternsSOne:
    return PatternsSOne(
        num_pix,
        np.concatenate([part.ones for part in parts]),
        np.concatenate([part.multi for part in parts]),
        np.concatenate([part.place_ones for part in parts]),
        np.concatenate([part.place_multi for part in parts]),
        np.concatenate([part.count_multi for part in parts]),
    )


def bounded_csr_select(
    source: PatternsSOne,
    selector: slice | np.ndarray,
    *,
    event_budget: int,
    row_block: int = 8192,
) -> PatternsSOne:
    """Build CSR only for selected runs or bounded source row blocks."""
    ids = normalize_ids(selector, source.num_data)
    if not ids.size:
        return PatternsSOne(
            source.num_pix,
            np.empty(0, np.uint32),
            np.empty(0, np.uint32),
            np.empty(0, np.uint32),
            np.empty(0, np.uint32),
            np.empty(0, np.int32),
        )
    if source.num_pix > np.iinfo(np.int32).max:
        raise ValueError("pixel positions do not fit signed int32")

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    groups: list[tuple[int, int, int, int]] = []
    sparse_limit = max(10_000, source.num_data // 20)
    if ids.size < sparse_limit:
        unique = np.unique(sorted_ids)
        run_start = previous = int(unique[0])
        for value in unique[1:]:
            current = int(value)
            if current != previous + 1:
                left = int(np.searchsorted(sorted_ids, run_start, side="left"))
                right = int(np.searchsorted(sorted_ids, previous, side="right"))
                groups.append((run_start, previous + 1, left, right))
                run_start = current
            previous = current
        left = int(np.searchsorted(sorted_ids, run_start, side="left"))
        right = int(np.searchsorted(sorted_ids, previous, side="right"))
        groups.append((run_start, previous + 1, left, right))
    else:
        cursor = 0
        while cursor < source.num_data:
            stop = _bounded_stop(
                source, cursor, min(source.num_data, cursor + row_block), event_budget
            )
            left = int(np.searchsorted(sorted_ids, cursor, side="left"))
            right = int(np.searchsorted(sorted_ids, stop, side="left"))
            if left < right:
                groups.append((cursor, stop, left, right))
            cursor = stop

    ordered_parts: list[PatternsSOne] = []
    per_output: list[PatternsSOne | None] | None = (
        [None] * ids.size if not np.array_equal(order, np.arange(ids.size)) else None
    )
    for start, stop, left, right in groups:
        ones_csr = _csr_for_rows(source, "ones", start, stop)
        multi_csr = _csr_for_rows(source, "multi", start, stop)
        if isinstance(selector, slice) and (selector.step is None or selector.step > 0):
            chunk_selector = slice(
                int(sorted_ids[left]) - start,
                int(sorted_ids[right - 1]) - start + 1,
                selector.step or 1,
            )
            selected_ones = ones_csr[chunk_selector]
            selected_multi = multi_csr[chunk_selector]
        else:
            local_sorted = sorted_ids[left:right] - start
            selected_ones = ones_csr[local_sorted]
            selected_multi = multi_csr[local_sorted]
        selected = _from_sparse(selected_ones, selected_multi, source.num_pix)
        if per_output is None:
            ordered_parts.append(selected)
        else:
            for local, output_position in enumerate(order[left:right]):
                row = selected[local : local + 1]
                assert isinstance(row, PatternsSOne)
                per_output[int(output_position)] = row

    if per_output is None:
        return _join(ordered_parts, source.num_pix)
    rows = [part for part in per_output if part is not None]
    if len(rows) != ids.size:
        raise AssertionError("Some requested rows were not reconstructed")
    return _join(rows, source.num_pix)


class PeakSampler:
    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.process = psutil.Process()
        self.baseline = 0
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self.stop.wait(self.interval)

    def __enter__(self):
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()


def numpy_select(source: PatternsSOne, selector: slice | np.ndarray) -> PatternsSOne:
    sys.modules[_NUMBA_MODULE] = None
    try:
        return source[selector]
    finally:
        sys.modules.pop(_NUMBA_MODULE, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--event-budget", type=int, default=500_000_000)
    parser.add_argument("--row-block", type=int, default=65_536)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    source = patterns(args.path)
    cases = make_cases(source.num_data)
    if args.cases:
        unknown = set(args.cases) - cases.keys()
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = {name: cases[name] for name in args.cases}
    numba_mod = importlib.import_module(_NUMBA_MODULE)
    warm_ids = np.array([0, min(1, source.num_data - 1)], dtype=np.intp)
    warm = source[warm_ids]
    del warm
    print(
        f"shape={source.shape},source_mib={source.nbytes / (1 << 20):.1f},"
        f"repetitions={args.repetitions},event_budget={args.event_budget}",
        flush=True,
    )
    print(
        "case,method,selected_rows,median_ms,min_ms,max_ms,median_peak_growth_mib,exact_equal",
        flush=True,
    )
    for case, selector in cases.items():
        ids = normalize_ids(selector, source.num_data)
        methods = {
            "numba": lambda: source[selector],
            "numpy": lambda: numpy_select(source, selector),
            "bounded_csr": lambda: bounded_csr_select(
                source,
                selector,
                event_budget=args.event_budget,
                row_block=args.row_block,
            ),
        }
        reference = methods["numba"]()
        for name, method in methods.items():
            candidate = method()
            equal = candidate == reference
            del candidate
            if not equal:
                raise AssertionError(f"{case}/{name} differs from Numba gather")
            print(f"validated,{case},{name}", flush=True)
        del reference
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except AttributeError:
            pass
        timings: dict[str, list[float]] = {name: [] for name in methods}
        peaks: dict[str, list[float]] = {name: [] for name in methods}
        names = list(methods)
        for trial in range(args.repetitions):
            order = names[trial % len(names) :] + names[: trial % len(names)]
            if trial % 2:
                order.reverse()
            for name in order:
                gc.collect()
                sampler = PeakSampler()
                with sampler:
                    start = time.perf_counter_ns()
                    result = methods[name]()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                growth = max(0, sampler.peak - sampler.baseline) / (1 << 20)
                timings[name].append(elapsed)
                peaks[name].append(growth)
                del result
                gc.collect()
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except AttributeError:
                    pass
        for name in methods:
            print(
                f"{case},{name},{ids.size},"
                f"{statistics.median(timings[name]):.3f},"
                f"{min(timings[name]):.3f},{max(timings[name]):.3f},"
                f"{statistics.median(peaks[name]):.1f},True",
                flush=True,
            )
    del numba_mod


if __name__ == "__main__":
    main()
