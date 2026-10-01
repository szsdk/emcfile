"""Paired full-data row/column benchmark with exact equality and sampled RSS.

Load an optional historical SZ-65 checkout solely as a row-gather reference;
no Numba implementation is retained in the production package. Source loading,
reference JIT compilation and equality checks are outside measured selections.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np
import psutil

from emcfile import PatternsSOne, patterns


class PeakSampler:
    def __init__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.wait(0.005):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def release():
    gc.collect()
    ctypes.CDLL(None).malloc_trim(0)


def historical_gather(source, selector, kernel):
    if isinstance(selector, slice):
        ids = np.arange(*selector.indices(source.num_data), dtype=np.intp)
    elif selector.dtype == bool:
        ids = np.flatnonzero(selector)
    else:
        ids = selector.astype(np.intp, copy=True)
        ids[ids < 0] += source.num_data
    ones, multi = source.ones[ids], source.multi[ids]
    result = PatternsSOne(
        source.num_pix,
        ones,
        multi,
        np.empty(int(ones.sum(dtype=np.uint64)), np.uint32),
        np.empty(int(multi.sum(dtype=np.uint64)), np.uint32),
        np.empty(int(multi.sum(dtype=np.uint64)), np.int32),
    )
    kernel(
        ids,
        source.ones_idx,
        source.multi_idx,
        source.place_ones,
        source.place_multi,
        source.count_multi,
        result.ones_idx,
        result.multi_idx,
        result.place_ones,
        result.place_multi,
        result.count_multi,
    )
    return result


def cases(rows):
    rng = np.random.default_rng(65)
    ids256 = rng.choice(rows, min(256, rows), replace=False)
    ids1000 = rng.choice(rows, min(1000, rows), replace=False)
    repeated = np.resize(ids256[:64], 256).astype(np.int64)
    repeated[1::2] -= rows
    return {
        "stride2": slice(None, None, 2),
        "stride4": slice(None, None, 4),
        "stride16": slice(None, None, 16),
        "sorted20": np.sort(rng.choice(rows, max(1, rows // 5), replace=False)),
        "random256": ids256,
        "sorted256": np.sort(ids256),
        "random1000": ids1000,
        "sorted1000": np.sort(ids1000),
        "repeated_negative": repeated,
        "boolean20": np.arange(rows) % 5 == 0,
        "columns1024": (slice(None), slice(None, 1024)),
        "combined1000": (ids1000, slice(None, 1024)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--reference-checkout", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    kernel = None
    if args.reference_checkout:
        path = args.reference_checkout / "src/emcfile/_row_gather_numba.py"
        spec = importlib.util.spec_from_file_location("sz65_reference_gather", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        kernel = module.gather_rows
    for path in args.paths:
        source = patterns(path)
        for attr in (*source.ATTRS, "ones_idx", "multi_idx"):
            getattr(source, attr).flags.writeable = False
        if kernel:
            historical_gather(source, np.array([0], np.intp), kernel)
        print(
            json.dumps(
                {
                    "path": path,
                    "shape": source.shape,
                    "source_bytes": source.nbytes,
                    "repetitions": args.repetitions,
                    "reference_checkout": str(args.reference_checkout),
                }
            ),
            flush=True,
        )
        selectors = cases(len(source))
        for case in args.cases or selectors:
            selector = selectors[case]
            methods = {
                "numpy" if not isinstance(selector, tuple) else "bounded_csr": lambda: (
                    source[selector]
                )
            }
            if isinstance(selector, tuple):
                methods["full_source_csr"] = lambda: source._get_subdataset(selector)
            elif kernel:
                methods["sz65_numba_reference"] = lambda: historical_gather(
                    source, selector, kernel
                )
            reference = list(methods.values())[-1]()
            for method in methods.values():
                result = method()
                assert result.shape == reference.shape
                for attr in source.ATTRS:
                    if not np.array_equal(
                        getattr(result, attr), getattr(reference, attr)
                    ):
                        raise AssertionError(f"{path}/{case}: {attr} differs")
                del result
            del reference
            release()
            measurements = {name: [] for name in methods}
            for trial in range(args.repetitions):
                names = list(methods)
                if trial % 2:
                    names.reverse()
                for name in names:
                    release()
                    with PeakSampler() as rss:
                        started = time.perf_counter()
                        result = methods[name]()
                        seconds = time.perf_counter() - started
                    sample = {
                        "trial": trial,
                        "seconds": seconds,
                        "peak_growth_bytes": max(0, rss.peak - rss.baseline),
                        "baseline_rss_bytes": rss.baseline,
                        "result_bytes": result.nbytes,
                        "selected_rows": len(result),
                    }
                    measurements[name].append(sample)
                    print(
                        json.dumps(
                            {"path": path, "case": case, "method": name, **sample}
                        ),
                        flush=True,
                    )
                    del result
            for name, samples in measurements.items():
                print(
                    json.dumps(
                        {
                            "summary": True,
                            "path": path,
                            "case": case,
                            "method": name,
                            "median_seconds": statistics.median(
                                s["seconds"] for s in samples
                            ),
                            "median_peak_growth_bytes": statistics.median(
                                s["peak_growth_bytes"] for s in samples
                            ),
                            "exact_equal": True,
                            "readonly_source": True,
                        }
                    ),
                    flush=True,
                )
        source = None
        release()


if __name__ == "__main__":
    main()
