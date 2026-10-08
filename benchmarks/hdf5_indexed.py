"""Reproducible public-API warm indexed benchmark; JSON on stdout.

Run with PATH --repeats 3. Timings include opening/count initialization and
retrieval, not hashing. IDs sample the entire file, seed matches SZ-56.
"""

import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np

import emcfile as ef
from emcfile._delta import decode_segmented_delta_inplace
from emcfile._h5_full_scan import _unshuffle_u32


def digest(patterns):
    checksum = hashlib.blake2b(digest_size=16)
    checksum.update(np.asarray([patterns.num_pix, patterns.num_data], dtype="<i8"))
    for name in patterns.ATTRS:
        checksum.update(np.ascontiguousarray(getattr(patterns, name)).view("u1"))
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--raw")
    args = parser.parse_args()
    _unshuffle_u32(np.zeros(4, "u1"), np.empty(1, "u4"), 1)
    decode_segmented_delta_inplace(
        np.empty(0, "u4"), np.array([0], "u8"), np.empty(0, "u4"), accelerated=True
    )
    count = ef.open_patterns(args.path).num_data
    ids = np.random.default_rng(20260916).choice(
        count, size=min(1000, count), replace=False
    )
    for workload, selection in (("random1k", ids), ("sorted1k", np.sort(ids))):
        reference = digest(ef.open_patterns(args.raw)[selection]) if args.raw else None
        records = {"generic": [], "direct_threads": []}
        for repeat in range(args.repeats):
            # Alternate order; warm-cache test, no eviction claim.
            for mode in (
                ("generic", "direct_threads")
                if repeat % 2 == 0
                else ("direct_threads", "generic")
            ):
                os.environ["EMCFILE_H5_INDEXED_WORKERS"] = (
                    "0" if mode == "generic" else str(args.workers)
                )
                start = time.perf_counter()
                result = ef.open_patterns(args.path)[selection]
                elapsed = time.perf_counter() - start
                checksum = digest(result)
                if reference is None:
                    reference = checksum
                assert checksum == reference, (mode, workload, checksum, reference)
                records[mode].append(elapsed)
                print(
                    json.dumps(
                        {
                            "path": args.path,
                            "patterns": count,
                            "workload": workload,
                            "mode": mode,
                            "workers": args.workers,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "digest": checksum,
                        }
                    ),
                    flush=True,
                )
                del result
                gc.collect()
        print(
            json.dumps(
                {
                    "summary": workload,
                    "median_s": {
                        mode: float(np.median(times)) for mode, times in records.items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
