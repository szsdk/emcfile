"""Warm-cache single-frame and full-read benchmark; JSON lines on stdout.

Run the same command before and after reader changes, against the same file.
Opening/count initialization is measured separately from repeated access.
No OS cache eviction is performed. Checksums are outside the timed sections.
"""

import argparse
import hashlib
import json
import time
from contextlib import nullcontext

import numpy as np

import emcfile as ef


def digest(result):
    checksum = hashlib.blake2b(digest_size=16)
    arrays = (
        [result]
        if isinstance(result, np.ndarray)
        else [getattr(result, name) for name in result.ATTRS]
    )
    for array in arrays:
        checksum.update(np.ascontiguousarray(array).view("u1"))
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("--samples and --repeats must be positive")
    source = ef.open_patterns(args.path)
    if not source.num_data:
        parser.error("the source must contain at least one frame")
    ids = np.random.default_rng(20261008).integers(source.num_data, size=args.samples)
    selections = {
        "random_scalar": ids,
        "sequential_scalar": np.arange(args.samples) % source.num_data,
        "repeated_scalar": np.full(args.samples, ids[0]),
        "single_slice": [slice(int(i), int(i) + 1) for i in ids],
        "indexed": [ids],
        "full": [slice(None)],
    }
    reference = {}
    for persistent in (False, True):
        for workload, selection in selections.items():
            timings = []
            for repeat in range(args.repeats):
                start = time.perf_counter()
                source = ef.open_patterns(args.path)
                with source.open() if persistent else nullcontext(source) as reader:
                    first = reader[int(ids[0])]
                    initialization = time.perf_counter() - start
                    first_digest = digest(first)
                    assert first_digest == reference.setdefault("first", first_digest)
                    # Warm full scans/JIT separately from steady-state timings.
                    if workload == "full":
                        reader[:]
                    checksum = hashlib.blake2b(digest_size=16)
                    elapsed = 0.0
                    for index in selection:
                        start = time.perf_counter()
                        result = reader[index]
                        elapsed += time.perf_counter() - start
                        checksum.update(digest(result).encode())
                    value = checksum.hexdigest()
                    assert value == reference.setdefault(workload, value)
                timings.append(elapsed / len(selection))
                print(
                    json.dumps(
                        {
                            "workload": workload,
                            "persistent": persistent,
                            "repeat": repeat,
                            "initialization_s": initialization,
                            "seconds_per_access": timings[-1],
                            "digest": value,
                        }
                    ),
                    flush=True,
                )
            print(
                json.dumps(
                    {
                        "summary": workload,
                        "persistent": persistent,
                        "median_s": float(np.median(timings)),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
