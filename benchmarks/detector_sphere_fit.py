"""Compare Rust and SciPy sphere fitting on synthetic detector caps.

Run with ``uv run --extra fast python benchmarks/detector_sphere_fit.py``.
Reports median wall times, including public API validation and dispatch.
"""

from __future__ import annotations

import argparse
import os
from statistics import median
from time import perf_counter

import numpy as np

import emcfile as ef
from emcfile._detector import _get_det_rs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, nargs="+", default=[4_225, 100_000])
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.repetitions < 1 or any(n < 4 for n in args.points):
        parser.error("points must be >= 4 and repetitions must be >= 1")

    saved = os.environ.pop("EMCFILE_DISABLE_DET_ACCEL", None)
    try:
        if not hasattr(_get_det_rs(), "fit_ewald_sphere_center_rs"):
            parser.error("rebuild emcfile-fast with sphere fitting support")
        print("points,backend,median_ms,speedup")
        for size in args.points:
            rng = np.random.default_rng(17)
            xy = rng.uniform(-32, 32, size=(size, 2))
            coordinates = np.column_stack(
                [xy, np.sqrt(128**2 - np.sum(xy**2, axis=1)) - 128]
            )
            timings = {}
            results = {}
            for backend, disabled in (("python", "1"), ("rust", "0")):
                os.environ["EMCFILE_DISABLE_DET_ACCEL"] = disabled
                results[backend] = ef.fit_ewald_sphere_center(coordinates)
                samples = []
                for _ in range(args.repetitions):
                    start = perf_counter()
                    ef.fit_ewald_sphere_center(coordinates)
                    samples.append(perf_counter() - start)
                timings[backend] = median(samples)
            np.testing.assert_allclose(
                results["rust"], results["python"], rtol=1e-5, atol=1e-5
            )
            for backend, elapsed in timings.items():
                print(
                    f"{size},{backend},{elapsed * 1000:.6f},"
                    f"{timings['python'] / elapsed:.2f}",
                    flush=True,
                )
    finally:
        if saved is None:
            os.environ.pop("EMCFILE_DISABLE_DET_ACCEL", None)
        else:
            os.environ["EMCFILE_DISABLE_DET_ACCEL"] = saved


if __name__ == "__main__":
    main()
