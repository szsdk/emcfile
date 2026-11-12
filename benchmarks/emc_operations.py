"""Benchmark Rust-accelerated vs pure-Python/SciPy operations on .emc files and EMCPatternArray.

Run with ``uv run --extra fast python benchmarks/emc_operations.py``.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import numpy as np

import emcfile as ef
from emcfile._emc_accel import get_emc_rs, invalidate_emc_cache_for_path


def _bench(
    fn: Callable[[], Any],
    *,
    reps: int,
    setup: Callable[[], None] | None = None,
) -> float:
    if setup is not None:
        setup()
    fn()  # warmup
    samples: list[float] = []
    for _ in range(reps):
        if setup is not None:
            setup()
        t0 = perf_counter()
        fn()
        samples.append(perf_counter() - t0)
    return median(samples)


def _read_100_persistent(emc_path: Path, n_pat: int) -> None:
    with ef.open_patterns(emc_path).open() as reader:
        for i in range(100):
            _ = reader[i * 17 % n_pat]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", type=int, default=4_000)
    parser.add_argument("--pixels", type=int, default=16_384)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    saved_disable = os.environ.pop("EMCFILE_DISABLE_EMC_ACCEL", None)
    saved_workers = os.environ.get("EMCFILE_EMC_WORKERS")
    try:
        if get_emc_rs() is None:
            parser.error("emcfile-fast with emc submodule is required")
        os.environ["EMCFILE_EMC_WORKERS"] = str(args.workers)

        rng = np.random.default_rng(42)
        n_pat = args.patterns
        n_pix = args.pixels
        dense = np.zeros((n_pat, n_pix), dtype=np.int32)
        ones_cols = rng.integers(0, n_pix, size=(n_pat, 320))
        multi_cols = rng.integers(0, n_pix, size=(n_pat, 80))
        row_grid = np.arange(n_pat)[:, None]
        dense[row_grid, ones_cols] = 1
        dense[row_grid, multi_cols] = rng.integers(
            2, 16, size=(n_pat, 80), dtype=np.int32
        )
        coo = ef.patterns(dense).tocsr().tocoo()
        csr = coo.tocsr()

        pat = ef.patterns(dense)
        rhs_mat = rng.standard_normal((n_pix, 16), dtype=np.float64)
        row_ids = rng.integers(0, n_pat, size=n_pat // 2, dtype=np.int64)
        col_mask = rng.random(n_pix) < 0.25
        col_ids = rng.integers(0, n_pix, size=n_pix // 4, dtype=np.int64)
        rev_cols = np.arange(n_pix - 1, -1, -1, dtype=np.int64)
        unsorted_template = pat[:, rev_cols]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            emc_path = tmp_dir / "bench.emc"
            write_path = tmp_dir / "write_out.emc"
            pat.write(emc_path, overwrite=True)

            cases: list[tuple[str, Callable[[], Any], Callable[[], None] | None]] = [
                (
                    "emc_file_full_read",
                    lambda: ef.patterns(emc_path),
                    lambda: invalidate_emc_cache_for_path(emc_path),
                ),
                (
                    "emc_file_single_frame_x100",
                    lambda: [
                        ef.open_patterns(emc_path)[i * 17 % n_pat] for i in range(100)
                    ],
                    None,
                ),
                (
                    "emc_open_reader_frame_x100",
                    lambda: _read_100_persistent(emc_path, n_pat),
                    None,
                ),
                (
                    "emc_file_indexed_rows",
                    lambda: ef.open_patterns(emc_path)[row_ids],
                    None,
                ),
                (
                    "emc_file_extract_column",
                    lambda: ef.open_patterns(emc_path)[:, n_pix // 2],
                    None,
                ),
                (
                    "emc_file_sum_axis0",
                    lambda: ef.open_patterns(emc_path).sum(axis=0),
                    None,
                ),
                (
                    "emc_file_write",
                    lambda: pat.write(write_path, overwrite=True),
                    None,
                ),
                ("inmem_sum_total", lambda: pat.sum(), None),
                ("inmem_sum_axis0", lambda: pat.sum(axis=0), None),
                ("inmem_sum_axis1", lambda: pat.sum(axis=1), None),
                ("inmem_matmul_spmm16", lambda: pat @ rhs_mat, None),
                ("inmem_todense", lambda: pat.todense(), None),
                ("inmem_gather_rows", lambda: pat[row_ids], None),
                ("inmem_select_cols_mask", lambda: pat[:, col_mask], None),
                ("inmem_select_cols_fancy", lambda: pat[:, col_ids], None),
                ("inmem_extract_col", lambda: pat[:, n_pix // 3], None),
                ("convert_from_dense", lambda: ef.patterns(dense), None),
                ("convert_from_coo", lambda: ef.patterns(coo), None),
                ("convert_from_csr", lambda: ef.patterns(csr), None),
            ]

            print(
                f"# shape=({n_pat}, {n_pix}), "
                f"events={len(pat.place_ones) + len(pat.place_multi):,}, "
                f"workers={args.workers}"
            )
            print(f"{'operation':<28} {'python_ms':>11} {'rust_ms':>11} {'speedup':>9}")
            print("-" * 63)

            for name, fn, setup in cases:
                os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = "1"
                invalidate_emc_cache_for_path(emc_path)
                t_py = _bench(fn, reps=args.repetitions, setup=setup)

                os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = "0"
                invalidate_emc_cache_for_path(emc_path)
                t_rs = _bench(fn, reps=args.repetitions, setup=setup)

                print(
                    f"{name:<28} {t_py * 1000:11.3f} {t_rs * 1000:11.3f} "
                    f"{t_py / max(t_rs, 1e-12):8.2f}x",
                    flush=True,
                )

            # In-place sort_indices benchmark (fresh unsorted copy each iteration)
            work: list[ef.EMCPatternArray] = []

            def _prep_unsorted() -> None:
                work.clear()
                work.append(
                    ef.PatternsSOne(
                        unsorted_template.num_pix,
                        unsorted_template.ones.copy(),
                        unsorted_template.multi.copy(),
                        unsorted_template.place_ones.copy(),
                        unsorted_template.place_multi.copy(),
                        unsorted_template.count_multi.copy(),
                    )
                )

            os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = "1"
            t_sort_py = _bench(
                lambda: work[0].sort_indices(),
                reps=args.repetitions,
                setup=_prep_unsorted,
            )
            os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = "0"
            t_sort_rs = _bench(
                lambda: work[0].sort_indices(),
                reps=args.repetitions,
                setup=_prep_unsorted,
            )
            print(
                f"{'inmem_sort_indices':<28} {t_sort_py * 1000:11.3f} "
                f"{t_sort_rs * 1000:11.3f} {t_sort_py / max(t_sort_rs, 1e-12):8.2f}x",
                flush=True,
            )
    finally:
        if saved_disable is None:
            os.environ.pop("EMCFILE_DISABLE_EMC_ACCEL", None)
        else:
            os.environ["EMCFILE_DISABLE_EMC_ACCEL"] = saved_disable
        if saved_workers is None:
            os.environ.pop("EMCFILE_EMC_WORKERS", None)
        else:
            os.environ["EMCFILE_EMC_WORKERS"] = saved_workers


if __name__ == "__main__":
    main()
