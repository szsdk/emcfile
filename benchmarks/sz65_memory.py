"""Measure SZ-65 selection RSS in isolated processes on Linux.

Run with ``PYTHONPATH=src python benchmarks/sz65_memory.py DATA.emc``.
Each (case, implementation) loads the source independently. RSS before the
selection excludes file loading; peak growth is the increase in Linux's
high-water RSS during selection, and held growth keeps the result alive.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
from pathlib import Path

from psutil import Process

from emcfile import patterns

from sz65_real_slicing import legacy_select, make_cases


def rss() -> int:
    return Process().memory_info().rss


def high_water_rss() -> int:
    # ru_maxrss is KiB on Linux, the platform used for this benchmark.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def child(path: str, case: str, method: str) -> None:
    source = patterns(path)
    selector = make_cases(len(source))[case]
    gc.collect()
    baseline_rss = rss()
    baseline_hwm = high_water_rss()
    selected = legacy_select(source, selector) if method == "old" else source[selector]
    held_rss = rss()
    peak_hwm = high_water_rss()
    result_bytes = selected.nbytes
    rows_selected = len(selected)
    del selected
    gc.collect()
    print(
        json.dumps(
            {
                "case": case,
                "method": method,
                "source_bytes": source.nbytes,
                "rows_selected": rows_selected,
                "result_bytes": result_bytes,
                "baseline_rss": baseline_rss,
                "held_rss": held_rss,
                "baseline_hwm": baseline_hwm,
                "peak_hwm": peak_hwm,
                "post_release_rss": rss(),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "contiguous_16",
            "stride_2_256",
            "random_ids_256",
            "random_ids_1000",
            "sorted_ids_20pct",
        ],
    )
    parser.add_argument("--child", nargs=2, metavar=("CASE", "METHOD"))
    parser.add_argument(
        "--methods", nargs="+", choices=("old", "new"), default=("old", "new")
    )
    args = parser.parse_args()
    if args.child:
        child(args.path, *args.child)
        return

    print(
        "case,method,source_mib,result_mib,baseline_rss_mib,held_growth_mib,"
        "peak_growth_mib,post_release_growth_mib",
        flush=True,
    )
    for case in args.cases:
        for method in args.methods:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                args.path,
                "--child",
                case,
                method,
            ]
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            data = json.loads(completed.stdout)
            mb = 1 << 20
            print(
                f"{case},{method},{data['source_bytes'] / mb:.1f},"
                f"{data['result_bytes'] / mb:.1f},{data['baseline_rss'] / mb:.1f},"
                f"{(data['held_rss'] - data['baseline_rss']) / mb:.1f},"
                f"{max(0, data['peak_hwm'] - data['baseline_hwm']) / mb:.1f},"
                f"{(data['post_release_rss'] - data['baseline_rss']) / mb:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
