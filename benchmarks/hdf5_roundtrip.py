"""Real-data writer/full-scan smoke benchmark, excluding source materialization."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import emcfile as ef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--patterns", type=int, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    expected = ef.open_patterns(args.raw)[: args.patterns]
    os.environ["EMCFILE_H5_WRITE_WORKERS"] = "4"
    with tempfile.TemporaryDirectory(
        prefix="hdf5-roundtrip-", dir=args.scratch
    ) as directory:
        for extension in ("emc", "h5"):
            path = Path(directory) / f"patterns.{extension}"
            options = (
                {}
                if extension == "emc"
                else dict(
                    position_encoding="delta",
                    compression="zstd",
                    compression_opts=1,
                    shuffle=True,
                )
            )
            start = time.perf_counter()
            expected.write(path, **options)
            writing = time.perf_counter() - start
            for workers in (0,) if extension == "emc" else (4, 0):
                os.environ["EMCFILE_H5_FULL_SCAN_WORKERS"] = str(workers)
                start = time.perf_counter()
                actual = ef.open_patterns(path)[:]
                reading = time.perf_counter() - start
                assert actual == expected
                print(
                    json.dumps(
                        {
                            "raw": args.raw,
                            "patterns": args.patterns,
                            "logical_bytes": expected.nbytes,
                            "format": extension,
                            "full_scan_workers": workers,
                            "write_s": writing,
                            "full_scan_s": reading,
                            "file_bytes": path.stat().st_size,
                            "exact_equal": True,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
