# SZ-65: chunk-wise int32 CSR for full-file column selection

This experiment selects all rows and the first 1,024 columns from complete
raw EMC datasets. `patterns(path)` is fully loaded before timing. Each mode
and event budget runs in a fresh process; the peak is Linux `ru_maxrss` and
growth subtracts RSS immediately after loading. Wall times are single runs
on a shared node, **not** stable throughput estimates. File-read time is
excluded.

| Dataset | Path | Event budget | Chunks | Time | Peak RSS above loaded source |
| --- | --- | ---: | ---: | ---: | ---: |
| cube17 | old full-file CSR | — | 1 | 92.53 s | 40.85 GiB |
| cube17 | adaptive full-file CSR | — | 1 | 8.58 s | 40.85 GiB |
| cube17 | chunked int32 CSR | 500M | 8 | 32.37 s | 1.86 GiB |
| cube17 | chunked int32 CSR | 1,000M | 4 | 36.19 s | 3.72 GiB |
| cube17 | chunked int32 CSR | 2,000M | 2 | 31.40 s | 7.45 GiB |
| cube17 | integrated public API | 1,000M | 4 | 30.35 s | 3.72 GiB |
| r0378 | old full-file CSR | — | 1 | 138.99 s | 54.64 GiB |
| r0378 | adaptive full-file CSR | — | 1 | 113.63 s | 54.64 GiB |
| r0378 | chunked int32 CSR | 500M | 10 | 52.33 s | 1.93 GiB |
| r0378 | chunked int32 CSR | 1,000M | 5 | 7.04 s | 3.81 GiB |
| r0378 | chunked int32 CSR | 2,000M | 3 | 53.57 s | 7.56 GiB |
| r0378 | integrated public API | 1,000M | 5 | 14.45 s | 3.80 GiB |

The full source baselines were ~18.03 GiB RSS for cube17 and ~30.53 GiB
RSS for r0378. Selected outputs were 27.56 MiB and 62.40 MiB respectively.
The 1,000M-budget prototype and integrated public API were each checked
for exact equality against the adaptive CSR path on **both complete
datasets**. Earlier paired single-pass comparisons also verified equality
of old and adaptive results.

The wide full-file ones CSR requires int64 indices, which copies its entire
event-index array. Chunking keeps each ones and multi `indptr` within signed
int32. There is an important SciPy detail: a normal NumPy slice retains the
full event array as its base, and SciPy's `_prune_array` deliberately copies
views smaller than half their base, even with `copy=False`. Creating a
chunk-sized `memoryview` and then `np.frombuffer(..., dtype=np.int32)` avoids
this copy; the benchmark asserts that large chunk CSR indices share memory
with the source and remain int32. The compact, rebased int32 `indptr` does
allocate four bytes per selected row.

The memory benefit is large and consistent. Timing is highly variable on
this shared node: even the adaptive cube17 path was 87.17 s in the earlier
paired run but 8.58 s in the isolated RSS run. The table should not be used
to claim a precise speedup or optimal event budget. A 500M–1,000M budget
keeps temporary RSS to approximately 2–4 GiB; the remaining work is
SciPy's column filter over all events and assembling the selected output.

Reproduce with `PYTHONPATH=src:benchmarks`:

```bash
python benchmarks/sz65_chunked_columns.py DATA.emc --mode old
python benchmarks/sz65_chunked_columns.py DATA.emc --mode adaptive
python benchmarks/sz65_chunked_columns.py DATA.emc --mode chunked --budget 1000000000 --verify
python benchmarks/sz65_chunked_columns.py DATA.emc --mode integrated --verify
```
