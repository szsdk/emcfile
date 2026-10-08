# SZ-68 final slicing validation — 2026-10-01

The final implementation uses NumPy event-array views for contiguous row slices, pure NumPy gathers for strided/fancy/Boolean rows, and bounded int32 SciPy CSR for column selection. Combined selectors with at least one slice first view/gather only the requested rows. Other advanced selectors keep the general SciPy path. The optional HDF5 Numba dependency remains because the delta, shuffle and reader kernels still use it.

The custom production row-gather kernel and its dispatch were removed. The historical comparison loads the original kernel from an explicitly supplied external checkout at commit `0b47eff`; there is no production Numba row gather. Legacy HDF5 signed-int32 positions are converted to the public uint32 gather output. Both CSR indices and contiguous multi-count data use chunk-sized `memoryview`/`np.frombuffer` buffers to avoid SciPy's hidden pruning copies. Single-chunk results avoid a redundant concatenate.

## Method

Complete, previously validated raw EMC copies were loaded before timing:

| Dataset | Patterns | Pixels | Source arrays |
| --- | ---: | ---: | ---: |
| cube17 | 2,996,920 | 1,048,576 | 17.87 GiB |
| r0378 | 122,342 | 262,144 | 30.46 GiB |

Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1; Linux on a shared GPFS node. Three paired repetitions, reversing method order each repetition; medians reported. Source loading, historical Numba compilation, and exact-equality validation are excluded. Allocation, gathering/filtering, offsets, and returned `PatternsSOne` construction are included. The source arrays and offsets are read-only. RSS is sampled every 5 ms and at action completion; growth is relative to the loaded process immediately before each selection. Garbage collection and `malloc_trim` run outside timing. Small transient allocations may escape the RSS sampler.

Rows are validated by exact equality of all five EMC arrays against the historical allocation/gather kernel. Columns are validated against the adaptive full-source CSR path, including exact shape and every array. Random IDs use seed 65; sorted cases use the same IDs sorted, and `sorted20` is a sorted, uniform, no-replacement 20% sample. `boolean20` selects every fifth row. Duplicate/negative IDs are exercised explicitly. These are in-memory selection benchmarks, not file-I/O or cold-cache results.

## Dense/strided rows

Time is seconds; peak allocation is GiB above the loaded source.

| Dataset / selection | NumPy s | Historical Numba s | NumPy / Numba | NumPy peak GiB | Numba peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| cube17 / stride2 | 6.284 | 1.200 | 5.24× | 9.265 | 8.985 |
| cube17 / stride4 | 3.192 | 0.611 | 5.23× | 4.629 | 4.495 |
| cube17 / stride16 | 0.800 | 0.155 | 5.16× | 1.149 | 1.122 |
| cube17 / sorted20 | 2.119 | 0.468 | 4.53× | 3.667 | 3.591 |
| cube17 / boolean20 | 2.565 | 0.485 | 5.29× | 3.700 | 3.591 |
| r0378 / stride2 | 1.821 | 1.620 | 1.12× | 15.229 | 15.222 |
| r0378 / stride4 | 0.914 | 0.815 | 1.12× | 7.621 | 7.618 |
| r0378 / stride16 | 0.230 | 0.204 | 1.12× | 1.902 | 1.902 |
| r0378 / sorted20 | 0.708 | 0.646 | 1.10× | 6.075 | 6.073 |
| r0378 / boolean20 | 0.732 | 0.649 | 1.13× | 6.094 | 6.091 |

The previous “20–25% slower” estimate does not generalize to this run. NumPy is about 4.5–5.3× slower on dense cube17 selections and 10–13% slower on r0378. Cube17 has millions of smaller patterns, so per-range Python/NumPy view construction is a likely explanation for its larger penalty. The requested simpler NumPy architecture has a measurable speed trade-off; it is not a throughput-equivalent replacement for the warmed historical kernel. Absolute measured NumPy dense-selection times are at most 6.28 s, while peak memory remains close to the required output. Contiguous slices retain their zero-copy behavior.

## Sparse rows

Time is milliseconds; peak allocation is MiB above the loaded source.

| Dataset / selection | NumPy ms | Historical Numba ms | NumPy peak MiB | Numba peak MiB |
| --- | ---: | ---: | ---: | ---: |
| cube17 / random256 | 1.86 | 0.88 | 1.6 | 1.5 |
| cube17 / sorted256 | 1.79 | 0.89 | 1.6 | 1.5 |
| cube17 / random1000 | 6.04 | 2.45 | 6.3 | 6.2 |
| cube17 / sorted1000 | 5.99 | 2.43 | 6.3 | 6.2 |
| cube17 / repeated_negative | 1.69 | 0.78 | 1.5 | 1.5 |
| r0378 / random256 | 15.04 | 13.90 | 66.3 | 66.3 |
| r0378 / sorted256 | 15.38 | 13.96 | 66.3 | 66.3 |
| r0378 / random1000 | 39.53 | 35.88 | 257.2 | 257.1 |
| r0378 / sorted1000 | 39.50 | 35.75 | 257.2 | 257.1 |
| r0378 / repeated_negative | 13.82 | 13.17 | 67.7 | 67.7 |

The historical warmed kernel was faster on these sparse cases too, unlike the earlier SZ-65 sample. NumPy still selects 1,000 random rows in about 6 ms on cube17 and 40 ms on r0378, without a first-selection JIT compilation cost.

## Full-data columns and combined selection

These timings are the final rerun after the multi-count buffer-copy fix. Column filtering scans all source events; the default chunk budget is one billion events per stream. Peak allocations include SciPy's per-chunk work and output assembly, so this is not a strict 4 GiB process limit.

| Dataset / selection | Bounded s | Full-source CSR s | Bounded peak GiB | Full-source peak GiB |
| --- | ---: | ---: | ---: | ---: |
| cube17 / columns1024 | 4.4632 | 8.5813 | 3.831 | 40.889 |
| cube17 / combined1000 | 0.0076 | 3.8708 | 0.013 | 40.853 |
| r0378 / columns1024 | 6.3485 | 12.4835 | 3.813 | 54.539 |
| r0378 / combined1000 | 0.0893 | 5.2731 | 0.399 | 54.831 |

`columns1024` means all rows and the first 1,024 pixels. `combined1000` means the same 1,000 random IDs used above and the first 1,024 pixels. The source-wide CSR baseline already includes SZ-65's adaptive int32 optimization where its total offsets permit it.

For full columns, bounded CSR cuts temporary memory from roughly 41–55 GiB to 3.8 GiB and is about twice as fast in these paired measurements. Combined random-row/column selection avoids the unrelated source payload and sharply reduces both allocation and runtime. Bounds are limited by output size, chunk budget, and exceptional individual rows; wide pixel spaces or incompatible buffers use the general selected-source CSR fallback.

## Contiguous slicing scaling and regression checks

A separate 15-repetition synthetic benchmark holds a 16-row selection constant while source rows increase:

| Source rows | View slice median ms | CSR slice median ms | View traced peak bytes | CSR traced peak bytes |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 0.00909 | 0.13077 | 2,363 | 38,856 |
| 100,000 | 0.00901 | 0.22439 | 2,414 | 3,602,776 |
| 1,000,000 | 0.00895 | 1.83471 | 2,363 | 36,002,776 |

The view path scales with selected rows rather than unrelated source events. These traced allocation numbers are separate from the sampled full-data RSS measurements.

Validation: **280 passed, 8 skipped**; Ruff and `git diff --check` pass. Regression coverage includes contiguous sharing, arbitrary row order/duplicates/negative IDs, empty rows/events, Boolean masks, reverse/strided columns, repeated/negative columns, signed legacy positions, read-only buffers, forced small chunks, invalid selectors even for empty selections, and CSR index/data sharing. The HDF5 codec, full-scan/indexed and writer tests remain passing.

SZ-66 can use this contiguous-view path when writing a materialized source, but should still implement file-backed streaming conversion. SZ-67 can reuse `benchmarks/contiguous_slice.py` and the full-data measurement/equality/RSS harness; this issue does not claim to complete its separate HDF5 write-scaling benchmark.

## Reproduction

Use a scratch-backed environment and cache. Raw trial records and summaries are in [sz68_results.json](sz68_results.json).

```bash
git worktree add /scratch/emcfile-sz65-reference 0b47eff
export PYTHONPATH=src
export NUMBA_CACHE_DIR=/scratch/numba-cache
python benchmarks/sz68_slicing.py FULL_CUBE17.emc FULL_R0378.emc \
  --reference-checkout /scratch/emcfile-sz65-reference --repetitions 3
python benchmarks/sz68_slicing.py FULL_CUBE17.emc FULL_R0378.emc \
  --cases columns1024 combined1000 --repetitions 3
python benchmarks/contiguous_slice.py --repetitions 15
pytest -q
ruff check .
```

The committed row trial records come from the full paired run; column trial records come from the final buffer-copy rerun. The interrupted initial validation run and earlier column measurements are excluded.
