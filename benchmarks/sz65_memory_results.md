# SZ-65 selection memory, 2026-09-30

Measured with `benchmarks/sz65_memory.py` on two existing raw EMC subsets:

- `sz54_results/cub17-100000.emc`: 100,000 rows; 559.1 MiB logical EMC arrays.
- `sz54_results/r0378-8000.emc`: 8,000 rows; 2,086.2 MiB logical EMC arrays.

Each case and implementation ran in a fresh process. The entire EMC file was
loaded before measurement. `baseline RSS` was recorded after loading and, for
the accelerated implementation, after warming Numba. `peak growth` is the
increase in Linux high-water RSS during one selection. `result size` is the
logical size of the selected EMC arrays, **not** necessarily additional RSS:
contiguous selection keeps event-array views, and allocators may reuse pages.
All sizes below are MiB (2²⁰ bytes). Values are one-run observations on a
shared node, not a memory-cap guarantee.

| Dataset | Selection | Path | Result size | Baseline RSS | Peak growth | Held-result growth |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cube17 | 16 contiguous | old CSR | 0.1 | 631.0 | 1,296.2 | 0.1 |
| cube17 | 16 contiguous | current | 0.1 | 631.0 | 0.0 | 0.0 |
| cube17 | 256 stride-2 | old CSR | 1.5 | 631.2 | 1,296.4 | 0.1 |
| cube17 | 256 stride-2 | current + Numba | 1.5 | 687.9 | 1.0 | 0.0 |
| cube17 | 256 random IDs | old | 1.4 | 631.0 | 0.0 | 0.1 |
| cube17 | 256 random IDs | current + Numba | 1.4 | 687.8 | 0.0 | 0.0 |
| cube17 | 1,000 random IDs | old | 5.6 | 631.1 | 0.0 | 0.1 |
| cube17 | 1,000 random IDs | current + Numba | 5.6 | 687.8 | 0.0 | 0.0 |
| cube17 | sorted 20% | old | 111.9 | 631.0 | 96.6 | 97.4 |
| cube17 | sorted 20% | current + Numba | 111.9 | 687.9 | 104.6 | 105.0 |
| r0378 | 16 contiguous | old CSR | 4.1 | 2,157.8 | 3,749.6 | 0.8 |
| r0378 | 16 contiguous | current | 4.1 | 2,159.2 | 0.0 | 0.0 |
| r0378 | 256 stride-2 | old CSR | 68.6 | 2,159.0 | 3,874.5 | 52.7 |
| r0378 | 256 stride-2 | current + Numba | 68.6 | 2,214.4 | 60.9 | 60.6 |
| r0378 | 256 stride-2 | current, no Numba | 68.6 | 2,157.9 | 53.8 | 52.8 |
| r0378 | 256 random IDs | old | 66.8 | 2,158.7 | 50.1 | 50.8 |
| r0378 | 256 random IDs | current + Numba | 66.8 | 2,214.4 | 58.7 | 58.5 |
| r0378 | 256 random IDs | current, no Numba | 66.8 | 2,157.7 | 50.9 | 50.9 |
| r0378 | 1,000 random IDs | old | 261.6 | 2,157.8 | 261.0 | 261.7 |
| r0378 | 1,000 random IDs | current + Numba | 261.6 | 2,214.5 | 261.5 | 261.6 |
| r0378 | 1,000 random IDs | current, no Numba | 261.6 | 2,158.2 | 262.0 | 261.7 |
| r0378 | sorted 20% | old | 416.3 | 2,157.8 | 416.7 | 416.5 |
| r0378 | sorted 20% | current + Numba | 416.3 | 2,214.4 | 416.4 | 416.3 |

The direct row path eliminates the full-source CSR temporary. On r0378, the
256-row stride case drops from roughly 5.89 GiB total process peak RSS
(2,159.0 + 3,874.5 MiB) to roughly 2.22 GiB with Numba, or 2.16 GiB without
Numba. Random selections already avoided CSR, so their output copy dominates
memory and the accelerated path does not reduce it. Loading/warming Numba adds
roughly 56 MiB to process baseline RSS in these runs. The no-Numba fallback
removes that baseline cost while retaining the stride-slice memory fix.

Reproduce from the repository root with `PYTHONPATH=src` and a scratch-backed
`NUMBA_CACHE_DIR`:

```bash
python benchmarks/sz65_memory.py /path/to/dataset.emc
python benchmarks/sz65_memory.py /path/to/dataset.emc --disable-numba --methods new --cases stride_2_256 random_ids_256 random_ids_1000
```
