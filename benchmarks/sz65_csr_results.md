# SZ-65 follow-up: CSR buffer sharing

The experiment compares three input dtype combinations when constructing
SciPy `csr_array(..., copy=False)` from an in-memory `PatternsSOne`. Results
were measured with SciPy 1.18.1 and NumPy 2.5.3 on two real EMC subsets.
File loading is excluded. Times are medians from five runs for CSR
construction and five runs for cube17 column selection (three for r0378).
Allocation peaks use `tracemalloc` and include NumPy array allocations; they
are not process RSS. Shared-node timing can vary.

## Construction

| Dataset / component | Input indices / indptr | Actual CSR indices / indptr | Shares event indices? | Shares source offsets? | Shares data? | Median time | Allocation peak |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| cube17 / multi | uint32 / uint64 | int64 / int64 | No | No | Yes | 12.43 ms | 126.93 MiB |
| cube17 / multi | int32 / int32 | int32 / int32 | Yes | No¹ | Yes | 0.051 ms | 0.38 MiB |
| cube17 / multi | int32 / int64 | int64 / int64 | No | Yes | Yes | 12.34 ms | 126.16 MiB |
| cube17 / ones | uint32 / uint64 | int64 / int64 | No | No | Yes² | 834.80 ms | 865.07 MiB |
| cube17 / ones | int32 / int32 | int32 / int32 | Yes | No¹ | Yes² | 0.065 ms | 0.38 MiB |
| cube17 / ones | int32 / int64 | int64 / int64 | No | Yes | Yes² | 827.66 ms | 864.30 MiB |
| r0378 / multi | uint32 / uint64 | int64 / int64 | No | No | Yes | 1,555.54 ms | 836.55 MiB |
| r0378 / multi | int32 / int32 | int32 / int32 | Yes | No¹ | Yes | 0.032 ms | 0.03 MiB |
| r0378 / multi | int32 / int64 | int64 / int64 | No | Yes | Yes | 1,556.01 ms | 836.49 MiB |
| r0378 / ones | uint32 / uint64 | int64 / int64 | No | No | Yes² | 4,718.48 ms | 2,499.35 MiB |
| r0378 / ones | int32 / int32 | int32 / int32 | Yes | No¹ | Yes² | 0.046 ms | 0.03 MiB |
| r0378 / ones | int32 / int64 | int64 / int64 | No | Yes | Yes² | 4,692.67 ms | 2,499.29 MiB |

¹ A compact int32 offset array is allocated from the source's uint64 offsets,
then shared with SciPy's CSR. It costs only four bytes per pattern plus one.

² The ones CSR data shares the existing zero-stride array of ones. SciPy
retains stride zero with `copy=False`; it does not materialize a source-sized
data buffer during construction. This behavior was already present with the
old unsigned inputs.

The `int32 / int64` combination shares the uint64 source offsets through an
int64 view, but SciPy promotes the event indices to int64 and copies them.
Using int32 for both indices and offsets is the only tested combination that
shares the source event indices and avoids source-sized construction copies.

## Column-selection effect

Selecting all rows and the first 1,024 columns includes both CSR builds and
SciPy's column-slicing work. Old and new results were checked for exact
equality before timing.

| Dataset | Old time | New time | Old allocation peak | New allocation peak |
| --- | ---: | ---: | ---: | ---: |
| cube17 | 1,863.88 ms | 139.02 ms | 1,298.43 MiB | 433.21 MiB |
| r0378 | 8,781.37 ms | 2,432.05 ms | 3,759.53 MiB | 1,256.69 MiB |

The remaining allocation comes primarily from SciPy's column-selection work
over the full event arrays. This change removes the separate CSR-construction
copies. The implementation uses signed 32-bit buffers only when the dataset
shape and total event count fit signed int32; wider cases retain SciPy's
general construction path.

Reproduce from the repository root with `PYTHONPATH=src`:

```bash
python benchmarks/sz65_csr_buffers.py /path/to/dataset.emc
python benchmarks/sz65_csr_columns.py /path/to/dataset.emc
```
