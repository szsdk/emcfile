# emcfile

`emcfile` reads, writes, and manipulates sparse photon-count patterns and
detector geometry used by EMC workflows. It provides NumPy-compatible pattern
arrays, file-backed access to large datasets, detector masks and coordinates,
and rendering of detector data onto a regular image grid.

## Installation

`emcfile` requires Python 3.10 or later.

```bash
pip install emcfile
```

## Quick start

```python
import numpy as np
import emcfile as ef

# Rows are patterns; columns are detector pixels. Photon counts must be integers.
dense = np.array(
    [
        [0, 1, 0, 3],
        [1, 0, 2, 0],
    ],
    dtype=np.int32,
)

patterns = ef.patterns(dense)
assert isinstance(patterns, ef.EMCPatternArray)
assert patterns.shape == (2, 4)
assert patterns.num_patterns == 2
assert patterns.num_pixels == 4

# Integer indexing returns one dense pattern. Slicing remains sparse.
first_pattern = patterns[0]
subset = patterns[:1]

patterns.write("patterns.emc", overwrite=True)
restored = ef.patterns("patterns.emc")
assert restored == patterns
```

## Mental model

An EMC pattern dataset is a two-dimensional integer array with shape
`(num_patterns, num_pixels)`. Each row is one diffraction pattern, and each
column corresponds to one detector pixel.

The EMC sparse representation treats pixels with value `1` specially:

- `ones` stores how many one-photon pixels occur in each pattern.
- `place_ones` stores their pixel indices.
- `multi` stores how many pixels with values greater than one occur in each
  pattern.
- `place_multi` and `count_multi` store those pixel indices and photon counts.

This `ones` terminology is part of the EMC format and is intentionally retained
throughout the API.

## Choose the right entry point

| Task | API | Result |
| --- | --- | --- |
| Convert an array or load an entire file | `ef.patterns(source)` | `EMCPatternArray` in memory |
| Open a pattern file without loading it all | `ef.open_patterns(path)` | File-backed pattern source |
| Accumulate patterns incrementally | `ef.EMCPatternCollector(...)` | Batched collector |
| Combine several pattern sources | `ef.EMCPatternCollection(...)` | One logical pattern sequence |
| Create or load detector geometry | `ef.detector(...)` | `Detector` |
| Render detector values on an image grid | `ef.detector_renderer(det)` | `DetectorRenderer` |

Use `patterns()` for small or repeatedly accessed datasets. Use
`open_patterns()` when a dataset is too large to load eagerly or when only a
subset is needed.

```python
source = ef.open_patterns("patterns.emc")
print(source.shape)

# Only the selected patterns are materialized in memory.
batch = source[:1]
```

## Pattern arrays

`EMCPatternArray` supports common NumPy and SciPy workflows:

```python
# Convert representations.
dense = patterns.todense()
csr = patterns.tocsr()

# Select patterns or pixels.
one_pattern = patterns[0]
every_other_pattern = patterns[::2]
selected_pixels = patterns[:, [0, 2]]

# Aggregate or combine data.
counts_per_pattern = patterns.sum(axis=1)
counts_per_pixel = patterns.sum(axis=0)
combined = np.concatenate([patterns, patterns])

# Matrix multiplication does not require a dense intermediate array.
projection = patterns @ np.ones((patterns.num_pixels, 3))
```

To write several arrays or file-backed sources as one dataset, use
`write_patterns()`:

```python
ef.write_patterns(
    [source[:1], source[1:]],
    "selection.h5::/patterns",
    overwrite=True,
)
```

### Incremental collection

`EMCPatternCollector` is useful when patterns arrive one at a time or in small
batches:

```python
collector = ef.EMCPatternCollector(batch_size=128)
collector.append(np.array([0, 1, 0, 2], dtype=np.int32))
collector.extend(np.zeros((10, 4), dtype=np.int32))

patterns = collector.to_patterns()
collector.write("collected.emc", overwrite=True)
```

## Detector geometry

A `Detector` stores per-pixel coordinates, correction factors, and mask values,
plus the detector distance and Ewald radius. The short variable name `det` is
used conventionally for detector objects.

```python
det = ef.detector(
    coordinates=(64, 64),
    detector_distance=100.0,
)

print(det.coordinates.shape)
print(det.correction_factors.shape)
print(det.mask.shape)

good_pixels = det.mask == ef.PixelType.GOOD
good_det = det[good_pixels]
```

For complete geometry data, pass `coordinates`, `mask`, `correction_factors`,
`detector_distance`, and `ewald_radius`. Detector files can also be loaded
directly:

```python
det = ef.detector("detector.dat", normalize=False)
det.write("detector.h5", overwrite=True)
```

### Rendering detector values

```python
renderer = ef.detector_renderer(det)
pixel_values = np.arange(det.num_pix, dtype=np.float64)
image = renderer.render(pixel_values)

# A batch with shape (num_patterns, num_pixels) produces a stack of images.
images = renderer.render(np.stack([pixel_values, pixel_values]))
```

Pixels not covered by the rendered detector remain `NaN` in the underlying
rendered data.

## File formats and paths

| Data | Supported formats |
| --- | --- |
| Photon patterns | EMC binary (`.emc`), HDF5 (`.h5`) |
| Detector geometry | ASCII (`.dat`), HDF5 (`.h5`) |
| NumPy arrays | NumPy (`.npy`), raw binary, HDF5 |

An HDF5 object path uses the form `file.h5::/path/inside/file`:

```python
path = ef.as_hdf5_path("experiment.h5::/patterns")
patterns.write(path, overwrite=True)
restored = ef.patterns(path)
```

HDF5 version 2 is the default pattern layout. Version 1 remains readable and
can be written explicitly for interoperability:

```python
patterns.write(
    "legacy.h5::/patterns",
    hdf5_version="1",
    overwrite=True,
)
```

General arrays and nested Python dictionaries can be stored with the HDF5
helpers:

```python
ef.write_array("data.h5::/weights", np.arange(5), overwrite=True)
weights = ef.read_array("data.h5::/weights")

metadata = {"run": 42, "energy": 9.5}
ef.write_hdf5_object("data.h5::/metadata", metadata, overwrite=True)
restored_metadata = ef.read_hdf5_object("data.h5::/metadata")
```

## API summary

- Pattern creation and conversion: `patterns`, `EMCPatternArray`
- File-backed patterns: `open_patterns`, `FileBackedEMCPatterns`
- Pattern output: `EMCPatternArray.write`, `write_patterns`
- Streaming collection: `EMCPatternCollector`
- Combined sources: `EMCPatternCollection`
- Detector geometry: `detector`, `Detector`, `PixelType`
- Detector rendering: `detector_renderer`, `DetectorRenderer`
- HDF5 paths: `as_hdf5_path`, `as_path`, `H5Path`
- Generic data I/O: `read_array`, `write_array`, `read_hdf5_object`,
  `write_hdf5_object`

## Tutorial

The [tutorial directory](tutorial) contains guided, interactive Marimo
notebooks. From the repository root, install the development dependencies and
open the tutorial browser with:

```bash
uv sync --group dev
uv run marimo edit tutorial
```

Marimo starts a local server and opens a page where you can choose any tutorial.
Run or edit cells in the browser; dependent cells update automatically. Press
`Ctrl+C` in the terminal to stop the server.

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy src/emcfile
uv run pre-commit run --all-files
```
