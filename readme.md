# emcfile

`emcfile` is a Python package providing essential utilities for handling the `emc` photon format and its associated detector files. It offers a streamlined interface for reading, writing, and manipulating diffraction patterns and detector geometry, with a focus on performance and ease of use.

## Key Features

- **Pattern Manipulation**: Create, access, and modify diffraction patterns with a `numpy`-like API.
- **Detector Geometry**: Define and manage detector geometry, including pixel coordinates, masks, and scaling factors.
- **File I/O**: Read and write patterns and detectors from/to various file formats, including the `.emc` format, text-based `.dat` files, and HDF5.
- **HDF5 Integration**: A suite of helper functions to simplify interactions with HDF5 files, allowing you to easily store and retrieve Python objects and NumPy arrays.
- **Visualization**: Render detector data for quick visualization and analysis.

## Installation

You can install `emcfile` using pip:

```bash
pip install emcfile
```

**Dependencies:**
- python >= 3.9
- numpy >= 1.22
- h5py
- scipy

## Quick Start

Here's a brief example of how to use `emcfile` to create and save a `patterns` object:

```python
import numpy as np
import emcfile as ef

# Create random patterns
num_data = 5
num_pix = 10
patterns_data = np.random.rand(num_data, num_pix) ** 3 * 5
patterns = ef.patterns(patterns_data.astype("int"))

# Write patterns to an .emc file
patterns.write("test_pattern.emc", overwrite=True)

# Read patterns from the .emc file
p_emc = ef.patterns("test_pattern.emc")
print(p_emc)
```

## Documentation

For a comprehensive guide to using the package, please see the [tutorial](tutorial/tutorial.md).

## Conventions

### Patterns
For a pattern object, its first index is always the pattern index.
