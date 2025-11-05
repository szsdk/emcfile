# Tutorial

This tutorial provides a hands-on introduction to the `emcfile` package. You will learn how to create, manipulate, and store pattern and detector data, as well as how to use the HDF5 helper functions.

First, let's import the necessary libraries. We'll need `numpy` for creating data, `matplotlib` for plotting, and of course `emcfile`.

```python
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
import emcfile as ef
```

## Patterns

The `emcfile.patterns` object is a core component of the package, designed to handle diffraction patterns.

### Generate patterns

Let's start by generating some random patterns. We'll create a `numpy` array and then convert it into a `patterns` object.

```python
# Define number of data and pixels
num_data = 5
num_pix = 10

# Create random patterns
patterns_data = np.random.rand(num_data, num_pix) ** 3 * 5
patterns = ef.patterns(patterns_data.astype("int"))
print(patterns)
```

The `patterns` object has several useful attributes for inspecting your data, such as `shape`, `num_data`, and `num_pix`.

```python
# Shape of patterns
print(patterns.shape)

# Number of data and pixels in patterns
print(patterns.num_data, patterns.num_pix)
```

It also provides attributes for analyzing the photon counts in the patterns:
- `ones`: indices of single-photon pixels.
- `multi`: indices of multi-photon pixels.
- `place_ones`: positions of single-photon events.
- `place_multi`: positions of multi-photon events.
- `count_multi`: photon counts for multi-photon events.

```python
print(patterns.ones)
print(patterns.multi)
print(patterns.place_ones)
print(patterns.place_multi)
print(patterns.count_multi)
```

You can access individual or multiple patterns using standard numpy-like slicing.

```python
# Accessing a single pattern
print(patterns[1])

# Accessing multiple patterns
print(patterns[:2])

# Accessing specific pixels of multiple patterns
print(patterns[:, :2])
```

### Write and read patterns

`emcfile` supports several file formats for storing patterns, including the `.emc` format and HDF5 files.

```python
# Write patterns to .emc file
patterns.write("test_pattern.emc", overwrite=True)

# Read patterns from .emc file
p_emc = ef.patterns("test_pattern.emc")
print(p_emc)
```

You can also store patterns in HDF5 files. This is particularly useful for larger datasets or when you need to group data. Here we concatenate two `patterns` objects and save them to a `.h5` file.

```python
# Concatenate two patterns and write the result to group 'patterns' in a .h5 file
np.concatenate([p_emc] * 2).write("test_pattern.h5::patterns", overwrite=True)

# Read patterns from the .h5 file
p_h5 = ef.patterns("test_pattern.h5::patterns")
assert p_h5.num_data == 2 * p_emc.num_data
print(p_h5)
```

## Detector

The `emcfile.detector` object stores the geometry and properties of a detector.

### Generate a detector

Let's create a detector. We need to define the coordinates of its pixels, a mask for pixel types (e.g., good, bad, corner), and other physical parameters.

```python
# Ewald radius in pixels
ewald_rad = 128

# 2D coordinate of pixels
coor2d = np.mgrid[-32:33, -32:33].reshape(2, -1).T.astype(np.float64)
num_pix = len(coor2d)

# 3D coordinate of pixels
coor = np.zeros((num_pix, 3))
coor[:, :2] = coor2d
r2d = np.linalg.norm(coor2d, axis=1)
coor[:, 2] = np.sqrt(ewald_rad**2 - r2d**2) - ewald_rad

# Random pixel factors
factor = np.random.uniform(3e-4, 4e-4, num_pix)

# Pixel masks
mask = np.zeros(num_pix, np.int16)
mask[r2d < 10] = ef.PixelType.BAD
mask[r2d > 32] = ef.PixelType.CORNER

# Create detector object
det = ef.detector(
    coor=coor,
    mask=mask,
    factor=factor,
    detd=ewald_rad * 0.1,
    ewald_rad=ewald_rad,
    norm_flag=False,
)
print(det)
```

You can select a subset of pixels to create a new detector object.

```python
# Get a new detector by selecting pixels
subset_det = det[np.random.rand(det.num_pix) < 0.5]
print(subset_det)

# Get only the good pixels from the detector object
good_det = det[[ef.PixelType.GOOD]]
print(good_det)
```

Detectors can be saved to and loaded from text-based `.dat` files or HDF5 files.

```python
# Save the detector object to a .dat file
det.write("test_det.dat", overwrite=True)

# Load the detector object from the .dat file
det_from_dat = ef.detector("test_det.dat")
print(det_from_dat)

# Save the detector object to a HDF5 file
det.write("test_det.h5::detector", overwrite=True)

# Load the detector object from the HDF5 file
det_from_h5 = ef.detector("test_det.h5::detector")
print(det_from_h5)
```

### `DetRender`: render with a detector

The `det_render` class helps visualize detector data. You can use it to create 2D images from pixel data.

```python
detr = ef.det_render(det)

# Plot the render of the detector using the x coordinate of the pixels
plt.imshow(detr.render(det.coor[:, 0]), extent=detr.frame_extent(), origin="lower")
plt.xlabel("$x$ / mm")
plt.ylabel("$y$ / mm")
plt.colorbar(label="x-coordinate")
plt.title("Detector Render")
plt.show()
```

## HDF5 Helper Functions

`emcfile` includes helper functions to simplify working with HDF5 files.

### `make_path`

The `make_path` function intelligently creates a path object, automatically detecting whether the path is for a standard file or an HDF5 dataset.

```python
# HDF5 file path
path1 = "/tmp/example.h5::dataset"
h5_path = ef.make_path(path1)
print(f"HDF5 path: {h5_path}, type: {type(h5_path)}")

# Regular file path
path2 = "/tmp/file.txt"
regular_path = ef.make_path(path2)
print(f"Regular path: {regular_path}, type: {type(regular_path)}")
```

### Reading and writing Python objects

You can easily read and write Python objects (like dictionaries) to HDF5 files.

```python
obj = {"name": "sz", "age": 27, "data": {"test": np.random.rand(3, 5)}}
obj_path = "file.h5::person"

# Writing the object to the HDF5 file
ef.write_obj_h5(obj_path, obj, overwrite=True)

# Reading the object from the HDF5 file
read_obj = ef.read_obj_h5(obj_path)
print(read_obj)
```

### Reading and writing NumPy arrays

Similarly, you can read and write NumPy arrays to and from files.

```python
array = np.random.rand(10)
array_path = "file.h5::dataset"

# Write array to HDF5
ef.write_array(array_path, array, overwrite=True)

# Read array from HDF5
read_array = ef.read_array(array_path)
print(read_array)
```

This concludes the tutorial. You should now have a good understanding of the basic functionalities of the `emcfile` package.