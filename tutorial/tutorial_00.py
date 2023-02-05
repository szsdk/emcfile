# Import necessary libraries

# %matplotlib inline

import matplotlib.pylab as plt
import numpy as np

import emcfile as ef

# ## Patterns

# ### Generate patterns

# Define number of data and pixels

num_data = 5
num_pix = 10

# Create random patterns

patterns = np.random.rand(num_data, num_pix) ** 3 * 5
patterns = ef.patterns(patterns.astype("int"))
print(patterns)

# Shape of patterns

patterns.shape

# Number of data and pixels in patterns

patterns.num_data, patterns.num_pix

patterns.ones, patterns.multi, patterns.place_ones, patterns.place_multi, patterns.count_multi

[getattr(patterns, g) for g in patterns.ATTRS]

# Accessing a single pattern

patterns[1]

# Accessing multiple patterns

patterns[:2]

# Accessing specific pixels of multiple patterns

patterns[:, :2]


# ### Write and read patterns to/from different file formats

# Write patterns to .emc file

patterns.write("test_pattern.emc", overwrite=True)

# Read patterns from .emc file

p_emc = ef.patterns("test_pattern.emc")

# Write patterns to group 'patterns' in a .h5 file

patterns.write("test_pattern.h5::patterns", overwrite=True)

# Read patterns from the .h5 file

p_h5 = ef.patterns("test_pattern.h5::patterns")

# ## Detector

# ### Generate a detector

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
    coor=coor,  # coordinate of the detector pixels
    mask=mask,  # mask indicating the type of each pixel
    factor=factor,  # scaling factor for each pixel
    detd=ewald_rad
    * 0.1,  # distance between the detector and the sample, 0.1 mm / pixel
    ewald_rad=ewald_rad,  # Ewald sphere radius
    norm_flag=False,  # Normalization flag, set to False
)
det

# Get a new detector by selecting pixels

det[np.random.rand(det.num_pix) < 0.5]

# Get only the good pixels from the detector object

det[[ef.PixelType.GOOD]]

# Save the detector object to a .dat file, a text format

det.write("test_det.dat", overwrite=True)

# Load the detector object from the .dat file

ef.detector("test_det.dat")

# The `ef.detector` is a function that can create a detector object from input arguments such as the coordinate of pixels, mask of pixels, and pixel-wise factors, along with additional parameters like detector distance, Ewald radius, and normalization flag. It can also read a detector from a file on disk. The detector object can be saved to disk using the method `.write`.


# Save the detector object to a HDF5 file

det.write("test_det.h5::detector", overwrite=True)

# Load the detector object from the HDF5 file

ef.detector("test_det.h5::detector")

# ### `DetRender`: render with a detector

detr = ef.det_render(det)
type(detr)

# Plot the render of the detector using the $x$ coordinate of the pixels

plt.imshow(detr.render(det.coor[:, 0]), extent=detr.frame_extent(), origin="lower")
plt.xlabel("$x$ / mm")
plt.ylabel("$y$ / mm")

# ## HDF5 helper
