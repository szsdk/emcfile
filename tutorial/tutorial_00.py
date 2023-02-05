# ## Imports

# %matplotlib inline

import matplotlib.pylab as plt
import numpy as np

import emcfile as ef

# ## Patterns

# ### Generate patterns

num_data = 5
num_pix = 10
patterns = np.random.rand(num_data, num_pix) ** 3 * 5
patterns = ef.patterns(patterns.astype("int"))
print(patterns)

patterns.shape

patterns.num_data, patterns.num_pix

patterns.ones, patterns.multi, patterns.place_ones, patterns.place_multi, patterns.count_multi

[patterns.attrs(g) for g in patterns.ATTRS]

patterns[1]

patterns[:2]


# ### IO of Patterns

patterns.write("test_pattern.emc", overwrite=True)

p_emc = ef.patterns("test_pattern.emc")

# IO of h5 file

patterns.write("test_pattern.h5::patterns", overwrite=True)
p_h5 = ef.patterns("test_pattern.h5::patterns")

# ## Detector

# ### Generate a detector

ewald_rad = 128
coor2d = np.mgrid[-32:33, -32:33].reshape(2, -1).T.astype(np.float64)
num_pix = len(coor2d)
coor = np.zeros((num_pix, 3))
coor[:, :2] = coor2d
r2d = np.linalg.norm(coor2d, axis=1)
coor[:, 2] = np.sqrt(ewald_rad**2 - r2d**2) - ewald_rad
factor = np.random.uniform(3e-4, 4e-4, num_pix)
mask = np.zeros(num_pix, np.int16)
mask[r2d < 10] = ef.PixelType.BAD
mask[r2d > 32] = ef.PixelType.CORNER
det = ef.detector(
    coor=coor,
    mask=mask,
    factor=factor,
    detd=ewald_rad * 0.1,
    ewald_rad=ewald_rad,
    norm_flag=False,
)

# ### IO of `Detector`

det.write("test_det.dat", overwrite=True)
ef.detector("test_det.dat")

det.write("test_det.h5::detector", overwrite=True)
ef.detector("test_det.h5::detector")

# ### `DetRender`: render with a detector
detr = ef.det_render(det)
type(detr)

plt.imshow(detr.render(det.coor[:, 0]), extent=detr.frame_extent(), origin="lower")

# ## HDF5 helper
