# ## Imports

# %matplotlib inline

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


# ### IO of Patterns

patterns.write("test_pattern.emc")

p_emc = ef.patterns("test_pattern.emc")

# IO of h5 file

patterns.write("test_pattern.h5::patterns")
p_h5 = ef.patterns("test_pattern.h5::patterns")

# ## Detector

# ## HDF5 helper
