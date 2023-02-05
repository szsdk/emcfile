# emcfile

This package provides useful utilities for the `emc` photon format and its corresponding
detector files.

## Installation
Dependencies:
- python >= 3.9
- numpy >= 1.20
- h5py
- scipy
- beartype

## Documentation

This package can be divided into three parts roughly:
- handle classes and functions for HDF5 file
- detector file related classes and functions
- pattern file related classes and functions

The tutorial, `tutorial/tutorial_00.py`, is recommended as the starting point of this package.
You could use [jupytext](https://github.com/mwouts/jupytext) to generate a jupyter notebook file
for a more interactive experience with the following command
```bash
jupytext tutorial_00.py -o tutorial_00.ipynb
```

## Conventions
### Patterns
For a pattern object, its first index is always the pattern index.
