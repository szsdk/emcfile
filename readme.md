# emcfile

This package provides useful utilities for the `emc` photon format and its corresponding
detector files.

## Installation
Dependencies:
- numpy
- h5py
- scipy
- beartype

## Documentation

This package can be divided into three parts roughly:
- handle classes and functions for HDF5 file
- detector file related classes and functions
- pattern file related classes and functions

The tutorial, `tutorial/tutorial_00.py`, is recommended as the starting point of this package.

## Conventions
### Patterns
For a pattern object, its first index is always the pattern index.
