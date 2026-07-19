# Naming modernization

This release adds clearer Python-facing names while preserving the existing API
and all EMC and HDF5 file layouts. The compatibility names are intentional:
`emcfile` is used by multiple downstream projects, so this is a migration rather
than an immediate removal of established names.

## Pattern types

| Descriptive name | Compatibility name |
| --- | --- |
| `EMCPatternArray` | `PatternsSOne` |
| `EMCPatternSource` | `PatternsSOneBase` |
| `SparsePattern` | `SPARSE_PATTERN` |
| `EMCPatternCollector` | `PatternsSOneCollector` |
| `FileBackedEMCPatterns` | `PatternsSOneFile` |
| `EMCBinaryPatternFile` | `PatternsSOneEMC` |
| `HDF5PatternFile` | `PatternsSOneH5` |
| `LegacyHDF5PatternFile` | `PatternsSOneH5V1` |
| `EMCPatternCollection` | `PatternsSOneList` |

These pairs are aliases of the same runtime classes. This preserves exact type
identity, `isinstance` checks, subclassing, NumPy dispatch, and pickling behavior.

`num_patterns` and `num_pixels` are descriptive alternatives to `num_data` and
`num_pix`. `ones_offsets` and `multi_offsets` describe the cumulative offset
arrays more precisely than `ones_idx` and `multi_idx`.

The EMC terms `ones`, `multi`, `place_ones`, `place_multi`, and `count_multi`
have not changed. They describe the special treatment of value-1 elements in
the EMC sparse representation and are also part of the storage format.

Method alternatives include:

| Preferred | Deprecated callable |
| --- | --- |
| `mean_photon_count()` | `get_mean_count()` |
| `has_sorted_indices()` | `check_indices_ordered()` |
| `sort_indices()` | `ensure_indices_ordered()` |
| `EMCPatternCollector.to_patterns()` | `PatternsSOneCollector.patterns()` |
| `EMCPatternCollector.pattern_batches()` | `PatternsSOneCollector.pattern_list()` |

## Detector geometry and rendering

The original fields remain readable and writable. The alternatives are
synchronized views of the same data, not copies.

| Descriptive field | EMC-compatible field |
| --- | --- |
| `coordinates` | `coor` |
| `correction_factors` | `factor` |
| `detector_distance` | `detd` |
| `ewald_radius` | `ewald_rad` |
| `is_normalized` | `norm_flag` |
| `geometry_array` | `coor_factor` |

`detector()` accepts either vocabulary. Supplying both spellings for one value
raises `TypeError` instead of silently choosing one.

Rendering alternatives preserve the original fields as well:

| Descriptive name | Compatibility name |
| --- | --- |
| `DetectorRenderer` | `DetRender` |
| `detector_renderer()` | `det_render()` |
| `renderer.detector` | `renderer._det` |
| `renderer.projected_coordinates` | `renderer.cxy` |
| `renderer.pixel_coordinates` | `renderer.xy` |

Detector operation names now describe the operation they perform:

| Preferred | Deprecated callable |
| --- | --- |
| `project_detector_to_2d()` | `get_2ddet()` |
| `resample_detector()` | `get_3ddet_from_shape()` |
| `detectors_allclose()` | `det_isclose()` |
| `fit_ewald_sphere_center()` | `get_ewald_vec()` |

## HDF5 helpers

The alternatives `as_hdf5_path`, `as_path`, `is_hdf5_path`,
`write_hdf5_object`, and `read_hdf5_object` use conventional conversion and
predicate naming. Existing helper names remain supported. `H5Path.file_path`
and `H5Path.object_path` are synchronized alternatives to `fn` and `gn`.

Pattern writers accept `hdf5_version` as an alternative to `h5version`.
Collectors accept `batch_size` as an alternative to `max_buffer_size`.

## Implementation modules

Internal imports now use focused module names. The former module paths remain
as deprecated compatibility aliases, including access to private attributes in
the one-to-one aliases.

| Preferred module | Deprecated module |
| --- | --- |
| `emcfile._emc_patterns` | `emcfile._pattern_sone` |
| `emcfile._pattern_files` | `emcfile._pattern_sone_file` |
| `emcfile._pattern_factory` | `emcfile._patterns` |
| `emcfile._pattern_collector` | `emcfile._collector` |
| `emcfile._hdf5` | `emcfile._h5helper` |
| `emcfile._formatting` | formatting helpers from `emcfile._misc` |
| `emcfile._indexing` | `emcfile._utils` and indexing helpers from `emcfile._misc` |

## Deprecation policy

Superseded callable names use the PEP 702-compatible
`typing_extensions.deprecated` decorator. This provides a `__deprecated__`
marker for type checkers and emits `DeprecationWarning` when an old callable is
used.

Frequently accessed data fields do not emit warnings. Warning on every access
to fields such as `coor` or `ones` would be disruptive to numerical workloads
and to projects that treat warnings as errors. Those compatibility fields remain
fully supported.

## File compatibility

No serialized names or layouts changed. In particular, EMC/HDF5 pattern fields
and detector fields such as `ones`, `place_ones`, `count_multi`, `qx`, `corr`,
`detd`, and `ewald_rad` retain their original spelling and meaning.
