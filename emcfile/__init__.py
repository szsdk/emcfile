from ._detector import (
    Detector,
    DetRender,
    PixelType,
    det_isclose,
    det_render,
    detector,
    get_2ddet,
    get_3ddet_from_shape,
    get_ewald_vec,
)
from ._h5helper import (
    PATH_TYPE,
    H5Path,
    check_h5path,
    check_remove_groups,
    h5group,
    h5path,
    make_path,
    read_array,
    read_obj_h5,
    write_array,
    write_obj_h5,
)
from ._pattern_sone import PATTERNS_HEADER, PatternsSOne, vstack, write_photons
from ._pattern_sone_file import PatternsSOneEMC
from ._patterns import patterns

__all__ = [
    "det_render",
    "Detector",
    "DetRender",
    "detector",
    "get_2ddet",
    "get_3ddet_from_shape",
    "det_isclose",
    "get_ewald_vec",
    "PixelType",
    "H5Path",
    "PATH_TYPE",
    "check_h5path",
    "check_remove_groups",
    "h5group",
    "write_array",
    "read_array",
    "h5path",
    "make_path",
    "write_obj_h5",
    "read_obj_h5",
    "patterns",
    "PATTERNS_HEADER",
    "PatternsSOne",
    "vstack",
    "write_photons",
    "PatternsSOneEMC",
]

__pdoc__ = {"tests": False}
