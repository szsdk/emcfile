"""Optional compiled gather for non-contiguous EMC row selection."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def gather_rows(
    ids: np.ndarray,
    source_ones_offsets: np.ndarray,
    source_multi_offsets: np.ndarray,
    source_place_ones: np.ndarray,
    source_place_multi: np.ndarray,
    source_count_multi: np.ndarray,
    target_ones_offsets: np.ndarray,
    target_multi_offsets: np.ndarray,
    target_place_ones: np.ndarray,
    target_place_multi: np.ndarray,
    target_count_multi: np.ndarray,
) -> None:
    for output_row in range(ids.size):
        source_row = ids[output_row]
        source_start = source_ones_offsets[source_row]
        target_start = target_ones_offsets[output_row]
        for j in range(source_ones_offsets[source_row + 1] - source_start):
            target_place_ones[target_start + j] = source_place_ones[source_start + j]

        source_start = source_multi_offsets[source_row]
        target_start = target_multi_offsets[output_row]
        for j in range(source_multi_offsets[source_row + 1] - source_start):
            target_place_multi[target_start + j] = source_place_multi[source_start + j]
            target_count_multi[target_start + j] = source_count_multi[source_start + j]
