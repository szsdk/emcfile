from collections.abc import Sequence
from typing import Any, Optional, Union, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from ._h5helper import PATH_TYPE
from ._pattern_sone import PatternsSOne, write_patterns
from ._patterns import patterns

NP_IMG: TypeAlias = npt.NDArray[np.integer[Any]]


class PatternsSOneCollector:
    """
    Collects `np.ndarray` patterns and returns a `PatternsSOne` pattern set with
    `.patterns()`.

    Attributes
    ----------
    max_buffer_size : int
        Maximum number of `np.ndarray` patterns to be stored in the buffer before
        converting them to `PatternsSOne`.
    """

    def __init__(self, max_buffer_size: int = 128):
        self.max_buffer_size = max_buffer_size
        self._patterns: list[PatternsSOne] = []
        self._buffer: list[NP_IMG] = []
        self._num_pix: Optional[int] = None

    @property
    def num_pix(self) -> Optional[int]:
        """
        Number of pixels in the patterns.

        Returns
        -------
        Optional[int]
        """
        if self._num_pix is None:
            if len(self._patterns) > 0:
                self._num_pix = self._patterns[0].num_pix
            elif len(self._buffer) > 0:
                self._num_pix = self._buffer[0].size
        return self._num_pix

    def append(self, img: NP_IMG) -> None:
        """
        Append a `np.ndarray` pattern to the buffer.

        Parameters
        ----------
        img : NP_IMG
        """
        if self.num_pix is not None and self.num_pix != img.size:
            raise ValueError(
                f"Size of the input image is {img.size}. "
                f"It does not match the number of pixels {self.num_pix}."
            )
        self._buffer.append(img.ravel())
        if len(self._buffer) >= self.max_buffer_size:
            self._clear_buffer()

    def _clear_buffer(self) -> None:
        if len(self._buffer) <= 0:
            return
        self._patterns.append(patterns(np.array(self._buffer)))
        self._buffer = []

    def extend(self, imgs: Union[Sequence[NP_IMG], PatternsSOne]) -> None:
        """
        Append a sequence of `np.ndarray` patterns or a `PatternsSOne` pattern set to the
        buffer.

        Parameters
        ----------
        imgs : Union[Sequence[NP_IMG], PatternsSOne]
        """
        if isinstance(imgs, (list, np.ndarray)):
            self._buffer.extend(imgs)
            if len(self._buffer) >= self.max_buffer_size:
                self._clear_buffer()
        elif isinstance(imgs, PatternsSOne):
            self._clear_buffer()
            self._patterns.append(imgs)
        else:
            raise Exception()

    def patterns(self) -> PatternsSOne:
        """
        Generate the `PatternsSOne` pattern set from collected patterns.

        Returns
        -------
        PatternsSOne
        """
        self._clear_buffer()
        if len(self._patterns) == 0:
            raise ValueError("No pattern is added.")
        if len(self._patterns) > 0:
            self._patterns = [cast(PatternsSOne, np.concatenate(self._patterns))]
        return self._patterns[0]

    def pattern_list(self) -> list[PatternsSOne]:
        """
        Instead of concatenating all patterns, return a list of `PatternsSOne` pattern.
        This is useful when the number of patterns is too large to be concatenated in
        some circumstances.

        Returns
        -------
        list[PatternsSOne]
        """
        self._clear_buffer()
        return self._patterns

    def write(
        self,
        path: PATH_TYPE,
        *,
        h5version: str = "2",
        overwrite: bool = False,
        buffer_size: int = 1073741824,  # 2 ** 30 bytes = 1 GB
    ) -> None:
        """
        Write the collected patterns to a file.

        Parameters
        ----------
        path : PATH_TYPE
            Path to the file.
        h5version : str
            HDF5 file format version.
        overwrite : bool
            Whether to overwrite the file if it exists.
        buffer_size : int
            Buffer size for writing the patterns to the file.
        """
        self._clear_buffer()
        write_patterns(self._patterns, path)
