from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias

from ._h5helper import PATH_TYPE
from ._pattern_sone import PatternsSOne, write_patterns
from ._pattern_sone_file import PatternsSOneList
from ._patterns import patterns

NP_IMG: TypeAlias = npt.NDArray[np.int_]


class PatternsSOneCollector:
    """
    Collects `np.ndarray` patterns and efficiently converts them into a
    `PatternsSOne` object.

    This class is designed to incrementally build a `PatternsSOne` pattern set
    from a series of NumPy arrays. It uses a buffer to accumulate patterns
    and converts them to the more memory-efficient `PatternsSOne` format in
    batches, which is useful when dealing with a large number of patterns that
    may not fit into memory all at once.

    Parameters
    ----------
    max_buffer_size
        The maximum number of `np.ndarray` patterns to store in the buffer
        before converting them to a `PatternsSOne` object. Defaults to 128.

    Attributes
    ----------
    max_buffer_size : int
        The maximum size of the internal buffer.

    Examples
    --------
    >>> import numpy as np
    >>> from emcfile import PatternsSOneCollector

    >>> collector = PatternsSOneCollector(max_buffer_size=64)
    >>> for _ in range(100):
    ...     pattern = np.random.randint(0, 5, size=(10, 10))
    ...     collector.append(pattern)

    >>> patterns = collector.patterns()
    >>> patterns.num_data
    100
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
        Appends a single `np.ndarray` pattern to the collector.

        The pattern is added to an internal buffer. When the buffer size
        reaches `max_buffer_size`, the buffered patterns are converted to a
        `PatternsSOne` object and stored.

        Parameters
        ----------
        img
            A `np.ndarray` representing a single pattern.

        Raises
        ------
        ValueError
            If the size of the input image does not match the number of pixels
            of the previously added patterns.
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
        Extends the collector with a sequence of patterns.

        This method can be used to add multiple patterns at once, either as a
        sequence of `np.ndarray` objects or as a `PatternsSOne` object.

        Parameters
        ----------
        imgs
            A sequence of `np.ndarray` patterns or a `PatternsSOne` object.

        Raises
        ------
        Exception
            If the input `imgs` is not a supported type.
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

    def patterns(self) -> PatternsSOneList:
        """
        Finalizes the collection process and returns the collected patterns.

        This method clears the internal buffer and concatenates all stored
        `PatternsSOne` objects into a single `PatternsSOneList`.

        Returns
        -------
        PatternsSOneList
            A `PatternsSOneList` object containing all the collected patterns.

        Raises
        ------
        ValueError
            If no patterns have been added to the collector.
        """
        self._clear_buffer()
        if len(self._patterns) == 0:
            raise ValueError("No pattern is added.")
        return PatternsSOneList(self._patterns)

    def pattern_list(self) -> list[PatternsSOne]:
        """
        Returns a list of `PatternsSOne` objects.

        Instead of concatenating all patterns into a single object, this method
        returns a list of the `PatternsSOne` objects that have been created
        from the buffered NumPy arrays. This can be useful for processing large
        datasets in chunks.

        Returns
        -------
        list[PatternsSOne]
            A list of `PatternsSOne` objects.
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
        Writes the collected patterns to a file.

        This method first finalizes the collection process and then writes the
        patterns to the specified file.

        Parameters
        ----------
        path
            The path to the output file.
        h5version
            The HDF5 file format version to use. Defaults to "2".
        overwrite
            If `True`, the output file will be overwritten if it already
            exists. Defaults to `False`.
        buffer_size
            The buffer size in bytes for writing the patterns to the file.
            Defaults to 1 GB.
        """
        self._clear_buffer()
        write_patterns(
            self._patterns,
            path,
            h5version=h5version,
            overwrite=overwrite,
            buffer_size=buffer_size,
        )
