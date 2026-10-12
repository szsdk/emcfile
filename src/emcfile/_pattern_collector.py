from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import deprecated
from typing import TypeAlias

from ._emc_patterns import PatternsSOne, write_patterns
from ._hdf5 import PATH_TYPE
from ._html_display import html_card
from ._pattern_factory import patterns
from ._pattern_files import PatternsSOneList

NP_IMG: TypeAlias = npt.NDArray[np.int_]


class PatternsSOneCollector:
    """Accumulate dense patterns into sparse in-memory batches.

    ``EMCPatternCollector`` is the descriptive public alias. Each appended
    image is flattened; all patterns must have the same number of pixels.
    Converted batches remain in memory until the collector is discarded.

    Parameters
    ----------
    max_buffer_size : int, default 128
        Historical name for the number of dense patterns per batch.
    batch_size : int, optional
        Descriptive alias for ``max_buffer_size``. Supply only one name.

    Examples
    --------
    >>> import numpy as np
    >>> import emcfile as ef
    >>> collector = ef.EMCPatternCollector(batch_size=2)
    >>> collector.append(np.array([0, 1, 2], dtype=np.int32))
    >>> collector.extend(np.array([[1, 0, 0], [0, 0, 3]], dtype=np.int32))
    >>> collector.to_patterns().shape
    (3, 3)
    """

    def __init__(self, max_buffer_size: int = 128, *, batch_size: int | None = None):
        if batch_size is not None:
            if max_buffer_size != 128:
                raise TypeError(
                    "Use either 'batch_size' or 'max_buffer_size', not both"
                )
            max_buffer_size = batch_size
        self.max_buffer_size = max_buffer_size
        self._patterns: list[PatternsSOne] = []
        self._buffer: list[NP_IMG] = []
        self._num_pix: int | None = None

    @property
    def batch_size(self) -> int:
        """Maximum number of patterns converted in one batch."""
        return self.max_buffer_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.max_buffer_size = value

    @property
    def num_pix(self) -> int | None:
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

    @property
    def num_pixels(self) -> int | None:
        """Number of pixels in each collected pattern."""
        return self.num_pix

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
        self._validate_num_pix(img.size)
        self._buffer.append(img.ravel())
        if len(self._buffer) >= self.max_buffer_size:
            self._flush_buffer()

    def _validate_num_pix(self, num_pix: int) -> None:
        if self.num_pix is not None and self.num_pix != num_pix:
            raise ValueError(
                f"Size of the input image is {num_pix}. "
                f"It does not match the number of pixels {self.num_pix}."
            )

    def _flush_buffer(self) -> None:
        if len(self._buffer) <= 0:
            return
        self._patterns.append(patterns(np.array(self._buffer)))
        self._buffer = []

    def extend(self, imgs: Sequence[NP_IMG] | PatternsSOne) -> None:
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
        TypeError
            If `imgs` is not a supported collection or contains non-array items.
        ValueError
            If any pattern has a different number of pixels.
        """
        if isinstance(imgs, PatternsSOne):
            self._validate_num_pix(imgs.num_pix)
            self._flush_buffer()
            self._patterns.append(imgs)
            return

        if not isinstance(imgs, (Sequence, np.ndarray)) or isinstance(
            imgs, (str, bytes)
        ):
            raise TypeError(
                "imgs must be a sequence of NumPy arrays or a PatternsSOne object"
            )

        images = list(imgs)
        if not all(isinstance(img, np.ndarray) for img in images):
            raise TypeError("Each image must be a NumPy array")
        for img in images:
            self._validate_num_pix(img.size)
        for img in images:
            self.append(img)

    def to_patterns(self) -> PatternsSOneList:
        """Flush pending images and return a collection of sparse batches.

        Returns an ``EMCPatternCollection`` referencing the stored batches;
        no contiguous event-array concatenation is performed. The collector
        retains its batches and can accept additional patterns afterward.
        Raises ``ValueError`` if no patterns have been collected.
        """
        self._flush_buffer()
        if len(self._patterns) == 0:
            raise ValueError("No pattern is added.")
        return PatternsSOneList(self._patterns)

    @deprecated("Use to_patterns() instead.")
    def patterns(self) -> PatternsSOneList:
        return self.to_patterns()

    def pattern_batches(self) -> list[PatternsSOne]:
        """Flush pending images and return the internal list of sparse batches.

        The list and its pattern arrays are shared with the collector.
        Mutating them changes the collector's stored data.
        """
        self._flush_buffer()
        return self._patterns

    @deprecated("Use pattern_batches() instead.")
    def pattern_list(self) -> list[PatternsSOne]:
        return self.pattern_batches()

    def _repr_html_(self) -> str:
        buffered_patterns = len(self._buffer)
        summary = {
            "pixels": self.num_pix,
            "stored patterns": int(
                sum(len(p) for p in self._patterns) + buffered_patterns
            ),
            "stored chunks": len(self._patterns),
            "buffer size": self.max_buffer_size,
        }
        return html_card(
            "Pattern collector",
            summary,
            details={
                "type": self.__class__.__name__,
                "buffered_patterns": buffered_patterns,
            },
            bars=(
                (
                    "buffer fill",
                    100 * buffered_patterns / max(1, self.max_buffer_size),
                    "#7c3aed",
                ),
            ),
        )

    def write(
        self,
        path: PATH_TYPE,
        *,
        h5version: str = "2",
        overwrite: bool = False,
        buffer_size: int = 1073741824,  # 2 ** 30 bytes = 1 GB
        compression: None | int | str = None,
        compression_opts: Any = None,
        shuffle: bool = False,
        position_encoding: str = "absolute",
        check_sorted: bool = False,
        hdf5_version: str | None = None,
    ) -> None:
        """Flush pending images and write all collected batches to a file.

        Accepts the output, compression, encoding, and buffer options of
        ``write_patterns``. Stored batches are retained after writing.

        See Also
        --------
        write_patterns : Full parameter descriptions and format restrictions.
        """
        self._flush_buffer()
        write_patterns(
            self._patterns,
            path,
            h5version=h5version,
            overwrite=overwrite,
            buffer_size=buffer_size,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            position_encoding=position_encoding,
            check_sorted=check_sorted,
            hdf5_version=hdf5_version,
        )


EMCPatternCollector = PatternsSOneCollector
