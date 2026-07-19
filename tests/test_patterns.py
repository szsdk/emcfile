import copy
import gc
import io
import itertools
import logging
import tempfile
import time

import numpy as np
import pytest
from psutil import Process
from scipy.sparse import coo_array, csr_array

import emcfile as ef

from .utils import temporary_random_seed


def generate_dense_patterns(num_patterns, num_pixels):
    with temporary_random_seed(123):
        return (10 * np.random.rand(num_patterns, num_pixels) ** 5).astype(int)


@pytest.fixture()
def large_dense():
    return generate_dense_patterns(1000, 4096)


@pytest.fixture()
def large_patterns(large_dense):
    return ef.patterns(large_dense)


def test_mean_photon_count(large_patterns, large_dense):
    np.testing.assert_almost_equal(
        large_patterns.mean_photon_count(), large_dense.sum(1).mean()
    )


def test_todense(large_patterns, large_dense):
    np.testing.assert_equal(large_patterns.todense(), large_dense)
    np.testing.assert_equal(np.asarray(large_patterns, dtype=float), large_dense)


def test_tocsr(large_patterns, large_dense):
    np.testing.assert_equal(large_patterns.tocsr().todense(), large_dense)


@pytest.fixture()
def small_dense():
    return generate_dense_patterns(32, 4)


@pytest.fixture()
def small_patterns(small_dense):
    return ef.patterns(small_dense)


@pytest.fixture()
def emc_path(tmp_path_factory, large_patterns):
    path = tmp_path_factory.mktemp("data") / "photon.emc"
    large_patterns.write(path)
    return path


@pytest.fixture()
def hdf5_path(tmp_path_factory, large_patterns):
    path = tmp_path_factory.mktemp("data") / "photon.h5::patterns"
    large_patterns.write(path)
    return path


@pytest.fixture()
def legacy_hdf5_path(tmp_path_factory, large_patterns):
    path = tmp_path_factory.mktemp("data") / "photon.h5::patterns"
    large_patterns.write(path, hdf5_version="1")
    return path


def test_from_sparse_patterns(large_patterns):
    assert large_patterns[:10] == ef.patterns(
        [large_patterns.sparse_pattern(i) for i in range(10)]
    )


def test_pattern_statistics(large_patterns):
    large_patterns.mean_photon_count()
    large_patterns.nbytes
    large_patterns.sparsity()


def test_display(large_patterns, emc_path):
    html = large_patterns._repr_html_()
    assert isinstance(html, str)
    assert "Patterns" in html or "patterns" in html.lower()

    file_html = ef.open_patterns(emc_path)._repr_html_()
    assert isinstance(file_html, str)


def test_display_marimo(large_patterns):
    # _repr_html_() always returns HTML string
    html = large_patterns._repr_html_()
    assert isinstance(html, str)
    assert "Patterns" in html or "patterns" in html.lower()


def generate_pattern_inputs():
    with temporary_random_seed(123):
        dense = (5 * np.random.rand(400, 128**2) ** 5).astype("i4")
    expected = ef.patterns(dense)
    yield expected, expected
    yield dense, expected
    yield coo_array(dense), expected
    yield csr_array(dense), expected


def test_pattern_not_equal(small_patterns):
    """
    This test covers different cases for the `__eq__`  returns false.
    """
    modified = ef.patterns(
        generate_dense_patterns(
            small_patterns.num_patterns - 1, small_patterns.num_pixels
        )
    )
    assert modified != small_patterns

    modified = ef.patterns(
        generate_dense_patterns(
            small_patterns.num_patterns, small_patterns.num_pixels - 1
        )
    )
    assert modified != small_patterns

    modified = copy.deepcopy(small_patterns)
    modified.place_multi[0] = 9999999
    assert modified != small_patterns


@pytest.mark.parametrize(("source", "expected"), tuple(generate_pattern_inputs()))
def test_patterns(source, expected):
    patterns = ef.patterns(source)
    assert patterns == expected


def test_shape(large_patterns):
    assert len(large_patterns) == large_patterns.num_patterns
    assert large_patterns.shape == (
        large_patterns.num_patterns,
        large_patterns.num_pixels,
    )


def test_getitem(large_patterns, large_dense):
    for index in np.random.choice(large_patterns.num_patterns, 5):
        assert np.sum(large_patterns[index] == 1) == large_patterns.ones[index]

    mask = np.random.rand(large_patterns.num_patterns) < 0.5
    indices = np.where(mask)[0]
    start_time = time.time()
    selected_by_mask = large_patterns[mask]
    selected_by_index = large_patterns[indices]
    logging.info("Select patterns: %fs", (time.time() - start_time) / 2)
    assert selected_by_mask == selected_by_index
    for index in np.random.choice(selected_by_mask.num_patterns, 5):
        assert np.all(selected_by_mask[index] == large_patterns[indices[index]])

    for _ in range(10):
        indices = np.random.choice(
            large_patterns.num_patterns,
            size=np.random.randint(large_patterns.num_patterns),
        )
        assert np.all(large_patterns[indices].todense() == large_dense[indices])


def test_concatenate(small_patterns, large_patterns):
    pattern_arrays = [ef.patterns(large_patterns.num_pixels)] + [
        ef.patterns(large_patterns, start=i * 10, end=(i + 1) * 10) for i in range(5)
    ]
    concatenated = np.concatenate(pattern_arrays)
    assert large_patterns[:50] == concatenated
    concatenated = np.concatenate(pattern_arrays, casting="destroy")
    assert large_patterns[:50] == concatenated

    pattern_arrays = [ef.patterns(np.full((10000, 1000), 2)) for _ in range(2)]
    process = Process()
    memory_before = process.memory_info().rss
    concatenated = np.concatenate(pattern_arrays, casting="destroy")
    gc.collect()
    memory_after = process.memory_info().rss
    assert (memory_after - memory_before) < concatenated.nbytes * 0.9
    assert len(pattern_arrays) == 0

    assert np.concatenate([ef.patterns(small_patterns)] * 2, axis=1) == ef.patterns(
        np.concatenate([small_patterns] * 2, axis=1)
    )


# Fixture names are resolved by ``request`` in ``test_sum``.
def generate_sum_cases():
    for axis, keepdims, dtype in itertools.product(
        [None, 0, 1], [False, True], [int, float, None]
    ):
        yield axis, keepdims, dtype, "small_patterns", "small_dense"


@pytest.mark.parametrize(
    ("axis", "keepdims", "dtype", "patterns_fixture", "dense_fixture"),
    tuple(generate_sum_cases()),
)
def test_sum(axis, keepdims, dtype, patterns_fixture, dense_fixture, request):
    patterns = request.getfixturevalue(patterns_fixture)
    dense = request.getfixturevalue(dense_fixture)
    np.testing.assert_almost_equal(
        patterns.sum(axis=axis, keepdims=keepdims, dtype=dtype),
        dense.sum(axis=axis, keepdims=keepdims, dtype=dtype),
    )


def test_empty():
    num_pixels = 32
    empty_patterns = ef.patterns(num_pixels)
    assert empty_patterns.num_pixels == num_pixels
    assert empty_patterns.num_patterns == 0
    assert empty_patterns.mean_photon_count() == 0.0
    assert empty_patterns.sparsity() == 0.0


def test_unsupported_pattern_source():
    with pytest.raises(TypeError, match="Unsupported pattern source type"):
        ef.patterns(object())


@pytest.mark.parametrize(
    ("suffix", "write_options"),
    [
        (".emc", {}),
        (".h5", {"hdf5_version": "1"}),
        (".h5", {"hdf5_version": "2"}),
    ],
)
def test_file_roundtrip(suffix, write_options, large_patterns):
    with tempfile.NamedTemporaryFile(suffix=suffix) as file:
        start_time = time.time()
        large_patterns.write(file.name, overwrite=True, **write_options)
        logging.info(
            "Writing %d patterns to %s file (%s): %fs",
            large_patterns.num_patterns,
            suffix,
            write_options,
            time.time() - start_time,
        )

        start = large_patterns.num_patterns // 3
        end = start * 2
        start_time = time.time()
        restored = ef.patterns(file.name, start=start, end=end)
        logging.info(
            "Reading %d patterns: %fs",
            restored.num_patterns,
            time.time() - start_time,
        )
        assert restored == large_patterns[start:end]


def test_bytesio(small_patterns):
    buffer = io.BytesIO()
    small_patterns.write(buffer)
    assert ef.patterns(buffer) == small_patterns


def generate_write_cases():
    patterns = ef.patterns(np.random.randint(0, 10, size=(16, 256)))
    for count in 2 ** np.arange(0, 10, 2):
        yield ".emc", [patterns] * count
        yield ".h5", [patterns] * count


@pytest.mark.parametrize(("suffix", "pattern_arrays"), tuple(generate_write_cases()))
def test_write_patterns(suffix, pattern_arrays):
    with (
        tempfile.NamedTemporaryFile(suffix=suffix) as f0,
        tempfile.NamedTemporaryFile(suffix=suffix) as f1,
    ):
        t = time.time()
        combined = np.concatenate(pattern_arrays)
        combined.write(f1.name, overwrite=True)
        single_write_time = time.time() - t
        logging.info(
            "speed[single]: %.2f GB/s",
            combined.nbytes * 1e-9 / single_write_time,
        )

        t = time.time()
        ef.write_patterns(pattern_arrays, f0.name, buffer_size=2**12, overwrite=True)
        multi_write_time = time.time() - t
        logging.info(
            "speed[multiple; #patterns=%d]: %.2f GB/s",
            len(pattern_arrays),
            combined.nbytes * 1e-9 / multi_write_time,
        )

        logging.info(
            "speed ratio [%s]: %.3f",
            suffix,
            single_write_time / multi_write_time,
        )
        assert ef.patterns(f0.name) == combined


def test_matrix_multiplication(large_patterns):
    matrix = np.random.rand(large_patterns.num_pixels, 10)
    np.testing.assert_almost_equal(
        large_patterns @ matrix, np.asarray(large_patterns) @ matrix
    )
    matrix = matrix > 0.4
    np.testing.assert_almost_equal(
        large_patterns @ matrix, large_patterns.todense() @ matrix
    )
    matrix = coo_array(matrix)
    np.testing.assert_equal(
        (large_patterns @ matrix).todense(),
        large_patterns.todense() @ matrix,
    )
    matrix = csr_array(matrix)
    np.testing.assert_equal(
        (large_patterns @ matrix).todense(),
        large_patterns.todense() @ matrix,
    )


@pytest.mark.parametrize("path_fixture", ["emc_path", "hdf5_path", "legacy_hdf5_path"])
def test_pattern_file_shape(path_fixture, request):
    path = request.getfixturevalue(path_fixture)
    patterns = ef.open_patterns(path)
    assert len(patterns) == patterns.num_patterns
    assert patterns.shape == (patterns.num_patterns, patterns.num_pixels)


@pytest.mark.parametrize(
    ("path_fixture", "chunk_size"),
    [
        ("emc_path", 1),
        ("hdf5_path", 1),
        pytest.param("legacy_hdf5_path", 1, marks=pytest.mark.skip("Too slow")),
        ("emc_path", 128),
        ("hdf5_path", 128),
        ("legacy_hdf5_path", 128),
        ("emc_path", 99999),
        ("hdf5_path", 99999),
        ("legacy_hdf5_path", 99999),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_pattern_file_sum(path_fixture, axis, keepdims, chunk_size, request):
    path = request.getfixturevalue(path_fixture)
    expected = ef.patterns(path)
    patterns = ef.open_patterns(path)
    assert np.all(
        expected.sum(axis=axis, keepdims=keepdims)
        == patterns.sum(axis=axis, keepdims=keepdims, chunk_size=chunk_size)
    )


# HDF5-backed arrays do not expose every attribute without loading the data.
@pytest.mark.parametrize("path_fixture", ["emc_path"])
def test_pattern_file_attributes(path_fixture, request):
    path = request.getfixturevalue(path_fixture)
    expected = ef.patterns(path)
    patterns = ef.open_patterns(path)
    for attribute in ef.EMCPatternArray.ATTRS:
        np.testing.assert_equal(
            getattr(expected, attribute), getattr(patterns, attribute)
        )


@pytest.mark.parametrize("path_fixture", ["emc_path", "hdf5_path", "legacy_hdf5_path"])
def test_pattern_file_getitem(path_fixture, request):
    path = request.getfixturevalue(path_fixture)
    expected = ef.patterns(path)
    patterns = ef.open_patterns(path)
    assert patterns.sparsity() == expected.sparsity()
    np.testing.assert_equal(patterns[3], expected[3])
    assert patterns[::2] == expected[::2]
    with temporary_random_seed(12):
        mask = np.random.rand(expected.num_patterns) > 0.5
    assert patterns[mask] == expected[mask]
    indices = np.where(mask)[0]
    assert patterns[indices] == expected[indices]
    empty_selection = patterns[np.array([], dtype=np.int32)]
    assert empty_selection.num_patterns == 0


@pytest.mark.parametrize("path_fixture", ["emc_path", "hdf5_path", "legacy_hdf5_path"])
def test_pattern_file_sparse_pattern(path_fixture, request):
    path = request.getfixturevalue(path_fixture)
    expected = ef.patterns(path, end=10)
    patterns = ef.open_patterns(path)
    sparse_patterns = [patterns.sparse_pattern(i) for i in range(10)]
    assert expected == ef.patterns(sparse_patterns)


@pytest.mark.parametrize("path_fixture", ["emc_path", "hdf5_path"])
def test_open_reader(path_fixture, request):
    path = request.getfixturevalue(path_fixture)
    expected = ef.patterns(path, end=10)
    patterns = ef.open_patterns(path)
    with patterns.open() as reader:
        sparse_patterns = [reader.sparse_pattern(i) for i in range(10)]
    assert expected == ef.patterns(sparse_patterns)


def test_index(large_patterns, large_dense):
    indices = np.arange(large_patterns.shape[1])
    indices[0], indices[-1] = indices[-1], indices[0]
    selected = large_patterns[:, indices]
    assert selected.has_sorted_indices() is False
    selected.sort_indices()
    assert selected.has_sorted_indices() is True
    assert np.all(np.asarray(selected.todense()) == large_dense[:, indices])


@pytest.mark.parametrize("n", range(3))
def test_pow(n, request):
    patterns = request.getfixturevalue("small_patterns")
    dense = request.getfixturevalue("small_dense")
    np.testing.assert_equal(dense**n, (patterns**n).todense())


@pytest.mark.parametrize("shape", [(10, 3), (10, 0)])
def test_zeros(shape):
    patterns = ef.patterns((shape, 0))
    np.testing.assert_equal(
        patterns.todense(),
        np.zeros(shape),
    )
    if shape[1] == 0:
        assert patterns.sparsity() == 0.0


@pytest.mark.parametrize("shape", [(10, 3), (10, 0)])
def test_ones(shape):
    np.testing.assert_equal(
        ef.patterns((shape, 1)).todense(),
        np.ones(shape),
    )


@pytest.mark.parametrize("shape", [(10, 3), (10, 0)])
def test_full(shape):
    np.testing.assert_equal(
        ef.patterns((shape, 2)).todense(),
        np.full(shape, 2),
    )


def test_pattern_collection(emc_path, hdf5_path):
    emc_patterns = ef.open_patterns(emc_path)
    hdf5_patterns = ef.open_patterns(hdf5_path)
    collection = ef.EMCPatternCollection([emc_patterns, hdf5_patterns[:]])
    np.testing.assert_equal(collection[0], emc_patterns[0])
    np.testing.assert_equal(collection[len(emc_patterns) + 2], hdf5_patterns[2])
    assert len(collection) == len(emc_patterns) + len(hdf5_patterns)
    np.testing.assert_equal(collection[1::3][0], emc_patterns[1])
    np.testing.assert_equal(collection[1::3][1], emc_patterns[4])
    indices = [
        len(emc_patterns) + 2,
        1,
        len(emc_patterns) + 2,
        len(emc_patterns) - 1,
        0,
    ]
    np.testing.assert_equal(
        collection[indices].todense(),
        np.vstack([collection[i] for i in indices]),
    )
    indices = [len(emc_patterns) + 1, 2]
    np.testing.assert_equal(
        collection[indices].todense(),
        np.vstack([collection[i] for i in indices]),
    )
    np.testing.assert_equal(
        collection.ones,
        np.concatenate([emc_patterns.ones, hdf5_patterns.ones]),
    )
    combined_collection = ef.EMCPatternCollection([collection, emc_patterns])
    assert (
        combined_collection[: len(collection)][: len(emc_patterns)] == (emc_patterns[:])
    )
    html = collection._repr_html_()
    assert isinstance(html, str)


def test_pattern_collection_batches_random_indices_by_source():
    class CountingPatternArray(ef.EMCPatternArray):
        def __init__(self, source):
            super().__init__(
                source.num_pixels,
                source.ones,
                source.multi,
                source.place_ones,
                source.place_multi,
                source.count_multi,
            )
            self.selections = []

        def __getitem__(self, index):
            self.selections.append(index)
            return super().__getitem__(index)

    dense0 = np.arange(12).reshape(3, 4)
    dense1 = np.arange(12, 24).reshape(3, 4)
    source0 = CountingPatternArray(ef.patterns(dense0))
    source1 = CountingPatternArray(ef.patterns(dense1))
    collection = ef.EMCPatternCollection([source0, source1])
    indices = np.array([0, 3, 1, 4, 2, 5, 0])

    np.testing.assert_equal(
        collection[indices].todense(),
        np.vstack([np.vstack([dense0, dense1])[i] for i in indices]),
    )
    assert len(source0.selections) == 1
    assert len(source1.selections) == 1
    np.testing.assert_array_equal(source0.selections[0], [0, 1, 2, 0])
    np.testing.assert_array_equal(source1.selections[0], [0, 1, 2])


@pytest.mark.parametrize("filename", ["patterns.emc", "patterns.h5"])
def test_pattern_collection_write(emc_path, hdf5_path, filename, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / filename
    hdf5_patterns = ef.open_patterns(hdf5_path)
    in_memory_patterns = hdf5_patterns[:]
    collection = ef.EMCPatternCollection([emc_path, hdf5_patterns, in_memory_patterns])
    collection.write(path, overwrite=True)
    expected = np.concatenate(
        [ef.patterns(emc_path), hdf5_patterns[:], in_memory_patterns]
    )
    assert ef.patterns(path) == expected


def test_pattern_repr(small_patterns):
    representation = repr(small_patterns)
    assert f"Number of patterns: {small_patterns.num_patterns}" in representation
    assert f"Number of pixels: {small_patterns.num_pixels}" in representation
