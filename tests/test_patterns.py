import copy
import gc
import io
import itertools
import logging
import tempfile
import time
from pathlib import Path

import emcfile as ef
import numpy as np
import pytest
from psutil import Process
from scipy.sparse import coo_array, csr_array

from .utils import temp_seed


def gen_dense(num_data, num_pix):
    with temp_seed(123):
        return (10 * np.random.rand(num_data, num_pix) ** 5).astype(int)


@pytest.fixture()
def big_dense():
    return gen_dense(1000, 4096)


@pytest.fixture()
def big_data(big_dense):
    return ef.patterns(big_dense)


def test_get_mean_count(big_data, big_dense):
    np.testing.assert_almost_equal(big_data.get_mean_count(), big_dense.sum(1).mean())


def test_todense(big_data, big_dense):
    np.testing.assert_equal(big_data.todense(), big_dense)


@pytest.fixture()
def small_dense():
    return gen_dense(32, 4)


@pytest.fixture()
def small_data(small_dense):
    return ef.patterns(small_dense)


@pytest.fixture()
def data_emc(tmp_path_factory, big_data):
    fn = tmp_path_factory.mktemp("data") / "photon.emc"
    big_data.write(fn)
    return fn


@pytest.fixture()
def data_h5(tmp_path_factory, big_data):
    fn = tmp_path_factory.mktemp("data") / "photon.h5::patterns"
    big_data.write(fn)
    return fn


@pytest.fixture()
def data_h5_v1(tmp_path_factory, big_data):
    fn = tmp_path_factory.mktemp("data") / "photon.h5::patterns"
    big_data.write(fn, h5version="1")
    return fn


def test_from_sparse_patterns(big_data):
    assert big_data[:10] == ef.patterns([big_data.sparse_pattern(i) for i in range(10)])


def test_operation(big_data):
    big_data.get_mean_count()
    big_data.nbytes
    big_data.sparsity()


def gen_pattern_inputs():
    with temp_seed(123):
        dense = (5 * np.random.rand(400, 128**2) ** 5).astype("i4")
    ref = ef.patterns(dense)
    yield ref, ref
    yield dense, ref
    yield coo_array(dense), ref
    yield csr_array(dense), ref


def test_pattern_not_equal(small_data):
    """
    This test covers different cases for the `__eq__`  returns false.
    """
    a = ef.patterns(gen_dense(small_data.num_data - 1, small_data.num_pix))
    assert a != small_data

    a = ef.patterns(gen_dense(small_data.num_data, small_data.num_pix - 1))
    assert a != small_data

    a = copy.deepcopy(small_data)
    a.place_multi[0] = 9999999
    assert a != small_data


@pytest.mark.parametrize("inp, ref", gen_pattern_inputs())
def test_patterns(inp, ref):
    p = ef.patterns(inp)
    assert p == ref


def test_shape(big_data):
    assert len(big_data) == big_data.num_data
    assert big_data.shape == (big_data.num_data, big_data.num_pix)


def test_getitem(big_data, big_dense):
    for i in np.random.choice(big_data.num_data, 5):
        assert np.sum(big_data[i] == 1) == big_data.ones[i]

    mask = np.random.rand(big_data.num_data) < 0.5
    idx = np.where(mask)[0]
    t0 = time.time()
    subdata1 = big_data[mask]
    subdata2 = big_data[idx]
    logging.info(f"Select dataset: {(time.time() - t0) / 2}s")
    assert subdata1 == subdata2
    for _ in np.random.choice(subdata1.num_data, 5):
        assert np.all(subdata1[_] == big_data[idx[_]])

    for _ in range(10):
        i = np.random.choice(
            big_data.num_data, size=np.random.randint(big_data.num_data)
        )
        assert np.all(big_data[i].todense() == big_dense[i])


def test_concatenate(small_data, big_data):
    patterns = [ef.patterns(big_data.num_pix)] + [
        ef.patterns(big_data, start=i * 10, end=(i + 1) * 10) for i in range(5)
    ]
    ans = np.concatenate(patterns)
    assert big_data[:50] == ans
    ans = np.concatenate(patterns, casting="destroy")
    assert big_data[:50] == ans

    patterns = [ef.patterns(np.full((10000, 1000), 2)) for _ in range(2)]
    process = Process()
    m0 = process.memory_info().rss
    ans = np.concatenate(patterns, casting="destroy")
    gc.collect()
    m1 = process.memory_info().rss
    assert (m1 - m0) < ans.nbytes * 0.9
    assert len(patterns) == 0

    print(small_data)
    assert np.concatenate([ef.patterns(small_data)] * 2, axis=1) == ef.patterns(
        np.concatenate([small_data] * 2, axis=1)
    )


# This fixture is not used in the test, but it is used in the test_sum
def gen_sum_inputs():
    for axis, keepdims, dtype in itertools.product(
        [None, 0, 1], [False, True], [int, float, None]
    ):
        yield axis, keepdims, dtype, "small_data", "small_dense"


@pytest.mark.parametrize("axis, keepdims, dtype, data, dense", gen_sum_inputs())
def test_sum(axis, keepdims, dtype, data, dense, request):
    data = request.getfixturevalue(data)
    dense = request.getfixturevalue(dense)
    np.testing.assert_almost_equal(
        data.sum(axis=axis, keepdims=keepdims, dtype=dtype),
        dense.sum(axis=axis, keepdims=keepdims, dtype=dtype),
    )


def test_empty():
    num_pix = 32
    empty_data = ef.patterns(num_pix)
    assert empty_data.num_pix == num_pix
    assert empty_data.num_data == 0


@pytest.mark.parametrize(
    "suffix, kargs",
    [
        (".emc", dict()),
        (".h5", dict(h5version="1")),
        (".h5", dict(h5version="2")),
    ],
)
def test_fileio(suffix, kargs, big_data):
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        t0 = time.time()
        big_data.write(f.name, overwrite=True, **kargs)
        logging.info(
            "Writing %d patterns to %s file(%s): {time.time()-t0}",
            big_data.num_data,
            suffix,
            kargs,
        )

        start = big_data.num_data // 3
        end = start * 2
        t0 = time.time()
        d_read = ef.patterns(f.name, start=start, end=end)
        logging.info(
            f"Reading {d_read.num_data} patterns from h5 file(v1): {time.time() - t0}"
        )
        assert d_read == big_data[start:end]


def test_bytesio(small_data, tmp_path: Path):
    bio = io.BytesIO()
    small_data.write(bio)
    assert ef.patterns(bio) == small_data


def gen_write_patterns():
    data = ef.patterns(np.random.randint(0, 10, size=(16, 256)))
    for i in 2 ** np.arange(0, 10, 2):
        yield ".emc", [data] * i
        yield ".h5", [data] * i


@pytest.mark.parametrize("suffix, data_list", gen_write_patterns())
def test_write_patterns(suffix, data_list):
    with (
        tempfile.NamedTemporaryFile(suffix=suffix) as f0,
        tempfile.NamedTemporaryFile(suffix=suffix) as f1,
    ):
        t = time.time()
        all_data = np.concatenate(data_list)
        all_data.write(f1.name, overwrite=True)
        t1 = time.time() - t
        logging.info(f"speed[single]: {all_data.nbytes * 1e-9 / t1:.2f} GB/s")

        t = time.time()
        ef.write_patterns(data_list, f0.name, buffer_size=2**12, overwrite=True)
        t0 = time.time() - t
        logging.info(
            "speed[multiple; #patterns=%d]: %.2f GB/s",
            len(data_list),
            all_data.nbytes * 1e-9 / t0,
        )

        logging.info(f"speed ratio [{suffix}]: {t1 / t0:.3f}")
        assert ef.patterns(f0.name) == all_data


def test_pattern_mul(big_data):
    mtx = np.random.rand(big_data.num_pix, 10)
    np.testing.assert_almost_equal(big_data @ mtx, np.asarray(big_data) @ mtx)
    mtx = mtx > 0.4
    np.testing.assert_almost_equal(big_data @ mtx, big_data.todense() @ mtx)
    mtx = coo_array(mtx)
    np.testing.assert_equal((big_data @ mtx).todense(), big_data.todense() @ mtx)
    mtx = csr_array(mtx)
    np.testing.assert_equal((big_data @ mtx).todense(), big_data.todense() @ mtx)


@pytest.mark.parametrize("file", ["data_emc", "data_h5", "data_h5_v1"])
def test_PatternsSOneFile_shape(file, request):
    p = Path(request.getfixturevalue(file))
    d = ef.file_patterns(p)
    assert len(d) == d.num_data
    assert d.shape == (d.num_data, d.num_pix)


@pytest.mark.parametrize(
    "file, chunk_size",
    [
        ("data_emc", 1),
        ("data_h5", 1),
        pytest.param("data_h5_v1", 1, marks=pytest.mark.skip("Too slow")),
        ("data_emc", 128),
        ("data_h5", 128),
        ("data_h5_v1", 128),
        ("data_emc", 99999),
        ("data_h5", 99999),
        ("data_h5_v1", 99999),
    ],
)
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [True, False])
def test_PatternsSOneFile_sum(file, axis, keepdims, chunk_size, request):
    p = Path(request.getfixturevalue(file))
    d1 = ef.patterns(p)
    d2 = ef.file_patterns(p)
    assert np.all(
        d1.sum(axis=axis, keepdims=keepdims)
        == d2.sum(axis=axis, keepdims=keepdims, chunk_size=chunk_size)
    )


@pytest.mark.parametrize("file", ["data_emc", "data_h5", "data_h5_v1"])
def test_PatternsSOneFile_getitem(file, request):
    p = Path(request.getfixturevalue(file))
    d1 = ef.patterns(p)
    d2 = ef.file_patterns(p)
    assert d2.sparsity() == d1.sparsity()
    np.testing.assert_equal(d2[3], d1[3])
    assert d2[::2] == d1[::2]
    with temp_seed(12):
        idx = np.random.rand(d1.num_data) > 0.5
    assert d2[idx] == d1[idx]
    idx = np.where(idx)[0]
    assert d2[idx] == d1[idx]
    assert d2[np.array([], dtype=np.int32)].num_data == 0


@pytest.mark.parametrize("file", ["data_emc", "data_h5", "data_h5_v1"])
def test_PatternsSOneFile_spase_pattern(file, request):
    p = Path(request.getfixturevalue(file))
    d1 = ef.patterns(p, end=10)
    d2 = ef.file_patterns(p)
    assert d1 == ef.patterns([d2.sparse_pattern(i) for i in range(10)])


@pytest.mark.parametrize("file", ["data_emc", "data_h5"])
def test_readbuffer(file, request):
    p = Path(request.getfixturevalue(file))
    d1 = ef.patterns(p, end=10)
    d2 = ef.file_patterns(p)
    with d2.open() as fp:
        assert d1 == ef.patterns([fp.sparse_pattern(i) for i in range(10)])


def test_index(big_data, big_dense):
    idx = np.arange(big_data.shape[1])
    idx[0], idx[-1] = idx[-1], idx[0]
    p = big_data[:, idx]
    assert p.check_indices_ordered() is False
    p.ensure_indices_ordered()
    assert p.check_indices_ordered() is True
    assert np.all(np.asarray(p.todense()) == big_dense[:, idx])


@pytest.mark.parametrize("n", range(3))
def test_pow(n, request):
    data = request.getfixturevalue("small_data")
    dense = request.getfixturevalue("small_dense")
    np.testing.assert_equal(dense**n, (data**n).todense())


@pytest.mark.parametrize("shape", [(10, 3), (10, 0)])
def test_zeros(shape):
    np.testing.assert_equal(
        ef.patterns((shape, 0)).todense(),
        np.zeros(shape),
    )


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


def test_pattern_list(data_emc, data_h5):
    p0 = ef.file_patterns(data_emc)
    p1 = ef.file_patterns(data_h5)
    plst = ef.PatternsSOneFileList([p0, p1])
    np.testing.assert_equal(plst[0], p0[0])
    np.testing.assert_equal(plst[len(p0) + 2], p1[2])
    assert len(plst) == len(p0) + len(p1)
    np.testing.assert_equal(plst[1::3][0], p0[1])
    np.testing.assert_equal(plst[1::3][1], p0[4])
    plst.init_idx()
    np.testing.assert_equal(plst.ones, np.concatenate([p0.ones, p1.ones]))
    plst2 = ef.PatternsSOneFileList([plst, p0])
    assert plst2[: len(plst)][: len(p0)] == p0[:]


def test_aaa(small_data):
    print(small_data)
