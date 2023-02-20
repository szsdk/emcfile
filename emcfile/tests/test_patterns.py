import gc
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from psutil import Process
from scipy.sparse import coo_matrix, csr_matrix

import emcfile as ef


@pytest.fixture()
def data():
    np.random.seed(123)
    num_data, num_pix = 1000, 4096
    dense = (10 * np.random.rand(num_data, num_pix) ** 5).astype(int)
    den_data = ef.patterns(dense)
    np.testing.assert_almost_equal(den_data.get_mean_count(), np.sum(dense) / num_data)
    np.testing.assert_equal(den_data.todense(), dense)
    return den_data


@pytest.fixture()
def data_emc(tmp_path_factory, data):
    fn = tmp_path_factory.mktemp("data") / "photon.emc"
    data.write(fn)
    return fn


@pytest.fixture()
def data_h5(tmp_path_factory, data):
    fn = tmp_path_factory.mktemp("data") / "photon.h5::patterns"
    data.write(fn)
    return fn


@pytest.fixture()
def data_h5_v1(tmp_path_factory, data):
    fn = tmp_path_factory.mktemp("data") / "photon.h5::patterns"
    data.write(fn, h5version="1")
    return fn


def test_operation(data):
    data.get_mean_count()
    data.nbytes
    data.sparsity()


def gen_pattern_inputs():
    np.random.seed(123)
    dense = (5 * np.random.rand(400, 128**2) ** 5).astype("i4")
    ref = ef.patterns(dense)
    yield ref, ref
    yield dense, ref
    yield coo_matrix(dense), ref
    yield csr_matrix(dense), ref


@pytest.mark.parametrize("inp, ref", gen_pattern_inputs())
def test_patterns(inp, ref):
    p = ef.patterns(inp)
    assert p == ref


def test_getitem(data):
    for i in np.random.choice(data.num_data, 5):
        assert np.sum(data[i] == 1) == data.ones[i]

    mask = np.random.rand(data.num_data) < 0.5
    idx = np.where(mask)[0]
    t0 = time.time()
    subdata1 = data[mask]
    subdata2 = data[idx]
    logging.info(f"Select dataset: {(time.time()-t0)/2}s")
    assert subdata1 == subdata2
    for _ in np.random.choice(subdata1.num_data, 5):
        assert np.all(subdata1[_] == data[idx[_]])


def test_concatenate(data):
    patterns = [ef.patterns(data.num_pix)] + [
        ef.patterns(data, start=i * 10, end=(i + 1) * 10) for i in range(5)
    ]
    ans = np.concatenate(patterns)
    assert data[:50] == ans
    ans = np.concatenate(patterns, casting="destroy")
    assert data[:50] == ans

    patterns = [ef.patterns(np.full((10000, 1000), 2)) for _ in range(2)]
    process = Process()
    m0 = process.memory_info().rss
    ans = np.concatenate(patterns, casting="destroy")
    gc.collect()
    m1 = process.memory_info().rss
    assert (m1 - m0) < ans.nbytes * 0.9
    assert len(patterns) == 0


def test_sum():
    num_data, num_pix = 30, 4
    dense = (10 * np.random.rand(num_data, num_pix) ** 5).astype(int)
    data = ef.patterns(dense)
    np.testing.assert_equal(data.sum(), dense.sum())
    np.testing.assert_equal(data.sum(axis=1), dense.sum(axis=1))
    np.testing.assert_equal(
        data.sum(axis=1, keepdims=True), dense.sum(axis=1, keepdims=True)
    )
    np.testing.assert_equal(data.sum(axis=0), dense.sum(axis=0))
    np.testing.assert_equal(
        data.sum(axis=0, keepdims=True), dense.sum(axis=0, keepdims=True)
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
def test_fileio(suffix, kargs, data):
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        t0 = time.time()
        data.write(f.name, overwrite=True, **kargs)
        logging.info(
            "Writing %d patterns to %s file(%s): {time.time()-t0}",
            data.num_data,
            suffix,
            kargs,
        )

        start = data.num_data // 3
        end = start * 2
        t0 = time.time()
        d_read = ef.patterns(f.name, start=start, end=end)
        logging.info(
            f"Reading {d_read.num_data} patterns from h5 file(v1): {time.time()-t0}"
        )
        assert d_read == data[start:end]


def gen_write_patterns():
    data = ef.patterns(np.random.randint(0, 10, size=(16, 256)))
    for i in 2 ** np.arange(0, 10, 2):
        yield ".emc", [data] * i
        yield ".h5", [data] * i


@pytest.mark.parametrize("suffix, data_list", gen_write_patterns())
def test_write_patterns(suffix, data_list):
    with tempfile.NamedTemporaryFile(suffix=suffix) as f0, tempfile.NamedTemporaryFile(
        suffix=suffix
    ) as f1:
        t = time.time()
        all_data = np.concatenate(data_list)
        all_data.write(f1.name, overwrite=True)
        t1 = time.time() - t
        logging.info(f"speed[single]: {all_data.nbytes * 1e-9 /t1:.2f} GB/s")

        t = time.time()
        ef.write_patterns(data_list, f0.name, overwrite=True)
        t0 = time.time() - t
        logging.info(
            "speed[multiple; #patterns=%d]: %.2f GB/s",
            len(data_list),
            all_data.nbytes * 1e-9 / t0,
        )

        logging.info(f"speed ratio [{suffix}]: {t1 / t0:.3f}")
        assert ef.patterns(f0.name) == all_data


def test_pattern_mul(data):
    mtx = np.random.rand(data.num_pix, 10)
    np.testing.assert_almost_equal(data @ mtx, np.asarray(data) @ mtx)
    mtx = mtx > 0.4
    np.testing.assert_almost_equal(data @ mtx, data.todense() @ mtx)
    mtx = coo_matrix(mtx)
    np.testing.assert_equal((data @ mtx).todense(), data.todense() @ mtx)
    mtx = csr_matrix(mtx)
    np.testing.assert_equal((data @ mtx).todense(), data.todense() @ mtx)


@pytest.mark.parametrize("file", ["data_emc", "data_h5"])
def test_PatternsSOneFile_getitem(file, request):
    np.random.seed(12)
    p = Path(request.getfixturevalue(file))
    d1 = ef.patterns(p)
    d2 = ef.file_patterns(p)
    assert d2.sparsity() == d1.sparsity()
    np.testing.assert_equal(d2[3], d1[3])
    assert d2[::2] == d1[::2]
    idx = np.random.rand(d1.num_data) > 0.5
    assert d2[idx] == d1[idx]
    idx = np.where(idx)[0]
    assert d2[idx] == d1[idx]
    assert d2[np.array([], dtype=np.int32)].num_data == 0


def test_index(data):
    idx = np.arange(data.shape[1])
    idx[0], idx[-1] = idx[-1], idx[0]
    p = data[:, idx]
    assert p.check_indices_ordered() is False
    p.ensure_indices_ordered()
    assert p.check_indices_ordered() is True
    assert np.all(np.asarray(p.todense()) == np.asarray(data.todense())[:, idx])
