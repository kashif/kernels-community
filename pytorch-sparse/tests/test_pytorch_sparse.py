import pytest
import torch

import kernels

pytorch_sparse = kernels.get_kernel("kernels-community/pytorch-sparse", version=1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_csr(m: int, n: int, nnz: int, device):
    torch.manual_seed(0)
    row = torch.randint(0, m, (nnz,), device=device).sort().values
    col = torch.randint(0, n, (nnz,), device=device)
    value = torch.randn(nnz, device=device)
    rowptr = torch.zeros(m + 1, dtype=torch.long, device=device)
    rowptr[1:] = torch.bincount(row, minlength=m).cumsum(0)
    return row, rowptr, col, value


def dense_of(rowptr, col, value, m, n, device):
    row = pytorch_sparse.ptr2ind(rowptr, col.numel())
    dense = torch.zeros(m, n, device=device)
    # accumulate: the generated pattern contains duplicate (row, col) pairs
    dense.index_put_((row, col), value, accumulate=True)
    return dense


@pytest.mark.kernels_ci
def test_spmm_sum_matches_dense_matmul():
    device = get_device()
    m, n, k = 128, 96, 32
    row, rowptr, col, value = make_csr(m, n, 512, device)
    mat = torch.randn(n, k, device=device)

    out = pytorch_sparse.spmm_sum(row, rowptr, col, value, None, None, mat)
    expected = dense_of(rowptr, col, value, m, n, device) @ mat
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("reduce", ["min", "max"])
def test_spmm_min_max_argout_is_valid(reduce):
    device = get_device()
    m, n, k = 64, 48, 16
    _, rowptr, col, value = make_csr(m, n, 256, device)
    mat = torch.randn(n, k, device=device)

    fn = pytorch_sparse.spmm_min if reduce == "min" else pytorch_sparse.spmm_max
    out, _ = fn(rowptr, col, value, mat)
    assert out.shape == (m, k)
    assert torch.isfinite(out).all()


@pytest.mark.kernels_ci
def test_ind2ptr_ptr2ind_roundtrip():
    device = get_device()
    m = 64
    ind = torch.randint(0, m, (512,), device=device).sort().values
    ptr = pytorch_sparse.ind2ptr(ind, m)
    torch.testing.assert_close(pytorch_sparse.ptr2ind(ptr, ind.numel()), ind)


@pytest.mark.kernels_ci
def test_random_walk_stays_on_edges():
    device = get_device()
    m, n = 64, 64
    _, rowptr, col, _ = make_csr(m, n, 512, device)
    start = torch.arange(m, device=device)

    walk = pytorch_sparse.random_walk(rowptr, col, start, 5)
    assert walk.shape == (m, 6)
    assert walk.min() >= 0 and walk.max() < m


@pytest.mark.kernels_ci
def test_non_diag_mask():
    device = get_device()
    row = torch.tensor([0, 0, 1, 1, 2], device=device)
    col = torch.tensor([0, 1, 1, 2, 2], device=device)
    mask = pytorch_sparse.non_diag_mask(row, col, 3, 3, 0)
    assert mask.dtype == torch.bool
