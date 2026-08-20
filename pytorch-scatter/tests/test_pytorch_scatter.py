import pytest
import torch

import kernels

pytorch_scatter = kernels.get_kernel("kernels-community/pytorch-scatter", version=1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _csr_indptr(index: torch.Tensor, n: int) -> torch.Tensor:
    ptr = torch.zeros(n + 1, dtype=torch.long, device=index.device)
    ptr[1:] = torch.bincount(index, minlength=n).cumsum(0)
    return ptr


@pytest.mark.kernels_ci
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_scatter_matches_native(reduce):
    device = get_device()
    torch.manual_seed(0)
    src = torch.randn(4096, 16, device=device)
    index = torch.randint(0, 256, (4096,), device=device)

    out = pytorch_scatter.scatter(src, index, dim=0, dim_size=256, reduce=reduce)

    native_reduce = {"sum": "sum", "mean": "mean", "min": "amin", "max": "amax"}[reduce]
    expected = torch.zeros(256, 16, device=device).scatter_reduce_(
        0, index.view(-1, 1).expand(-1, 16), src, reduce=native_reduce,
        include_self=False,
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.kernels_ci
def test_scatter_max_argmax_is_valid():
    """The argmax must actually point at the maximum it reports."""
    device = get_device()
    torch.manual_seed(0)
    src = torch.randn(4096, 16, device=device)
    index = torch.randint(0, 256, (4096,), device=device)

    out, arg = pytorch_scatter.scatter_max(src, index, dim=0, dim_size=256)
    cols = torch.arange(16, device=device)
    gathered = src[arg.clamp(max=src.size(0) - 1), cols]
    torch.testing.assert_close(gathered, out)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_segment_csr_matches_scatter(reduce):
    device = get_device()
    torch.manual_seed(0)
    index = torch.randint(0, 256, (4096,), device=device).sort().values
    src = torch.randn(4096, 16, device=device)
    ptr = _csr_indptr(index, 256)

    out = pytorch_scatter.segment_csr(src, ptr, reduce=reduce)
    expected = pytorch_scatter.scatter(src, index, dim=0, dim_size=256, reduce=reduce)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max"])
def test_segment_coo_matches_scatter(reduce):
    device = get_device()
    torch.manual_seed(0)
    index = torch.randint(0, 256, (4096,), device=device).sort().values
    src = torch.randn(4096, 16, device=device)

    out = pytorch_scatter.segment_coo(src, index, dim_size=256, reduce=reduce)
    expected = pytorch_scatter.scatter(src, index, dim=0, dim_size=256, reduce=reduce)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.kernels_ci
def test_gather_csr_inverts_segment_csr():
    device = get_device()
    torch.manual_seed(0)
    index = torch.randint(0, 256, (4096,), device=device).sort().values
    src = torch.randn(4096, 16, device=device)
    ptr = _csr_indptr(index, 256)

    reduced = pytorch_scatter.segment_csr(src, ptr, reduce="sum")
    expanded = pytorch_scatter.gather_csr(reduced, ptr)
    torch.testing.assert_close(expanded, reduced[index])


@pytest.mark.kernels_ci
def test_scatter_backward():
    device = get_device()
    torch.manual_seed(0)
    src = torch.randn(1024, 8, device=device, requires_grad=True)
    index = torch.randint(0, 64, (1024,), device=device)

    pytorch_scatter.scatter(src, index, dim=0, dim_size=64, reduce="sum").sum().backward()
    torch.testing.assert_close(src.grad, torch.ones_like(src))
