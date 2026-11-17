import math

import pytest
import torch
import torch.nn.functional as F

import kernels

liger_kernels = kernels.get_kernel("kernels-community/liger-kernels", version=4)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA or ROCm"),
]


@pytest.mark.parametrize("fused", [False, True], ids=["ce", "fused-linear-ce"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("vocab_size", [3, 32769], ids=["single-block", "multi-block"])
def test_confident_target_loss_and_gradients(fused, dtype, vocab_size):
    margins = torch.tensor([-20.0, 0.0, 10.0, 20.0, 40.0], device="cuda")
    target = torch.tensor([0, 1, 2, 0, vocab_size - 1], device="cuda")
    logits = torch.zeros(5, vocab_size, device="cuda", dtype=dtype)
    target_logits = (margins + math.log(vocab_size - 1)).to(dtype)
    logits.scatter_(1, target[:, None], target_logits[:, None])

    reference = logits.double().requires_grad_()
    target_mask = F.one_hot(target, vocab_size).bool()
    # Ordinary cross_entropy also collapses in float64 for sufficiently large
    # margins. Excluding the target keeps both the reference loss and gradient.
    other_lse = reference.masked_fill(target_mask, -torch.inf).logsumexp(dim=-1)
    expected = F.softplus(other_lse - reference.gather(1, target[:, None]).squeeze(1))

    if fused:
        inputs = torch.eye(5, device="cuda", dtype=dtype, requires_grad=True)
        weight = logits.t().contiguous().requires_grad_()
        actual = liger_kernels.layers.liger_fused_linear_cross_entropy(
            inputs, weight, target, reduction="none"
        )
        actual_grad = torch.autograd.grad(actual.sum(), weight)[0].t()
    else:
        inputs = logits.clone().requires_grad_()
        actual = liger_kernels.cross_entropy.LigerCrossEntropyFunction.apply(
            inputs, target, None, -100, 0.0, 0.0, "none"
        )[0]
        actual_grad = torch.autograd.grad(actual.sum(), inputs)[0]

    expected_grad = torch.autograd.grad(expected.sum(), reference)[0]

    # Zero absolute tolerance is essential: the lost values are much smaller
    # than the usual tolerances, but still representable in both input dtypes.
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(actual.double(), expected, atol=0, rtol=rtol)
    torch.testing.assert_close(actual_grad.double(), expected_grad, atol=0, rtol=rtol)

    # A common logit shift cannot change CE. Scale this check by the gradient
    # magnitude so it also detects a missing target gradient for tiny tails.
    residual = actual_grad.double().sum(dim=-1)
    scale = actual_grad.double().abs().sum(dim=-1)
    torch.testing.assert_close(
        residual / scale, torch.zeros_like(residual), atol=rtol, rtol=0
    )
