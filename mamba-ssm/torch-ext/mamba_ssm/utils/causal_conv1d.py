"""Optional causal-conv1d dependency.

Resolution order for the fused short-convolution helpers:

1. the separately-installed ``causal_conv1d`` package (``pip install causal-conv1d``);
2. otherwise, if the ``kernels`` library is available, the Hub build
   ``kernels-community/causal-conv1d``, loaded lazily on first use;
3. otherwise every symbol is ``None`` and callers fall back to the
   pure-PyTorch / Triton implementations.
"""

from functools import lru_cache

_HUB_REPO_ID = "kernels-community/causal-conv1d"
# Pin to a published revision/tag to keep in sync, e.g. revision="v0.0.2".
_HUB_REVISION = None


@lru_cache(maxsize=1)
def _hub_causal_conv1d():
    from kernels import get_kernel

    return get_kernel(_HUB_REPO_ID, revision=_HUB_REVISION)


def _hub_attr(name):
    """A thin wrapper that resolves ``name`` from the Hub kernel on first call."""

    def wrapper(*args, **kwargs):
        return getattr(_hub_causal_conv1d(), name)(*args, **kwargs)

    wrapper.__name__ = name
    return wrapper


try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    try:
        import kernels  # noqa: F401

        causal_conv1d_fn = _hub_attr("causal_conv1d_fn")
        causal_conv1d_update = _hub_attr("causal_conv1d_update")
        causal_conv1d_varlen_states = _hub_attr("causal_conv1d_varlen_states")
    except ImportError:
        causal_conv1d_fn = None
        causal_conv1d_update = None
        causal_conv1d_varlen_states = None
