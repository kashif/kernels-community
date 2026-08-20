from typing import Optional, Tuple

import torch

from ._ops import ops


def ind2ptr(ind: torch.Tensor, M: int) -> torch.Tensor:
    return ops.ind2ptr(ind, M)


def ptr2ind(ptr: torch.Tensor, E: int) -> torch.Tensor:
    return ops.ptr2ind(ptr, E)


def non_diag_mask(row: torch.Tensor, col: torch.Tensor, M: int, N: int,
                  k: int) -> torch.Tensor:
    return ops.non_diag_mask(row, col, M, N, k)


def random_walk(rowptr: torch.Tensor, col: torch.Tensor, start: torch.Tensor,
                walk_length: int) -> torch.Tensor:
    return ops.random_walk(rowptr, col, start, walk_length)


def relabel(col: torch.Tensor,
            idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.relabel(col, idx)


def relabel_one_hop(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    value: Optional[torch.Tensor],
    idx: torch.Tensor,
    bipartite: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    return ops.relabel_one_hop(rowptr, col, value, idx, bipartite)


def saint_subgraph(
    idx: torch.Tensor, rowptr: torch.Tensor, row: torch.Tensor,
    col: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return ops.saint_subgraph(idx, rowptr, row, col)


def sample_adj(
    rowptr: torch.Tensor, col: torch.Tensor, idx: torch.Tensor,
    num_neighbors: int, replace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return ops.sample_adj(rowptr, col, idx, num_neighbors, replace)


def ego_k_hop_sample_adj(
    rowptr: torch.Tensor, col: torch.Tensor, idx: torch.Tensor, depth: int,
    num_neighbors: int, replace: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    return ops.ego_k_hop_sample_adj(rowptr, col, idx, depth, num_neighbors,
                                    replace)


def spmm_sum(row: Optional[torch.Tensor], rowptr: torch.Tensor,
             col: torch.Tensor, value: Optional[torch.Tensor],
             colptr: Optional[torch.Tensor], csr2csc: Optional[torch.Tensor],
             mat: torch.Tensor) -> torch.Tensor:
    return ops.spmm_sum(row, rowptr, col, value, colptr, csr2csc, mat)


def spmm_mean(row: Optional[torch.Tensor], rowptr: torch.Tensor,
              col: torch.Tensor, value: Optional[torch.Tensor],
              rowcount: Optional[torch.Tensor],
              colptr: Optional[torch.Tensor], csr2csc: Optional[torch.Tensor],
              mat: torch.Tensor) -> torch.Tensor:
    return ops.spmm_mean(row, rowptr, col, value, rowcount, colptr, csr2csc,
                         mat)


def spmm_min(rowptr: torch.Tensor, col: torch.Tensor,
             value: Optional[torch.Tensor],
             mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.spmm_min(rowptr, col, value, mat)


def spmm_max(rowptr: torch.Tensor, col: torch.Tensor,
             value: Optional[torch.Tensor],
             mat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.spmm_max(rowptr, col, value, mat)


__all__ = [
    "ind2ptr",
    "ptr2ind",
    "non_diag_mask",
    "random_walk",
    "relabel",
    "relabel_one_hop",
    "saint_subgraph",
    "sample_adj",
    "ego_k_hop_sample_adj",
    "spmm_sum",
    "spmm_mean",
    "spmm_min",
    "spmm_max",
]
