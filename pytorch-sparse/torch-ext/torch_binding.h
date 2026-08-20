#pragma once

#include <torch/torch.h>

namespace sparse {
int64_t cuda_version() noexcept;
} // namespace sparse

torch::Tensor ind2ptr(torch::Tensor ind, int64_t M);
torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E);

torch::Tensor non_diag_mask(torch::Tensor row, torch::Tensor col, int64_t M,
                            int64_t N, int64_t k);

torch::Tensor random_walk(torch::Tensor rowptr, torch::Tensor col,
                          torch::Tensor start, int64_t walk_length);

std::tuple<torch::Tensor, torch::Tensor> relabel(torch::Tensor col,
                                                 torch::Tensor idx);

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop(torch::Tensor rowptr, torch::Tensor col,
                std::optional<torch::Tensor> optional_value, torch::Tensor idx,
                bool bipartite);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
subgraph(torch::Tensor idx, torch::Tensor rowptr, torch::Tensor row,
         torch::Tensor col);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
           int64_t num_neighbors, bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
ego_k_hop_sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
                     int64_t depth, int64_t num_neighbors, bool replace);


torch::Tensor spmm_sum(std::optional<torch::Tensor> opt_row,
                       torch::Tensor rowptr, torch::Tensor col,
                       std::optional<torch::Tensor> opt_value,
                       std::optional<torch::Tensor> opt_colptr,
                       std::optional<torch::Tensor> opt_csr2csc,
                       torch::Tensor mat);

torch::Tensor spmm_mean(std::optional<torch::Tensor> opt_row,
                        torch::Tensor rowptr, torch::Tensor col,
                        std::optional<torch::Tensor> opt_value,
                        std::optional<torch::Tensor> opt_rowcount,
                        std::optional<torch::Tensor> opt_colptr,
                        std::optional<torch::Tensor> opt_csr2csc,
                        torch::Tensor mat);

std::tuple<torch::Tensor, torch::Tensor>
spmm_min(torch::Tensor rowptr, torch::Tensor col,
         std::optional<torch::Tensor> opt_value, torch::Tensor mat);

std::tuple<torch::Tensor, torch::Tensor>
spmm_max(torch::Tensor rowptr, torch::Tensor col,
         std::optional<torch::Tensor> opt_value, torch::Tensor mat);
