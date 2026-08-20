#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cuda_version() -> int", [] { return sparse::cuda_version(); });

  ops.def("ind2ptr(Tensor ind, int M) -> Tensor", &ind2ptr);
  ops.def("ptr2ind(Tensor ptr, int E) -> Tensor", &ptr2ind);

  ops.def("non_diag_mask(Tensor row, Tensor col, int M, int N, int k) "
          "-> Tensor",
          &non_diag_mask);

  ops.def("random_walk(Tensor rowptr, Tensor col, Tensor start, "
          "int walk_length) -> Tensor",
          &random_walk);

  ops.def("relabel(Tensor col, Tensor idx) -> (Tensor, Tensor)", &relabel);
  ops.def("relabel_one_hop(Tensor rowptr, Tensor col, Tensor? value, "
          "Tensor idx, bool bipartite) -> (Tensor, Tensor, Tensor?, Tensor)",
          &relabel_one_hop);

  ops.def("saint_subgraph(Tensor idx, Tensor rowptr, Tensor row, Tensor col) "
          "-> (Tensor, Tensor, Tensor)",
          &subgraph);

  ops.def("sample_adj(Tensor rowptr, Tensor col, Tensor idx, "
          "int num_neighbors, bool replace) "
          "-> (Tensor, Tensor, Tensor, Tensor)",
          &sample_adj);

  ops.def("ego_k_hop_sample_adj(Tensor rowptr, Tensor col, Tensor idx, "
          "int depth, int num_neighbors, bool replace) "
          "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
          &ego_k_hop_sample_adj);


  ops.def("spmm_sum(Tensor? row, Tensor rowptr, Tensor col, Tensor? value, "
          "Tensor? colptr, Tensor? csr2csc, Tensor mat) -> Tensor",
          &spmm_sum);
  ops.def("spmm_mean(Tensor? row, Tensor rowptr, Tensor col, Tensor? value, "
          "Tensor? rowcount, Tensor? colptr, Tensor? csr2csc, Tensor mat) "
          "-> Tensor",
          &spmm_mean);
  ops.def("spmm_min(Tensor rowptr, Tensor col, Tensor? value, Tensor mat) "
          "-> (Tensor, Tensor)",
          &spmm_min);
  ops.def("spmm_max(Tensor rowptr, Tensor col, Tensor? value, Tensor mat) "
          "-> (Tensor, Tensor)",
          &spmm_max);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
