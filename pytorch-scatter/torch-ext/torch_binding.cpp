#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

// Catch-all rather than per-dispatch-key: the upstream ops wrap
// torch::autograd::Function internally.
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cuda_version() -> int", [] { return scatter::cuda_version(); });

  ops.def("scatter_sum(Tensor src, Tensor index, int dim, Tensor? out, "
          "int? dim_size) -> Tensor",
          &scatter_sum);
  ops.def("scatter_mul(Tensor src, Tensor index, int dim, Tensor? out, "
          "int? dim_size) -> Tensor",
          &scatter_mul);
  ops.def("scatter_mean(Tensor src, Tensor index, int dim, Tensor? out, "
          "int? dim_size) -> Tensor",
          &scatter_mean);
  ops.def("scatter_min(Tensor src, Tensor index, int dim, Tensor? out, "
          "int? dim_size) -> (Tensor, Tensor)",
          &scatter_min);
  ops.def("scatter_max(Tensor src, Tensor index, int dim, Tensor? out, "
          "int? dim_size) -> (Tensor, Tensor)",
          &scatter_max);

  ops.def("segment_sum_csr(Tensor src, Tensor indptr, Tensor? out) -> Tensor",
          &segment_sum_csr);
  ops.def("segment_mean_csr(Tensor src, Tensor indptr, Tensor? out) -> Tensor",
          &segment_mean_csr);
  ops.def("segment_min_csr(Tensor src, Tensor indptr, Tensor? out) "
          "-> (Tensor, Tensor)",
          &segment_min_csr);
  ops.def("segment_max_csr(Tensor src, Tensor indptr, Tensor? out) "
          "-> (Tensor, Tensor)",
          &segment_max_csr);
  ops.def("gather_csr(Tensor src, Tensor indptr, Tensor? out) -> Tensor",
          &gather_csr);

  ops.def("segment_sum_coo(Tensor src, Tensor index, Tensor? out, "
          "int? dim_size) -> Tensor",
          &segment_sum_coo);
  ops.def("segment_mean_coo(Tensor src, Tensor index, Tensor? out, "
          "int? dim_size) -> Tensor",
          &segment_mean_coo);
  ops.def("segment_min_coo(Tensor src, Tensor index, Tensor? out, "
          "int? dim_size) -> (Tensor, Tensor)",
          &segment_min_coo);
  ops.def("segment_max_coo(Tensor src, Tensor index, Tensor? out, "
          "int? dim_size) -> (Tensor, Tensor)",
          &segment_max_coo);
  ops.def("gather_coo(Tensor src, Tensor index, Tensor? out) -> Tensor",
          &gather_coo);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
