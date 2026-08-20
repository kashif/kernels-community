#pragma once

// kernel-builder defines CPU_KERNEL / CUDA_KERNEL / ROCM_KERNEL per variant;
// the upstream sources gate on WITH_CUDA / USE_ROCM.
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
#ifndef WITH_CUDA
#define WITH_CUDA
#endif
#endif

#if defined(ROCM_KERNEL) && !defined(USE_ROCM)
#define USE_ROCM
#endif

#include "macros.h"
#include <torch/torch.h>
