// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>

#include "p2i_max.h"
#include "p2i_sum.h"

using namespace haya_ext;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("p2i_sum_forward_gpu", &p2i_sum_forward_gpu, "p2i sum forward (CUDA)");
  m.def("p2i_sum_backward_gpu", &p2i_sum_backward_gpu,
        "p2i sum backward (CUDA)");

  m.def("p2i_max_forward_gpu", &p2i_max_forward_gpu, "p2i max forward (CUDA)");
  m.def("p2i_max_backward_gpu", &p2i_max_backward_gpu,
        "p2i max backward (CUDA)");
}