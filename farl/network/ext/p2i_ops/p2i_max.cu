// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "p2i_max.h"

namespace haya_ext {
std::vector<at::Tensor>
p2i_max_forward_gpu(const at::Tensor &points, const at::Tensor &point_features,
                    const at::Tensor &batch_inds, const at::Tensor &background,
                    int kernel_kind, double kernel_radius) {
  return p2i_max_op::forward<gpu_device>(points, point_features, batch_inds,
                                         background, kernel_kind,
                                         kernel_radius);
}

std::vector<at::Tensor> p2i_max_backward_gpu(const at::Tensor &out_grad,
                                             const at::Tensor &out_point_ids,
                                             const at::Tensor &points,
                                             const at::Tensor &point_features,
                                             int kernel_kind,
                                             double kernel_radius) {
  return p2i_max_op::backward<gpu_device>(out_grad, out_point_ids, points,
                                          point_features, kernel_kind,
                                          kernel_radius);
}
} // namespace haya_ext