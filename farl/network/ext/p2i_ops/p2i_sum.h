// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "utility.h"

namespace haya_ext {

template <typename scalar_t> struct p2i_sum_forward_kernel {
  XDEVICE void
  operator()(int id,
             const scalar_t *RESTRICT points,         // npoints x 2
             const scalar_t *RESTRICT point_features, // npoints x channels
             const int32_t *RESTRICT batch_inds,      // npoints
             scalar_t *RESTRICT out, // batch x channels x out_h x out_w
             int batch, int npoints, int channels, int kernel_kind,
             scalar_t kernel_radius, int out_h, int out_w) const {

    // npoints x channels
    if (id >= npoints * channels) {
      return;
    }
    const int point_feature_offset = id;
    const int channel_id = id % channels;
    id = id / channels;
    const int point_id = id % npoints;
    const int32_t batch_id = batch_inds[point_id];
    if (batch_id < 0 || batch_id >= batch) {
      return;
    }

    const scalar_t point_y = points[point_id * 2 + 0];
    const scalar_t point_x = points[point_id * 2 + 1];

    for_each_pixel_near_point(
        point_y, point_x, out_h, out_w, kernel_radius,
        [=] XDEVICE(int y, int x, scalar_t dy, scalar_t dx, scalar_t r) {
          // lock, compare and replace
          const int index =
              ((batch_id * channels + channel_id) * out_h + y) * out_w + x;

          scalar_t weight = 0;
          switch (kernel_kind) {
          case 0: // cos
            weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
            break;
          }

          const scalar_t feature_value = point_features[point_feature_offset];
          atomic_add(&(out[index]), weight * feature_value);
        });
  }
};

template <typename scalar_t> struct p2i_sum_backward_kernel {
  XDEVICE void operator()(
      int id,
      const scalar_t *RESTRICT out_grad, // batch x channels x out_h x out_w
      const scalar_t *RESTRICT points,   // npoints x 2
      const scalar_t *RESTRICT point_features, // npoints x channels
      const int32_t *RESTRICT batch_inds,      // npoints
      scalar_t *RESTRICT points_grad,          // npoints x 2
      scalar_t *RESTRICT point_features_grad,  // npoints x channels
      int batch, int npoints, int channels, int kernel_kind,
      scalar_t kernel_radius, int out_h, int out_w) const {

    // npoints x channels
    if (id >= npoints * channels) {
      return;
    }
    const int point_feature_offset = id;
    const int channel_id = id % channels;
    id = id / channels;
    const int point_id = id % npoints;
    const int32_t batch_id = batch_inds[point_id];
    if (batch_id < 0 || batch_id >= batch) {
      return;
    }

    const int point_y_offset = point_id * 2 + 0;
    const int point_x_offset = point_id * 2 + 1;
    const scalar_t point_y = points[point_y_offset];
    const scalar_t point_x = points[point_x_offset];

    for_each_pixel_near_point(
        point_y, point_x, out_h, out_w, kernel_radius,
        [=] XDEVICE(int y, int x, scalar_t dy, scalar_t dx, scalar_t r) {
          scalar_t weight = 0;
          const scalar_t r_X_PI_DIV_kernel_radius = r * M_PI / kernel_radius;
          switch (kernel_kind) {
          case 0: // cos
            weight = cos(r_X_PI_DIV_kernel_radius) * 0.5 + 0.5;
            break;
          }

          scalar_t point_feature_value = point_features[point_feature_offset];

          // forward: out_value = point_feature_value * weight
          const int out_offset =
              ((batch_id * channels + channel_id) * out_h + y) * out_w + x;
          const scalar_t out_grad_value = out_grad[out_offset];

          // grad of point feature
          atomic_add(&(point_features_grad[point_feature_offset]),
                     out_grad_value * weight);

          // grad of weight
          const scalar_t weight_grad_value =
              out_grad_value * point_feature_value;

          // grad of point_y, point_x
          scalar_t point_y_grad = 0, point_x_grad = 0;
          switch (kernel_kind) {
          case 0: // cos
            // weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
            const scalar_t f = 0.5 * M_PI / kernel_radius;
            point_y_grad = weight_grad_value * sin(r_X_PI_DIV_kernel_radius) *
                           f * dy / max(r, static_cast<scalar_t>(1e-10));
            point_x_grad = weight_grad_value * sin(r_X_PI_DIV_kernel_radius) *
                           f * dx / max(r, static_cast<scalar_t>(1e-10));
            break;
          }
          atomic_add(&(points_grad[point_y_offset]), point_y_grad);
          atomic_add(&(points_grad[point_x_offset]), point_x_grad);
        });
  }
};

struct p2i_sum_op {
  template <typename XPU>
  static at::Tensor
  forward(const at::Tensor &points, const at::Tensor &point_features,
          const at::Tensor &batch_inds, const at::Tensor &background,
          int kernel_kind, double kernel_radius) {
    // inputs:
    //   - points: float, [npoints x 2]
    //   - point_features: float, [npoints x channels]
    //   - batch_inds: int32, [npoints]
    //   - background: float, [batch x channels x out_h x out_w]
    // returns:
    //   - output: float, [batch x channels x out_h x out_w]

    auto npoints = points.size(0);
    auto channels = point_features.size(1);

    auto batch = background.size(0);
    auto out_h = background.size(2);
    auto out_w = background.size(3);

    at::Tensor out = background.clone();

    auto N = npoints * channels;

    AT_DISPATCH_FLOATING_TYPES(
        points.type(), "p2i_sum_op::forward", ([&] {
          kernel<XPU>::launch(
              p2i_sum_forward_kernel<scalar_t>(), N, points.data<scalar_t>(),
              point_features.data<scalar_t>(), batch_inds.data<int32_t>(),
              out.data<scalar_t>(), batch, npoints, channels, kernel_kind,
              static_cast<scalar_t>(kernel_radius), out_h, out_w);
        }));

    cudaCheckError();
    return out;
  }

  template <typename XPU>
  static std::vector<at::Tensor>
  backward(const at::Tensor &out_grad, const at::Tensor &points,
           const at::Tensor &point_features, const at::Tensor &batch_inds,
           int kernel_kind, double kernel_radius) {
    // inputs:
    //  - out_grad: float, [batch x channels x out_h x out_w]
    //  - points: float, [npoints x 2]
    //  - point_features: float, [npoints x channels]
    //  - batch: int32, [npoints]
    // returns:
    //  - points_grad: float, [npoints x 2]
    //  - point_features_grad: float, [npoints x channels]

    auto npoints = points.size(0);
    auto channels = point_features.size(1);

    auto batch = out_grad.size(0);
    auto out_h = out_grad.size(2);
    auto out_w = out_grad.size(3);

    at::Tensor points_grad = at::zeros_like(points);
    at::Tensor point_features_grad = at::zeros_like(point_features);

    auto N = npoints * channels;
    AT_DISPATCH_FLOATING_TYPES(
        points.type(), "p2i_sum_op::backward", ([&] {
          kernel<XPU>::launch(
              p2i_sum_backward_kernel<scalar_t>(), N, out_grad.data<scalar_t>(),
              points.data<scalar_t>(), point_features.data<scalar_t>(),
              batch_inds.data<int32_t>(), points_grad.data<scalar_t>(),
              point_features_grad.data<scalar_t>(), batch, npoints, channels,
              kernel_kind, static_cast<scalar_t>(kernel_radius), out_h, out_w);
        }));

    cudaCheckError();
    return {points_grad, point_features_grad};
  }
};

at::Tensor p2i_sum_forward_gpu(const at::Tensor &points,
                               const at::Tensor &point_features,
                               const at::Tensor &batch_inds,
                               const at::Tensor &background, int kernel_kind,
                               double kernel_radius);

std::vector<at::Tensor> p2i_sum_backward_gpu(const at::Tensor &out_grad,
                                             const at::Tensor &points,
                                             const at::Tensor &point_features,
                                             const at::Tensor &batch_inds,
                                             int kernel_kind,
                                             double kernel_radius);
} // namespace haya_ext