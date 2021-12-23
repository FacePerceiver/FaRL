// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "utility.h"

namespace haya_ext {

template <typename scalar_t> struct p2i_max_forward_kernel {
  XDEVICE void
  operator()(int id,
             const scalar_t *RESTRICT points,         // npoints x 2
             const scalar_t *RESTRICT point_features, // npoints x channels
             const int32_t *RESTRICT batch_inds,      // npoints
             scalar_t *RESTRICT out, // batch x channels x out_h x out_w
             int32_t *RESTRICT out_point_ids, // batch x channels x out_h x
                                              // out_w, stores point_ids
             int32_t *RESTRICT out_lock, // batch x channels x out_h x out_w
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
          scalar_t weight = 0;
          switch (kernel_kind) {
          case 0: // cos
            weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
            break;
          case 1: // gaussian_awing
            weight = exp(-r * r * 16 / 2 / kernel_radius / kernel_radius);
            break;
          }

          const scalar_t weighted_value =
              point_features[point_feature_offset] * weight;

          // lock, compare and replace
          const int index =
              ((batch_id * channels + channel_id) * out_h + y) * out_w + x;
          bool locked = false;
          do {
            if (locked = atomic_cas(&out_lock[index], 0, 1) == 0) {
              const scalar_t current_value =
                  atomic_add(&out[index], static_cast<scalar_t>(0));
              if (current_value < weighted_value) {
                atomic_exch(&(out[index]), weighted_value);
                atomic_exch(&(out_point_ids[index]), point_id);
              }
              atomic_exch(&out_lock[index], 0);
            }
          } while (!locked);
        });
  }
};

template <typename scalar_t> struct p2i_max_backward_kernel {
  XDEVICE void operator()(
      int id,
      const scalar_t *RESTRICT out_grad, // batch x channels x out_h x out_w
      int32_t *RESTRICT out_point_ids,   // batch x channels x out_h x
                                         // out_w, stores point_ids
      const scalar_t *RESTRICT points,   // npoints x 2
      const scalar_t *RESTRICT point_features, // npoints x channels
      scalar_t *RESTRICT points_grad,          // npoints x 2
      scalar_t *RESTRICT point_features_grad,  // npoints x channels
      scalar_t *RESTRICT background_grad, // batch x channels x out_h x out_w
      int batch, int npoints, int channels, int kernel_kind,
      scalar_t kernel_radius, int out_h, int out_w) const {

    // batch x channels x out_h x out_w
    const int index = id;
    if (id >= batch * channels * out_h * out_w) {
      return;
    }
    const int x = id % out_w;
    id /= out_w;
    const int y = id % out_h;
    id /= out_h;

    const int channel_id = id % channels;
    id /= channels;
    const int batch_id = id % batch;

    const scalar_t out_grad_value = out_grad[index];

    const int point_id = out_point_ids[index];
    if (point_id < 0) { // background here, no grads to points or point_features
      atomic_add(&(background_grad[index]), out_grad_value);
      return;
    }

    const int point_y_offset = point_id * 2 + 0;
    const int point_x_offset = point_id * 2 + 1;
    const scalar_t point_y = points[point_y_offset];
    const scalar_t point_x = points[point_x_offset];

    const scalar_t dx = x - point_x, dy = y - point_y;
    const scalar_t r = sqrt(dx * dx + dy * dy);

    scalar_t weight = 0;
    switch (kernel_kind) {
    case 0: // cos
      weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
      break;
    case 1: // gaussian_awing (sigma=0.25)
      weight = exp(-(dx*dx + dy+dy) * 16.0 / 2.0 / kernel_radius / kernel_radius);
      break;
    }

    const int point_feature_offset = point_id * channels + channel_id;
    const scalar_t point_feature_value = point_features[point_feature_offset];

    // grad of point feature
    atomic_add(&(point_features_grad[point_feature_offset]),
               out_grad_value * weight);

    // grad of weight
    const scalar_t weight_grad_value = out_grad_value * point_feature_value;

    // grad of point_y, point_x
    scalar_t point_y_grad = 0, point_x_grad = 0;
    switch (kernel_kind) {
    case 0: { // cos
      // weight = cos(r * M_PI / kernel_radius) * 0.5 + 0.5;
      const scalar_t k = weight_grad_value * sin(r * M_PI / kernel_radius) *
                         0.5 * M_PI / kernel_radius /
                         max(r, static_cast<scalar_t>(1e-10));
      point_y_grad = k * dy;
      point_x_grad = k * dx;
      break;
    }
    case 1: { // gaussian_awing (sigma=0.25)
      // weight = exp(-r * r * 16 / 2 / kernel_radius / kernel_radius);
      const scalar_t c =
          static_cast<scalar_t>(16.0f) / 2 / kernel_radius / kernel_radius;
      const scalar_t k = weight_grad_value * exp(-c * r * r) * (-2 * c);
      point_y_grad = k * dy;
      point_x_grad = k * dx;
      break;
    }
    }
    atomic_add(&(points_grad[point_y_offset]), point_y_grad);
    atomic_add(&(points_grad[point_x_offset]), point_x_grad);
  }
};

struct p2i_max_op {
  template <typename XPU>
  static std::vector<at::Tensor>
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
    //   - out_point_ids: int32, [batch x channels x out_h x out_w]

    auto npoints = points.size(0);
    auto channels = point_features.size(1);

    auto batch = background.size(0);
    auto out_h = background.size(2);
    auto out_w = background.size(3);

    at::Tensor out = background.clone();
    at::Tensor out_lock = at::zeros({batch, channels, out_h, out_w},
                                    points.options().dtype(at::kInt));
    at::Tensor out_point_ids = at::full({batch, channels, out_h, out_w}, -1,
                                        background.options().dtype(at::kInt));

    auto N = npoints * channels;

    AT_DISPATCH_FLOATING_TYPES(
        points.type(), "p2i_max_op::forward", ([&] {
          kernel<XPU>::launch(
              p2i_max_forward_kernel<scalar_t>(), N, points.data<scalar_t>(),
              point_features.data<scalar_t>(), batch_inds.data<int32_t>(),
              out.data<scalar_t>(), out_point_ids.data<int32_t>(),
              out_lock.data<int32_t>(), batch, npoints, channels, kernel_kind,
              static_cast<scalar_t>(kernel_radius), out_h, out_w);
        }));

    cudaCheckError();
    return {out, out_point_ids};
  }

  template <typename XPU>
  static std::vector<at::Tensor>
  backward(const at::Tensor &out_grad, const at::Tensor &out_point_ids,
           const at::Tensor &points, const at::Tensor &point_features,
           int kernel_kind, double kernel_radius) {
    // inputs:
    //  - out_grad: float, [batch x channels x out_h x out_w]
    //  - out_point_ids: int32, [batch x channels x out_h x out_w]
    //  - points: float, [npoints x 2]
    //  - point_features: float, [npoints x channels]
    // returns:
    //  - points_grad: float, [npoints x 2]
    //  - point_features_grad: float, [npoints x channels]
    //  - background_grad: float, [batch x channels x out_h x out_w]

    auto npoints = points.size(0);
    auto channels = point_features.size(1);

    auto batch = out_grad.size(0);
    auto out_h = out_grad.size(2);
    auto out_w = out_grad.size(3);

    at::Tensor points_grad = at::zeros_like(points);
    at::Tensor point_features_grad = at::zeros_like(point_features);
    at::Tensor background_grad = at::zeros_like(out_grad);

    auto N = batch * channels * out_h * out_w;
    AT_DISPATCH_FLOATING_TYPES(
        points.type(), "p2i_max_op::backward", ([&] {
          kernel<XPU>::launch(
              p2i_max_backward_kernel<scalar_t>(), N, out_grad.data<scalar_t>(),
              out_point_ids.data<int32_t>(), points.data<scalar_t>(),
              point_features.data<scalar_t>(), points_grad.data<scalar_t>(),
              point_features_grad.data<scalar_t>(),
              background_grad.data<scalar_t>(), batch, npoints, channels,
              kernel_kind, static_cast<scalar_t>(kernel_radius), out_h, out_w);
        }));

    cudaCheckError();
    return {points_grad, point_features_grad, background_grad};
  }
};

std::vector<at::Tensor>
p2i_max_forward_gpu(const at::Tensor &points, const at::Tensor &point_features,
                    const at::Tensor &batch_inds, const at::Tensor &background,
                    int kernel_kind, double kernel_radius);

std::vector<at::Tensor> p2i_max_backward_gpu(const at::Tensor &out_grad,
                                             const at::Tensor &out_point_ids,
                                             const at::Tensor &points,
                                             const at::Tensor &point_features,
                                             int kernel_kind,
                                             double kernel_radius);
} // namespace haya_ext