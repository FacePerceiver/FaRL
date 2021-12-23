// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef __CUDACC__
#include <algorithm>
#include <atomic>
using std::max;
using std::min;
#endif

#include "common.h"

namespace haya_ext {

// atomic_add
template <typename T> XDEVICE T atomic_add(T *addr, T v) {
#ifdef __CUDACC__
  return atomicAdd(addr, v);
#else
  return *addr += v;
#endif
}

#if defined(__CUDACC__) && __CUDA_ARCH__ < 600 // old cuda
__forceinline__ __device__ double atomic_add(double *addr, double v) {
  unsigned long long int *address_as_ull = (unsigned long long int *)addr;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(v + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// atomic_cas
template <typename T> XDEVICE T atomic_cas(T *addr, T compare, T val) {
#ifdef __CUDACC__
  return atomicCAS(addr, compare, val);
#else
  std::atomic<T> this_val(*addr);
  this_val.compare_exchange_weak(compare, val);
  return *addr = this_val.load();
#endif
}

// atomic_exch
template <typename T> XDEVICE T atomic_exch(T *addr, T v) {
#ifdef __CUDACC__
  return atomicExch(addr, v);
#else
  T rd = *addr;
  *addr = v;
  return rd;
#endif
}

#ifdef __CUDACC__
__forceinline__ __device__ double atomic_exch(double *addr, double v) {
  return atomicExch((unsigned long long int *)addr, __double_as_longlong(v));
}
#endif

// square
template <typename T> XINLINE T square(T v) { return v * v; }

// clamp
template <typename T> XINLINE T clamp(T v, T min_v, T max_v) {
  v = v < min_v ? min_v : v;
  v = v > max_v ? max_v : v;
  return v;
}

// swap
template <typename T> XINLINE void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

// min_n & max_n
template <typename T> XINLINE T min_n(T a) { return a; }
template <typename T> XINLINE T max_n(T a) { return a; }
template <typename T, typename... Ts> XINLINE T min_n(T a, Ts... as) {
  return min(a, min_n(as...));
}
template <typename T, typename... Ts> XINLINE T max_n(T a, Ts... as) {
  return max(a, max_n(as...));
}

// argmin
template <typename scalar_t, int N>
XINLINE int argmin(const scalar_t (&vs)[N], scalar_t *min_v = nullptr) {
  scalar_t min_vv = vs[0];
  int min_i = 0;
#pragma unroll
  for (int i = 1; i < N; i++) {
    if (vs[i] < min_vv) {
      min_vv = vs[i];
      min_i = i;
    }
  }
  if (min_v) {
    *min_v = min_vv;
  }
  return min_i;
}

// for_each_pixel_near_point
template <typename scalar_t, typename FunT>
XDEVICE void for_each_pixel_near_point(scalar_t py, scalar_t px, int out_h,
                                       int out_w, scalar_t radius,
                                       FunT callback) {
  int min_x = clamp<int>(floor(px - radius), 0, out_w - 1);
  int max_x = clamp<int>(ceil(px + radius), 0, out_w - 1);
  int min_y = clamp<int>(floor(py - radius), 0, out_h - 1);
  int max_y = clamp<int>(ceil(py + radius), 0, out_h - 1);
  for (int x = min_x; x <= max_x; x++) {
    for (int y = min_y; y <= max_y; y++) {
      scalar_t dx = x - px;
      scalar_t dy = y - py;
      scalar_t r = sqrt(dx * dx + dy * dy);
      if (r <= radius) {
        callback(y, x, dy, dx, r);
      }
    }
  }
}

// is_on_left_side
template <typename scalar_t>
XINLINE bool is_on_left_side(const scalar_t *p, const scalar_t *a,
                             const scalar_t *b) {
  scalar_t data[9] = {a[0], a[1], 1, b[0], b[1], 1, p[0], p[1], 1};
  scalar_t tmp1 = data[0 * 3 + 0] * (data[1 * 3 + 1] * data[2 * 3 + 2] -
                                     data[1 * 3 + 2] * data[2 * 3 + 1]);
  scalar_t tmp2 = data[0 * 3 + 1] * (data[1 * 3 + 0] * data[2 * 3 + 2] -
                                     data[1 * 3 + 2] * data[2 * 3 + 0]);
  scalar_t tmp3 = data[0 * 3 + 2] * (data[1 * 3 + 0] * data[2 * 3 + 1] -
                                     data[1 * 3 + 1] * data[2 * 3 + 0]);
  return tmp1 - tmp2 + tmp3 >= 0;
}

// is_in_triangle
template <typename scalar_t>
XINLINE bool is_in_triangle(const scalar_t *p, const scalar_t *a,
                            const scalar_t *b, const scalar_t *c) {
  bool lab = is_on_left_side(p, a, b);
  bool lbc = is_on_left_side(p, b, c);
  bool lca = is_on_left_side(p, c, a);
  return lab == lbc && lbc == lca;
}

// get_barycentric_coefficients
template <typename scalar_t>
XINLINE void get_barycentric_coefficients(const scalar_t *a, const scalar_t *b,
                                          const scalar_t *c, scalar_t *coeffs) {
  // clang-format off
    /* #bc[0]:
       bx cy - by cx - bx py + by px + cx py - cy px
       ---------------------------------------------
       ax by - ay bx - ax cy + ay cx + bx cy - by cx

       #bc[1]:
       ax cy - ay cx - ax py + ay px + cx py - cy px
     - ---------------------------------------------
       ax by - ay bx - ax cy + ay cx + bx cy - by cx

       #bc[2]:
       ax by - ay bx - ax py + ay px + bx py - by px
       ---------------------------------------------
       ax by - ay bx - ax cy + ay cx + bx cy - by cx
     */
  // clang-format on

  scalar_t ax = a[0], ay = a[1];
  scalar_t bx = b[0], by = b[1];
  scalar_t cx = c[0], cy = c[1];

  scalar_t s = ax * by - ay * bx - ax * cy + ay * cx + bx * cy - by * cx;
  s = s > 0 ? max(s, 1e-6) : min(s, -1e-6);

  // bc[0] = (bx * cy - by * cx - bx * py + by * px + cx * py - cy * px) / s;
  coeffs[0 * 3 + 0] = (by - cy) / s;           // px
  coeffs[0 * 3 + 1] = (cx - bx) / s;           // py
  coeffs[0 * 3 + 2] = (bx * cy - cx * by) / s; // constant

  // bc[1] = (ax * cy - ay * cx - ax * py + ay * px + cx * py - cy * px) / (-s);
  coeffs[1 * 3 + 0] = (cy - ay) / s;           // px
  coeffs[1 * 3 + 1] = (ax - cx) / s;           // py
  coeffs[1 * 3 + 2] = (cx * ay - ax * cy) / s; // constant

  // bc[2] = (ax * by - ay * bx - ax * py + ay * px + bx * py - by * px) / s;
  coeffs[2 * 3 + 0] = (ay - by) / s;           // px
  coeffs[2 * 3 + 1] = (bx - ax) / s;           // py
  coeffs[2 * 3 + 2] = (ax * by - bx * ay) / s; // constant
}

// get_barycentric_coefficients_backward
template <typename scalar_t>
XINLINE void get_barycentric_coefficients_backward(
    const scalar_t *a, const scalar_t *b, const scalar_t *c,
    const scalar_t *coeffs_grad, scalar_t *a_grad, scalar_t *b_grad,
    scalar_t *c_grad) {
  scalar_t ax = a[0], ay = a[1];
  scalar_t bx = b[0], by = b[1];
  scalar_t cx = c[0], cy = c[1];

  scalar_t s = ax * by - ay * bx - ax * cy + ay * cx + bx * cy - by * cx;
  s = s > 0 ? max(s, 1e-6) : min(s, -1e-6);

  scalar_t s_grad = 0.0f;
  scalar_t ax_grad = 0.0f, ay_grad = 0.0f, bx_grad = 0.0f, by_grad = 0.0f,
           cx_grad = 0.0f, cy_grad = 0.0f;

  // coeffs[0 * 3 + 0] = (by - cy) / s;           // px
  // coeffs[0 * 3 + 1] = (cx - bx) / s;           // py
  // coeffs[0 * 3 + 2] = (bx * cy - cx * by) / s; // constant
  s_grad -= coeffs_grad[0 * 3 + 0] * (by - cy) / s / s;
  s_grad -= coeffs_grad[0 * 3 + 1] * (cx - bx) / s / s;
  s_grad -= coeffs_grad[0 * 3 + 2] * (bx * cy - cx * by) / s / s;
  by_grad += coeffs_grad[0 * 3 + 0] / s;
  cy_grad -= coeffs_grad[0 * 3 + 0] / s;
  cx_grad += coeffs_grad[0 * 3 + 1] / s;
  bx_grad -= coeffs_grad[0 * 3 + 1] / s;
  bx_grad += coeffs_grad[0 * 3 + 2] * cy / s;
  cy_grad += coeffs_grad[0 * 3 + 2] * bx / s;
  cx_grad -= coeffs_grad[0 * 3 + 2] * by / s;
  by_grad -= coeffs_grad[0 * 3 + 2] * cx / s;

  // coeffs[1 * 3 + 0] = (cy - ay) / s;           // px
  // coeffs[1 * 3 + 1] = (ax - cx) / s;           // py
  // coeffs[1 * 3 + 2] = (cx * ay - ax * cy) / s; // constant
  s_grad -= coeffs_grad[1 * 3 + 0] * (cy - ay) / s / s;
  s_grad -= coeffs_grad[1 * 3 + 1] * (ax - cx) / s / s;
  s_grad -= coeffs_grad[1 * 3 + 2] * (cx * ay - ax * cy) / s / s;
  cy_grad += coeffs_grad[1 * 3 + 0] / s;
  ay_grad -= coeffs_grad[1 * 3 + 0] / s;
  ax_grad += coeffs_grad[1 * 3 + 1] / s;
  cx_grad -= coeffs_grad[1 * 3 + 1] / s;
  cx_grad += coeffs_grad[1 * 3 + 2] * ay / s;
  ay_grad += coeffs_grad[1 * 3 + 2] * cx / s;
  ax_grad -= coeffs_grad[1 * 3 + 2] * cy / s;
  cy_grad -= coeffs_grad[1 * 3 + 2] * ax / s;

  // coeffs[2 * 3 + 0] = (ay - by) / s;           // px
  // coeffs[2 * 3 + 1] = (bx - ax) / s;           // py
  // coeffs[2 * 3 + 2] = (ax * by - bx * ay) / s; // constant
  s_grad -= coeffs_grad[2 * 3 + 0] * (ay - by) / s / s;
  s_grad -= coeffs_grad[2 * 3 + 1] * (bx - ax) / s / s;
  s_grad -= coeffs_grad[2 * 3 + 2] * (ax * by - bx * ay) / s / s;
  ay_grad += coeffs_grad[2 * 3 + 0] / s;
  by_grad -= coeffs_grad[2 * 3 + 0] / s;
  bx_grad += coeffs_grad[2 * 3 + 1] / s;
  ax_grad -= coeffs_grad[2 * 3 + 1] / s;
  ax_grad += coeffs_grad[2 * 3 + 2] * by / s;
  by_grad += coeffs_grad[2 * 3 + 2] * ax / s;
  bx_grad -= coeffs_grad[2 * 3 + 2] * ay / s;
  ay_grad -= coeffs_grad[2 * 3 + 2] * bx / s;

  // scalar_t s = ax * by - ay * bx - ax * cy + ay * cx + bx * cy - by * cx;
  ax_grad += (by - cy) * s_grad;
  ay_grad += (cx - bx) * s_grad;
  bx_grad += (cy - ay) * s_grad;
  by_grad += (ax - cx) * s_grad;
  cx_grad += (ay - by) * s_grad;
  cy_grad += (bx - ax) * s_grad;

  // update gradients
  a_grad[0] += ax_grad;
  a_grad[1] += ay_grad;
  b_grad[0] += bx_grad;
  b_grad[1] += by_grad;
  c_grad[0] += cx_grad;
  c_grad[1] += cy_grad;
}

// get_barycentric_coord
template <typename scalar_t>
XINLINE void get_barycentric_coord(const scalar_t *p, const scalar_t *coeffs,
                                   scalar_t *bc) {
  scalar_t px = p[0], py = p[1];
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    bc[i] = coeffs[i * 3 + 0] * px + coeffs[i * 3 + 1] * py + coeffs[i * 3 + 2];
  }
}

// get_barycentric_coord_backward
template <typename scalar_t>
XINLINE void
get_barycentric_coord_backward(const scalar_t *p, const scalar_t *coeffs,
                               const scalar_t *bc_grad, scalar_t *p_grad,
                               scalar_t *coeffs_grad) {
  scalar_t px = p[0], py = p[1];
// for (int i = 0; i < 3; i++) {
//   bc[i] = coeffs[i * 3 + 0] * px + coeffs[i * 3 + 1] * py + coeffs[i * 3 +
//   2];
// }
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    p_grad[0] += bc_grad[i] * coeffs[i * 3 + 0];
    p_grad[1] += bc_grad[i] * coeffs[i * 3 + 1];
    coeffs_grad[i * 3 + 0] += bc_grad[i] * px;
    coeffs_grad[i * 3 + 1] += bc_grad[i] * py;
    coeffs_grad[i * 3 + 2] += bc_grad[i];
  }
}

// barycentric_clip
// bc_raw -> bc_clipped
template <typename scalar_t>
XINLINE void barycentric_clip(const scalar_t *bc_raw, scalar_t *bc_clipped) {
#pragma unroll 3
  for (int k = 0; k < 3; k++)
    bc_clipped[k] =
        max(min(bc_raw[k], static_cast<scalar_t>(1)), static_cast<scalar_t>(0));
}

// bc_clipped_grad -> bc_raw_grad
template <typename scalar_t>
XINLINE void barycentric_clip_backward(const scalar_t *bc_raw,
                                       const scalar_t *bc_clipped_grad,
                                       scalar_t *bc_raw_grad) {
#pragma unroll 3
  for (int k = 0; k < 3; k++) {
    if (bc_raw[k] > 1 || bc_raw[k] < 0) {
      continue;
    }
    bc_raw_grad[k] += bc_clipped_grad[k];
  }
}

// barycentric_normalize
// bc_clipped -> bc
template <typename scalar_t>
XINLINE void barycentric_normalize(const scalar_t *bc_clipped, scalar_t *bc) {
  const scalar_t s = max(bc_clipped[0] + bc_clipped[1] + bc_clipped[2],
                         static_cast<scalar_t>(1e-5));
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    bc[i] = bc_clipped[i] / s;
  }
}

// bc_grad -> bc_clipped_grad
template <typename scalar_t>
XINLINE void barycentric_normalize_backward(const scalar_t *bc_clipped,
                                            const scalar_t *bc_grad,
                                            scalar_t *bc_clipped_grad) {
  const scalar_t s_raw = bc_clipped[0] + bc_clipped[1] + bc_clipped[2];
  const scalar_t eps = static_cast<scalar_t>(1e-5);
  const bool s_raw_too_small = s_raw < eps;
  const scalar_t s = max(s_raw, eps);
  scalar_t s_grad = 0;
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    bc_clipped_grad[i] += bc_grad[i] / s;
    if (!s_raw_too_small) {
      // bc[i] = bc_clipped[i] / s;
      s_grad -= bc_grad[i] * bc_clipped[i] / s / s;
    }
  }
// s_grad -> bc_clipped_grad
#pragma unroll 3
  for (int i = 0; i < 3; i++) {
    bc_clipped_grad[i] += s_grad;
  }
}

// point_line_nearest_2d
template <typename scalar_t>
XINLINE void point_line_nearest_2d(const scalar_t *p, const scalar_t *a,
                                   const scalar_t *b, scalar_t *nearest_p) {
  const scalar_t ap[2] = {p[0] - a[0], p[1] - a[1]};
  const scalar_t ab[2] = {b[0] - a[0], b[1] - a[1]};

  const scalar_t ab_len_sq = square(ab[0]) + square(ab[1]);
  scalar_t lambda = (ap[0] * ab[0] + ap[1] * ab[1]) /
                    (ab_len_sq + static_cast<scalar_t>(1e-10));
  lambda = max(min(lambda, static_cast<scalar_t>(1)), static_cast<scalar_t>(0));

  nearest_p[0] = a[0] + lambda * ab[0];
  nearest_p[1] = a[1] + lambda * ab[1];
}

// point_line_nearest_2d_backward
template <typename scalar_t>
XINLINE void point_line_nearest_2d_backward(
    const scalar_t *p, const scalar_t *a, const scalar_t *b,
    const scalar_t *nearest_p_grad, scalar_t *a_grad, scalar_t *b_grad) {
  const scalar_t ap[2] = {p[0] - a[0], p[1] - a[1]};
  const scalar_t ab[2] = {b[0] - a[0], b[1] - a[1]};

  const scalar_t ab_len_sq = square(ab[0]) + square(ab[1]);
  scalar_t lambda = (ap[0] * ab[0] + ap[1] * ab[1]) /
                    (ab_len_sq + static_cast<scalar_t>(1e-10));
  lambda = max(min(lambda, static_cast<scalar_t>(1)), static_cast<scalar_t>(0));

  // nearest_p_grad -> a_grad, ab_grad, lambda_grad
  // nearest_p[0] = a[0] + lambda * ab[0];
  // nearest_p[1] = a[1] + lambda * ab[1];
  scalar_t ab_grad[2] = {0.0f, 0.0f};
  scalar_t lambda_grad = 0.0f;
  a_grad[0] += nearest_p_grad[0];
  a_grad[1] += nearest_p_grad[1];
  ab_grad[0] += lambda * nearest_p_grad[0];
  ab_grad[1] += lambda * nearest_p_grad[1];
  lambda_grad += nearest_p_grad[0] * ab[0];
  lambda_grad += nearest_p_grad[1] * ab[1];

  // lambda_grad -> ap_grad, ab_grad, ab_len_sq_grad
  scalar_t ap_grad[2] = {0.0f, 0.0f};
  scalar_t ab_len_sq_grad = 0.0f;
  if (lambda > 0 && lambda < 1) {
    scalar_t inv_ab_len_sq = static_cast<scalar_t>(1.0f / (ab_len_sq + 1e-10));
    ap_grad[0] += lambda_grad * ab[0];
    ap_grad[1] += lambda_grad * ab[1] * inv_ab_len_sq;
    ab_grad[0] += lambda_grad * ap[0] * inv_ab_len_sq;
    ab_grad[1] += lambda_grad * ap[1] * inv_ab_len_sq;
    ab_len_sq_grad -= (ap[0] * ab[0] + ap[1] * ab[1]) * lambda_grad *
                      inv_ab_len_sq * inv_ab_len_sq;
  }

  // ab_len_sq_grad -> ab_grad
  ab_grad[0] += ab_len_sq_grad * 2 * ab[0];
  ab_grad[1] += ab_len_sq_grad * 2 * ab[1];

  // ab_grad -> a_grad, b_grad
  // ab[2] = {b[0] - a[0], b[1] - a[1]};
  a_grad[0] -= ab_grad[0];
  a_grad[1] -= ab_grad[1];
  b_grad[0] += ab_grad[0];
  b_grad[1] += ab_grad[1];

  // ap_grad -> a_grad
  // ap[2] = {p[0] - a[0], p[1] - a[1]};
  a_grad[0] -= ap_grad[0];
  a_grad[1] -= ap_grad[1];
}

// for_each_pixel_within_extended_triangle
template <typename scalar_t, typename FunT>
XINLINE void for_each_pixel_within_extended_triangle(
    const scalar_t *p1, const scalar_t *p2, const scalar_t *p3, int out_h,
    int out_w, scalar_t radius, int pixel_group_id, int npixel_groups,
    FunT callback) {

  const scalar_t min_x = min_n(p1[0], p2[0], p3[0]) - radius;
  const scalar_t min_y = min_n(p1[1], p2[1], p3[1]) - radius;
  const scalar_t max_x = max_n(p1[0], p2[0], p3[0]) + radius;
  const scalar_t max_y = max_n(p1[1], p2[1], p3[1]) + radius;

  if (min_x >= out_w || max_x < 0 || min_y >= out_h || max_y < 0) {
    return;
  }

  scalar_t coeffs[3 * 3];
  get_barycentric_coefficients(p1, p2, p3, coeffs);

  const int from_x = clamp(static_cast<int>(floor(min_x)), 0, out_w - 1);
  const int to_x = clamp(static_cast<int>(ceil(max_x)), 0, out_w - 1) + 1;
  const int from_y = clamp(static_cast<int>(floor(min_y)), 0, out_h - 1);
  const int to_y = clamp(static_cast<int>(ceil(max_y)), 0, out_h - 1) + 1;

  const int max_h = to_y - from_y;
  const int npixels = (to_x - from_x) * max_h;

  for (int pixel = pixel_group_id; pixel < npixels; pixel += npixel_groups) {
    const int y = pixel % max_h + from_y;
    const int x = pixel / max_h + from_x;

    const scalar_t p[2] = {scalar_t(x), scalar_t(y)};

    const scalar_t *tri[3] = {p1, p2, p3};
    bool on_left[3] = {false, false, false};
    scalar_t nearest_ps[3][2];
    scalar_t dist_sqs[3];

#pragma unroll 3
    for (int i = 0; i < 3; i++) {
      on_left[i] = is_on_left_side(p, tri[i], tri[(i + 1) % 3]);
      point_line_nearest_2d(p, tri[i], tri[(i + 1) % 3], nearest_ps[i]);
      dist_sqs[i] =
          square(p[0] - nearest_ps[i][0]) + square(p[1] - nearest_ps[i][1]);
    }

    const bool inside = on_left[0] == on_left[1] && on_left[1] == on_left[2];
    callback(x, y, dist_sqs, coeffs, p, nearest_ps, inside);
  }
}

template <typename scalar_t>
XINLINE scalar_t compute_t2i_weight(int kernel_kind, bool inside, scalar_t dist,
                                    scalar_t kernel_radius, scalar_t beta) {
  const scalar_t signed_dist = inside ? dist : -dist;
  switch (kernel_kind) {
  case 0: // sin
    if (signed_dist > kernel_radius) {
      return 1.0f;
    } else if (signed_dist < -kernel_radius) {
      return 0.0f;
    } else {
      return sin(signed_dist * M_PI / 2 / kernel_radius) * 0.5f + 0.5f;
    }
  case 1: // linear
    if (signed_dist > kernel_radius) {
      return 1.0f;
    } else if (signed_dist < -kernel_radius) {
      return 0.0f;
    } else {
      return signed_dist / kernel_radius * 0.5f + 0.5f;
    }
  case 2: // cos_out
    if (signed_dist > 0) {
      return 1.0f;
    } else if (signed_dist < -kernel_radius) {
      return 0.0f;
    } else {
      return cos(-signed_dist * M_PI / kernel_radius) * 0.5f + 0.5f;
    }
  case 3: // sigmoid: 1/(1+exp(-x/kernel_radius*beta))
    if (signed_dist > kernel_radius) {
      return 1.0f;
    } else if (signed_dist < -kernel_radius) {
      return 0.0f;
    } else {
      return 1.0f / (1.0f + exp(-signed_dist / kernel_radius * beta));
    }
  case 4: // log_sigmoid: log(1/(1+exp(-x/kernel_radius*beta))) =
          // -log(1+exp(-x/kernel_radius*beta))
    if (signed_dist > kernel_radius) {
      return 0.0f;
    } else {
      return -log(1.0f + exp(-signed_dist / kernel_radius * beta));
    }
  default:
    return 0.0f;
  }
}

// returns dist_grad
template <typename scalar_t>
XINLINE scalar_t compute_t2i_weight_backward(int kernel_kind, bool inside,
                                             scalar_t dist,
                                             scalar_t kernel_radius,
                                             scalar_t beta,
                                             scalar_t weight_grad) {
  const scalar_t signed_dist = inside ? dist : -dist;
  scalar_t signed_dist_grad = 0.0f;
  switch (kernel_kind) {
  case 0: // sin
    if (-kernel_radius <= signed_dist && signed_dist <= kernel_radius) {
      // weight = sin(signed_dist * M_PI / 2 / kernel_radius) * 0.5 + 0.5;
      scalar_t f = M_PI / 2 / kernel_radius;
      signed_dist_grad = weight_grad * 0.5f * cos(signed_dist * f) * f;
    }
    break;
  case 1: // linear
    if (-kernel_radius <= signed_dist && signed_dist <= kernel_radius) {
      signed_dist_grad = weight_grad / kernel_radius * 0.5f;
    }
    break;
  case 2: // cos_out
    if (-kernel_radius <= signed_dist && signed_dist <= 0) {
      // weight = cos(-signed_dist * M_PI / kernel_radius) * 0.5f + 0.5f;
      scalar_t f = M_PI / kernel_radius;
      // signed_dist_grad = weight_grad * 0.5f * -sin(-signed_dist * f) * -f;
      signed_dist_grad = weight_grad * 0.5f * sin(signed_dist * f) * -f;
    }
    break;
  case 3: // sigmoid
    if (-kernel_radius <= signed_dist && signed_dist <= kernel_radius) {
      // weight = 1/(1+exp(-x/kernel_radius*beta))
      // or
      // weight = sigmoid(x/kernel_radius*beta)
      scalar_t f = beta / kernel_radius;
      scalar_t g = exp(-signed_dist * f);
      signed_dist_grad = weight_grad * g * f / square(g + 1.0f);
    }
  case 4: // log_sigmoid: log(1/(1+exp(-x/kernel_radius*beta))) =
          // -log(1+exp(-x/kernel_radius*beta))
    if (signed_dist > kernel_radius) {
      return 0.0f;
    } else {
      scalar_t f = beta / kernel_radius;
      // -log(1+exp(-x*f))
      return f / (1.0f + exp(f * signed_dist));
    }
  default:
    break;
  }
  const scalar_t dist_grad = inside ? signed_dist_grad : -signed_dist_grad;
  return dist_grad;
}

} // namespace haya_ext
