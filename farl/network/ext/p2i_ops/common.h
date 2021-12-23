// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/ATen.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#define RESTRICT __restrict
#pragma warning(disable : 4068)
#else
#define FORCE_INLINE __attribute__((always_inline))
#define RESTRICT __restrict__
#endif

#ifdef __CUDACC__
#define XINLINE __device__ __host__
#define XGLOBAL __global__
#define XDEVICE __device__
#define XSHARED __shared__
#else
#define XINLINE
#define XGLOBAL
#define XDEVICE
#define XSHARED
#endif

namespace haya_ext {
#ifdef __CUDACC__
// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                                       \
  do {                                                                         \
    cudaError_t e = cudaGetLastError();                                        \
    if (e != cudaSuccess) {                                                    \
      char buffer[512] = {'\0'};                                               \
      sprintf(buffer, "Cuda failure %s:%d: '%s(%s)'", __FILE__, __LINE__,      \
              cudaGetErrorName(e), cudaGetErrorString(e));                     \
      AT_ERROR(buffer);                                                        \
    }                                                                          \
  } while (0)
#else
#define cudaCheckError()
#endif

struct cpu_device {};
struct gpu_device {};

template <typename XPU> struct kernel;

template <> struct kernel<cpu_device> {
  template <typename OP, typename... Args>
  inline static FORCE_INLINE void launch(OP op, const int N, Args... args) {
#ifdef _OPENMP
    const int omp_cores = omp_get_thread_num();
    if (omp_cores <= 1) {
      // Zero means not to use OMP, but don't interfere with external OMP
      // behavior
      for (int i = 0; i < N; ++i) {
        op(i, args...);
      }
    } else {
#pragma omp parallel for num_threads(omp_cores)
      for (int i = 0; i < N; ++i) {
        op(i, args...);
      }
    }
#else
    for (int i = 0; i < N; ++i) {
      op(i, args...);
    }
#endif
  }
};

#if defined(NO_CUDA) // try launching gpu kernel from a no cuda build
template <> struct kernel<gpu_device> {
  template <typename OP, typename... Args>
  inline static FORCE_INLINE void launch(OP op, const int N, Args... args) {
    AT_ERROR("failed to launch cuda kernel in a NO CUDA build");
  }
};
#elif defined(__CUDACC__) // launching gpu kernel within nvcc compilation
namespace detail {
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kMaxGridNum = 65535;
constexpr int kBaseThreadBits = 8;
constexpr int kBaseThreadNum = 1 << kBaseThreadBits;
constexpr int kBaseGridNum = 1024;

template <typename OP, typename... Args>
XGLOBAL void _generic_kernel(OP op, int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    op(i, args...);
  }
}
} // namespace detail

template <> struct kernel<gpu_device> {
  template <typename OP, typename... Args>
  inline static FORCE_INLINE void launch(OP op, const int N, Args... args) {
    static_assert(std::is_class<OP>::value,
                  "You should pass a functor (including lambda) to "
                  "kernel::launch. Passing a function pointer "
                  "will cause cuda error in runtime.");
    const dim3 blocks =
        (N + detail::kBaseThreadNum - 1) / detail::kBaseThreadNum;
    detail::_generic_kernel<OP, Args...>
        <<<blocks, detail::kBaseThreadNum>>>(op, N, args...);
  }
  template <typename OP, typename... Args>
  inline static FORCE_INLINE void launch_max_threads(OP op, const int N,
                                                     Args... args) {
    static_assert(std::is_class<OP>::value,
                  "You should pass a functor (including lambda) to "
                  "kernel::launch. Passing a function pointer "
                  "will cause cuda error in runtime.");
    const dim3 blocks =
        (N + detail::kMaxThreadsPerBlock - 1) / detail::kMaxThreadsPerBlock;
    detail::_generic_kernel<OP, Args...>
        <<<blocks, detail::kMaxThreadsPerBlock>>>(op, N, args...);
  }
};
#else // try launching gpu kernel without nvcc compilation, this should not
      // compile
namespace detail {
template <typename T> struct always_false {
  static constexpr bool value = false;
};
} // namespace detail
template <> struct kernel<gpu_device> {
  template <typename OP, typename... Args>
  inline static FORCE_INLINE void launch(OP op, const int N, Args... args) {
    static_assert(detail::always_false<OP>::value,
                  "trying to instantiate gpu kernel under non cuda context");
  }
};
#endif
} // namespace haya_ext
