// WSTHORNTON -- ifdef
#if 0
#include "hip/hip_runtime.h"
#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>
#include "common.h"
#ifdef CURAND_PATH
  #include <curand.h>
  #include <curand_kernel.h>
#else
  #include <hip/hip_hcc.h>
  #include "MTGP/hiprand_mtgp32.h"
#endif

// copied from cutorch/lib/THC/THCTensorRandom.cu
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS(n) min((int)THCCeilDiv(n, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)

#ifdef CURAND_PATH
template<typename T>
inline T __device__ curand_uniform_type(curandStateMtgp32 *state);

#ifdef CUDA_HALF_TENSOR
template <>
inline half __device__ curand_uniform_type<half>(curandStateMtgp32 *state) {
  return ScalarConvert<float, half>::to(curand_uniform(state));
}
#endif

template <>
inline float __device__ curand_uniform_type<float>(curandStateMtgp32 *state) {
  return curand_uniform(state);
}

template <>
inline double __device__ curand_uniform_type<double>(curandStateMtgp32 *state) {
  return curand_uniform_double(state);
}
#endif

#ifdef CURAND_PATH
template <typename T>
__global__ void rreluUpdateOutputTrain(int n, curandStateMtgp32 *state,
  T *input, T* noise, T *output, double a, double b)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    if (input[i] <= 0)
    {
      T r = curand_uniform_type<T>(&state[hipBlockIdx_x]);
      r = ScalarConvert<double, T>::to(r * (b-a) + a);
      output[i] = input[i] * r;
      noise[i] = r;
    }
    else
    {
      output[i] = input[i];
      noise[i] = ScalarConvert<int, T>::to(1);
    }
  }
}
#else
struct user_uniform_functor {
  double _a;
  double _b;
  user_uniform_functor(double a, double b) __attribute__((hc, cpu)) : _a(a), _b(b) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return x * (_b - _a) + _a;
  }
  // User should provide copy ctor
  user_uniform_functor(const user_uniform_functor&other) __attribute__((hc, cpu)) : _a(other._a), _b(other._b) { }
  // User should provide copy assign ctor
  user_uniform_functor& operator = (const user_uniform_functor&other) __attribute__((hc, cpu)) {
    _a = other._a;
    _b = other._b;
    return *this;
  }
};

__global__ void rreluUpdateOutputTrain( int n, HipRandStateMtgp32 *state,
  float *input, float* noise, float *output, double a, double b)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    if (input[i] <= 0) {
      output[i] = input[i] * noise[i];
    }
    else
    {
      output[i] = input[i];
      noise[i] = 1;
    }
  }
}
#endif

template <typename T>
struct RReLUUpdateOutputEval_functor
{
  const T negSlope_;

  RReLUUpdateOutputEval_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    const T x = *in;
    const T r = x <= 0 ? negSlope_ : ScalarConvert<int, T>::to(1);
    *out = x * r;
  }
};

template <typename T>
struct RReLUUpdateOutputEvalIP_functor
{
  const T negSlope_;

  RReLUUpdateOutputEvalIP_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *x)
  {
    if (*x <= 0)
    {
      *x = *x * negSlope_;
    }
  }
};

template <typename T>
struct RReLUupdateGradInputEval_functor
{
  const T negSlope_;

  RReLUupdateGradInputEval_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *gradIn, T *gradOut, T *in)
  {
    *gradIn = (*in) <= 0 ? (*gradOut) * negSlope_ : (*gradOut);
  }
};

template <typename T>
struct RReLUupdateGradInputEvalIP_functor
{
  const T negSlope_;

  RReLUupdateGradInputEvalIP_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(T *gradOut, T *in)
  {
    if (*in <= 0)
    {
      *gradOut = (*gradOut) * negSlope_;
    }
  }
};

#include "generic/RReLU.cu"
#include "THCGenerateFloatTypes.h"
#endif
