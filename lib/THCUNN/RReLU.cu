// WSTHORNTON -- ifdef
#if 1
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
  #include "hiprng_kernel.h"
#endif

// copied from cutorch/lib/THC/THCTensorRandom.cu
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS(n) min((int)THCCeilDiv(n, (ptrdiff_t) BLOCK_SIZE), MAX_NUM_BLOCKS)

template<typename T>
inline T __device__ hiprng_uniform_type(hiprngStateMtgp32 *state);

#ifdef CUDA_HALF_TENSOR
template <>
inline half __device__ hiprng_uniform_type<half>(hiprngStateMtgp32 *state) {
  return ScalarConvert<float, half>::to(hiprng_uniform(state));
}
#endif

template <>
inline float __device__ hiprng_uniform_type<float>(hiprngStateMtgp32 *state) {
  return hiprng_uniform(state);
}
  
template <>
inline double __device__ hiprng_uniform_type<double>(hiprngStateMtgp32 *state) {
  //return hiprng_uniform_double(state);
  //TODO: double support
  return hiprng_uniform(state);
}

template <typename T>
__global__ void rreluUpdateOutputTrain(int n, hiprngStateMtgp32 *state,
    T *input, T* noise, T *output, double a, double b)
  {
    CUDA_KERNEL_LOOP(i, n)
    {
      if (input[i] <= 0)
      {
        T r = hiprng_uniform_type<T>(&state[hipBlockIdx_x]);
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

template <typename T>
struct RReLUUpdateOutputEval_functor
{
  T negSlope_;

  __host__ __device__ 
  RReLUUpdateOutputEval_functor() = default;

  __host__ __device__ 
  RReLUUpdateOutputEval_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __host__ __device__ 
  RReLUUpdateOutputEval_functor(const RReLUUpdateOutputEval_functor& f) = default;

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    const T x = *in;
    const T r = x <= 0 ? negSlope_ : ScalarConvert<int, T>::to(1);
    *out = x * r;
  }

  __host__ __device__ 
  ~RReLUUpdateOutputEval_functor() {}
};

template <typename T>
struct RReLUUpdateOutputEvalIP_functor
{
  T negSlope_;

  __host__ __device__
  RReLUUpdateOutputEvalIP_functor() = default;

  __host__ __device__
  RReLUUpdateOutputEvalIP_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __host__ __device__
  RReLUUpdateOutputEvalIP_functor(const RReLUUpdateOutputEvalIP_functor& f) = default;

  __device__ __forceinline__ void operator()(T *x)
  {
    if (*x <= 0)
    {
      *x = *x * negSlope_;
    }
  }
  __host__ __device__
  ~RReLUUpdateOutputEvalIP_functor() {}
};

template <typename T>
struct RReLUupdateGradInputEval_functor
{
  T negSlope_;

  __host__ __device__
  RReLUupdateGradInputEval_functor() = default;

  __host__ __device__
  RReLUupdateGradInputEval_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __host__ __device__
  RReLUupdateGradInputEval_functor(const RReLUupdateGradInputEval_functor& f) = default;

  __device__ __forceinline__ void operator()(T *gradIn, T *gradOut, T *in)
  {
    *gradIn = (*in) <= 0 ? (*gradOut) * negSlope_ : (*gradOut);
  }

  __host__ __device__
  ~RReLUupdateGradInputEval_functor() {}

};

template <typename T>
struct RReLUupdateGradInputEvalIP_functor
{
  T negSlope_;

  __host__ __device__
  RReLUupdateGradInputEvalIP_functor() = default;

  __host__ __device__
  RReLUupdateGradInputEvalIP_functor(T negSlope)
    : negSlope_(negSlope)
  {}

  __host__ __device__
  RReLUupdateGradInputEvalIP_functor(const RReLUupdateGradInputEvalIP_functor& f) = default;

  __device__ __forceinline__ void operator()(T *gradOut, T *in)
  {
    if (*in <= 0)
    {
      *gradOut = (*gradOut) * negSlope_;
    }
  }

  __host__ __device__
  ~RReLUupdateGradInputEvalIP_functor() {}
};

#include "generic/RReLU.cu"
#include "THCGenerateFloatTypes.h"
#endif
