#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct LeakyReLUUpdateOutput
{
  T negval_;

  __host__ __device__
  LeakyReLUUpdateOutput() = default;

  __host__ __device__
  LeakyReLUUpdateOutput(T negval)
    : negval_(negval)
  {}

  __host__ __device__
  LeakyReLUUpdateOutput(const LeakyReLUUpdateOutput& o) = default;

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > 0) ? x : x * negval_;
  }

  __host__ __device__
  ~LeakyReLUUpdateOutput() {}
};

// in-place variant
template <typename T>
struct LeakyReLUUpdateOutputIP
{
  T negval_;

  __host__ __device__
  LeakyReLUUpdateOutputIP() = default;

  __host__ __device__
  LeakyReLUUpdateOutputIP(T negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(T *x)
  {
    *x = (*x > 0) ? *x : negval_ * (*x);
  }
 
  __host__ __device__
  ~LeakyReLUUpdateOutputIP() {}
};

template <typename T>
struct LeakyReLUUpdateGradInput
{
  T negval_;

  __host__ __device__
  LeakyReLUUpdateGradInput() = default;

  __host__ __device__
  LeakyReLUUpdateGradInput(T negval)
    : negval_(negval)
  {}

  __host__ __device__
  LeakyReLUUpdateGradInput(const LeakyReLUUpdateGradInput& f) = default;

  __device__ __forceinline__ void operator()(
    T* gradInput,
    T* input,
    T* gradOutput) const
  {
    *gradInput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }

  __host__ __device__
  ~LeakyReLUUpdateGradInput() {}

};

template <typename T>
struct LeakyReLUUpdateGradInputIP
{
  T negval_;

  __host__ __device__
  LeakyReLUUpdateGradInputIP() = default;

  __host__ __device__
  LeakyReLUUpdateGradInputIP(T negval)
    : negval_(negval)
  {}

  __host__ __device__
  LeakyReLUUpdateGradInputIP(const LeakyReLUUpdateGradInputIP& t) = default;

  __device__ __forceinline__ void operator()(
    T* gradOutput,
    T* input) const
  {
    *gradOutput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }

  __host__ __device__
  ~LeakyReLUUpdateGradInputIP() {}
};

#include "generic/LeakyReLU.cu"
#include "THCGenerateFloatTypes.h"
