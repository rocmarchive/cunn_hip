#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct SoftShrinkUpdateOutput
{
  T lambda_;

  __host__ __device__
  SoftShrinkUpdateOutput() = default;

  __host__ __device__
  SoftShrinkUpdateOutput(T lambda)
    : lambda_(lambda)
  {}

  __host__ __device__
  SoftShrinkUpdateOutput(const SoftShrinkUpdateOutput& f) = default;

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    if (x > lambda_) *out = x - lambda_;
    else if (x < -lambda_) *out = x + lambda_;
    else *out = ScalarConvert<int, T>::to(0);
  }

  __host__ __device__
  ~SoftShrinkUpdateOutput() {}

};

template <typename T>
struct SoftShrinkUpdateGradInput
{
  T lambda_;

  __host__ __device__
  SoftShrinkUpdateGradInput() = default;

  __host__ __device__
  SoftShrinkUpdateGradInput(T lambda)
    : lambda_(lambda)
  {}

  __host__ __device__
  SoftShrinkUpdateGradInput(const SoftShrinkUpdateGradInput& f) = default;

  __device__ __forceinline__ void operator()(T *gradInput, T *input, T *gradOutput) const
  {
    T x = *input;
    if (x > lambda_ || x < -lambda_)
      *gradInput = *gradOutput;
    else
      *gradInput = ScalarConvert<int, T>::to(0);
  }

  __host__ __device__
  ~SoftShrinkUpdateGradInput() {}
};

#include "generic/SoftShrink.cu"
#include "THCGenerateFloatTypes.h"
