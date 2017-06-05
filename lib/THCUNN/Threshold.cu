#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ThresholdUpdateOutput
{
  T threshold_;
  T val_;

  __host__ __device__
  ThresholdUpdateOutput() = default;

  __host__ __device__
  ThresholdUpdateOutput(T threshold, T val)
    : threshold_(threshold)
    , val_(val)
  {}

  __host__ __device__
  ThresholdUpdateOutput(const ThresholdUpdateOutput& t) = default;

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > threshold_) ? x : val_;
  }

  __host__ __device__
  ~ThresholdUpdateOutput() {}
};

// in-place variant
template <typename T>
struct ThresholdUpdateOutputIP
{
  T threshold_;
  T val_;

  __host__ __device__
  ThresholdUpdateOutputIP() = default;

  __host__ __device__  
  ThresholdUpdateOutputIP(T threshold, T val)
    : threshold_(threshold)
    , val_(val)
  {}

  __host__ __device__
  ThresholdUpdateOutputIP(const ThresholdUpdateOutputIP& t) = default;

  __device__ __forceinline__ void operator()(T *x)
  {
    *x = (*x > threshold_) ? *x : val_;
  }

  __host__ __device__  
  ~ThresholdUpdateOutputIP() {}
};

template <typename T>
struct ThresholdUpdateGradInput
{
  T threshold_;

  __host__ __device__  
  ThresholdUpdateGradInput() = default;

  __host__ __device__  
  ThresholdUpdateGradInput(T threshold)
    : threshold_(threshold)
  {}

  __host__ __device__  
  ThresholdUpdateGradInput(const ThresholdUpdateGradInput& f) = default;

  __device__ __forceinline__ void operator()(
    T *gradInput, T *input, T *gradOutput) const
  {
    *gradInput = (*input > threshold_) ? *gradOutput : ScalarConvert<int, T>::to(0);
  }

  __host__ __device__  
  ~ThresholdUpdateGradInput() {}
};

template <typename T>
struct ThresholdUpdateGradInputIP
{
  T threshold_;

  __host__ __device__  
  ThresholdUpdateGradInputIP() = default;

  __host__ __device__  
  ThresholdUpdateGradInputIP(T threshold)
    : threshold_(threshold)
  {}

  __host__ __device__  
  ThresholdUpdateGradInputIP(const ThresholdUpdateGradInputIP& t) = default;

  __device__ __forceinline__ void operator()(
    T *gradOutput, T *input) const
  {
    *gradOutput = (*input > threshold_) ? *gradOutput : ScalarConvert<int, T>::to(0);
  }

  __host__ __device__  
  ~ThresholdUpdateGradInputIP() {}
};

#include "generic/Threshold.cu"
#include "THCGenerateFloatTypes.h"
