#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct sqrtupdateOutput_functor
{
  T bias;

  __host__ __device__
  sqrtupdateOutput_functor() = default;

  __host__ __device__
  sqrtupdateOutput_functor(T bias_)
    : bias(bias_)
  {}

  __host__ __device__
  sqrtupdateOutput_functor(const sqrtupdateOutput_functor& f) = default;

  __device__ void operator()(T *output, const T *input) const
  {
// WSTHORNTON -- temporary comment kernel
    //*output = sqrt(*input + bias);
    *output = T(0.0);
  }

  __host__ __device__
  ~sqrtupdateOutput_functor() {}

};

template <typename T>
struct sqrtupdateGradInput_functor
{
  __host__ __device__
  sqrtupdateGradInput_functor() {}

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (THCNumerics<T>::eq(*output,ScalarConvert<float, T>::to(0.0f))) ? ScalarConvert<float, T>::to(0.0f) : ((ScalarConvert<float, T>::to(0.5f) * *gradOutput) / *output);
  }
};

#include "generic/Sqrt.cu"
#include "THCGenerateFloatTypes.h"
