#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct ELUupdateOutput_functor
{
  T alpha_;

  __host__ __device__ 
  ELUupdateOutput_functor() = default;

  __host__ __device__ 
  ELUupdateOutput_functor(T alpha)
    : alpha_(alpha)
  {}

  __host__ __device__ 
  ELUupdateOutput_functor(const ELUupdateOutput_functor& f) = default;

  __device__ void operator()(T *output, const T *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * alpha_ : *input;
  }

  __host__ __device__ 
  ~ELUupdateOutput_functor() {}
};

// in-place variant
template <typename T>
struct ELUupdateOutputIP_functor
{
  const T alpha_;

  __host__ __device__ 
  ELUupdateOutputIP_functor() = default;

  __host__ __device__ 
  ELUupdateOutputIP_functor(T alpha)
    : alpha_(alpha)
  {}

  __host__ __device__ 
  ELUupdateOutputIP_functor(const ELUupdateOutputIP_functor& f) = default;

  __device__ void operator()(T *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * alpha_ : *x;
  }

  __host__ __device__ 
  ~ELUupdateOutputIP_functor() {}
};

template <typename T>
struct ELUupdateGradInput_functor
{
  T alpha_;

  __host__ __device__ 
  ELUupdateGradInput_functor() = default;

  __host__ __device__ 
  ELUupdateGradInput_functor(T alpha)
    : alpha_(alpha)
  {}

  __host__ __device__ 
  ELUupdateGradInput_functor(const ELUupdateGradInput_functor& f) = default;

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }

  __host__ __device__ 
  ~ELUupdateGradInput_functor() {}
};

template <typename T>
struct ELUupdateGradInputIP_functor
{
  T alpha_;

  __host__ __device__ 
  ELUupdateGradInputIP_functor() = default;

  __host__ __device__ 
  ELUupdateGradInputIP_functor(T alpha)
    : alpha_(alpha)
  {}

  __host__ __device__ 
  ELUupdateGradInputIP_functor(const ELUupdateGradInputIP_functor& f) = default;

  __device__ void operator()(T *gradOutput, const T *output) const
  {
    *gradOutput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }

  __host__ __device__ 
  ~ELUupdateGradInputIP_functor() {}

};

#include "generic/ELU.cu"
#include "THCGenerateFloatTypes.h"
