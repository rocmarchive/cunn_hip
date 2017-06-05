#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

template <typename T>
struct softPlusupdateOutput_functor
{
  T threshold;
  T beta;

  __host__ __device__
  softPlusupdateOutput_functor() = default;

  __host__ __device__
  softPlusupdateOutput_functor(T threshold_, T beta_)
    : threshold(threshold_)
    , beta(beta_)
  {}

  __host__ __device__
  softPlusupdateOutput_functor(const softPlusupdateOutput_functor& f) = default;

  __device__ void operator()(T *output, const T *input) const {
    T betain = beta * (*input);
    *output = ((betain) > threshold) ? *input : (1/beta) * log1p(exp(betain));
  }

  __host__ __device__
  ~softPlusupdateOutput_functor() {}
};

template <typename T>
struct softPlusupdateGradInput_functor
{
  T threshold;
  T beta;

  __host__ __device__
  softPlusupdateGradInput_functor() = default;

  __host__ __device__
  softPlusupdateGradInput_functor(T threshold_, T beta_)
    : threshold(threshold_)
    , beta(beta_)
  {}

  __host__ __device__
  softPlusupdateGradInput_functor(const softPlusupdateGradInput_functor& f) = default;

  __device__ void operator()(T *gradInput, const T *output, const T *gradOutput) const
  {
    T betaout = beta * (*output);
    T exp_bo = exp(betaout);
    *gradInput = ((betaout) > threshold) ? *gradOutput : *gradOutput * (exp_bo - 1) / exp_bo;
  }

  __host__ __device__
  ~softPlusupdateGradInput_functor() {}
};

#include "generic/SoftPlus.cu"
#include "THCGenerateFloatTypes.h"
