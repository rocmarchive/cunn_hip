#include "THCUNN.h"
#include "common.h"

struct ELUupdateOutput_functor
{
  float alpha_;

  __host__ __device__
  ELUupdateOutput_functor() = default;

  __host__ __device__
  ELUupdateOutput_functor(float alpha)
    : alpha_(alpha)
  {}

  ELUupdateOutput_functor(const ELUupdateOutput_functor& fun) = default;

  __device__ void operator()(float *output, const float *input) const
  {
    *output = *input <= 0 ? (exp(*input) - 1) * alpha_ : *input;
  }
  __host__ __device__
  ~ELUupdateOutput_functor() {}
};

// in-place variant
struct ELUupdateOutputIP_functor
{
  float alpha_;

  __host__ __device__
  ELUupdateOutputIP_functor() = default;

  __host__ __device__
  ELUupdateOutputIP_functor(float alpha)
    : alpha_(alpha)
  {}

  ELUupdateOutputIP_functor(const ELUupdateOutputIP_functor& fun) = default;

  __device__ void operator()(float *x) const
  {
    *x = *x <= 0 ? (exp(*x) - 1) * alpha_ : *x;
  }
  __host__ __device__
  ~ELUupdateOutputIP_functor() {}
};

void THNN_CudaELU_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output,
  float alpha, bool inplace)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    stub_THC_pointwiseApply1(state, input, ELUupdateOutputIP_functor(alpha));
    THCudaTensor_set(state, output, input);
  }
  else
  {
    THCudaTensor_resizeAs(state, output, input);
    stub_THC_pointwiseApply2(state, output, input, ELUupdateOutput_functor(alpha));
  }
}

struct ELUupdateGradInput_functor
{
  float alpha_;

  __host__ __device__
  ELUupdateGradInput_functor() = default;

  __host__ __device__
  ELUupdateGradInput_functor(float alpha)
    : alpha_(alpha)
  {}

  ELUupdateGradInput_functor(const ELUupdateGradInput_functor& fun) = default;

  __device__ void operator()(float *gradInput, const float *output, const float *gradOutput) const
  {
    *gradInput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
  __host__ __device__
  ~ELUupdateGradInput_functor() {}
};

struct ELUupdateGradInputIP_functor
{
  float alpha_;

  __host__ __device__
  ELUupdateGradInputIP_functor() = default;

  __host__ __device__
  ELUupdateGradInputIP_functor(float alpha)
    : alpha_(alpha)
  {}

  ELUupdateGradInputIP_functor(const ELUupdateGradInputIP_functor& fun) = default;

  __device__ void operator()(float *gradOutput, const float *output) const
  {
    *gradOutput = (*output) <= 0 ? (*gradOutput * (*output + alpha_)) : (*gradOutput);
  }
  __host__ __device__
  ~ELUupdateGradInputIP_functor(){}
};

void THNN_CudaELU_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, THCudaTensor *output, float alpha, bool inplace)
{
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  if (inplace)
  {
    stub_THC_pointwiseApply2(state, gradOutput, output, ELUupdateGradInputIP_functor(alpha));
    THCudaTensor_set(state, gradInput, gradOutput);
  }
  else
  {
    THCudaTensor_resizeAs(state, gradInput, output);
    stub_THC_pointwiseApply3(state, gradInput, output, gradOutput, ELUupdateGradInput_functor(alpha));
  }
}
