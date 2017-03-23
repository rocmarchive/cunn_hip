#include "THCUNN.h"
#include "common.h"

struct LeakyReLUUpdateOutput
{
  const float negval_;

  __host__ __device__
  LeakyReLUUpdateOutput(float negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(float *out, float *in)
  {
    float x = *in;
    *out = (x > 0) ? x : x * negval_;
  }
  
  __host__ __device__
  ~LeakyReLUUpdateOutput() {}
};

// in-place variant
struct LeakyReLUUpdateOutputIP
{
  const float negval_;

  __host__ __device__
  LeakyReLUUpdateOutputIP(float negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(float *x)
  {
    *x = (*x > 0) ? *x : negval_ * (*x);
  }
  
  __host__ __device__
  ~LeakyReLUUpdateOutputIP() {}
};

void THNN_CudaLeakyReLU_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output,
  double negval, bool inplace)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  if (inplace)
  {
    THC_pointwiseApply1(state, input, LeakyReLUUpdateOutputIP(negval));
    THCudaTensor_set(state, output, input);
  }
  else
  {
    THCudaTensor_resizeAs(state, output, input);
    THC_pointwiseApply2(state, output, input, LeakyReLUUpdateOutput(negval));
  }

  THCudaCheck(hipGetLastError());
}

struct LeakyReLUUpdateGradInput
{
  const float negval_;

  __host__ __device__
  LeakyReLUUpdateGradInput(float negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(
    float* gradInput,
    float* input,
    float* gradOutput) const
  {
    *gradInput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }

  __host__ __device__
  ~LeakyReLUUpdateGradInput() {}
};

struct LeakyReLUUpdateGradInputIP
{
  const float negval_;

  __host__ __device__
  LeakyReLUUpdateGradInputIP(float negval)
    : negval_(negval)
  {}

  __device__ __forceinline__ void operator()(
    float* gradOutput,
    float* input) const
  {
    *gradOutput = (*input > 0) ? *gradOutput : (*gradOutput) * negval_;
  }

  __host__ __device__
  ~LeakyReLUUpdateGradInputIP() {}
};

void THNN_CudaLeakyReLU_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, double negval, bool inplace)
{
  THCUNN_assertSameGPU(state, 3, input, gradInput, gradOutput);

  if (inplace)
  {
    THC_pointwiseApply2(state, gradOutput, input, LeakyReLUUpdateGradInputIP(negval));
    THCudaTensor_set(state, gradInput, gradOutput);
  }
  else
  {
    THCudaTensor_resizeAs(state, gradInput, input);
    THC_pointwiseApply3(state, gradInput, input, gradOutput, LeakyReLUUpdateGradInput(negval));
  }

  THCudaCheck(hipGetLastError());
}
