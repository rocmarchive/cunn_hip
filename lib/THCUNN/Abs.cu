#include "THCUNN.h"
#include "common.h"

struct absupdateOutput_functor
{
  __host__ __device__
  absupdateOutput_functor() = default;

  __device__ 
  void operator()(float* output, const float* input) const
  {
#ifdef __HIP_PLATFORM_HCC__
    *output = fabsf(*input);
#else
    *output = abs(*input);
#endif
  }

   __host__ __device__
   ~absupdateOutput_functor() {}
};

void THNN_CudaAbs_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 2, input, output);
  THCudaTensor_resizeAs(state, output, input);
  THC_pointwiseApply2(state, output, input, absupdateOutput_functor());
}

struct absupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    *gradInput = *input < 0 ? - *gradOutput : *gradOutput;
  }
  __device__ __host__
  ~absupdateGradInput_functor() {}
};

void THNN_CudaAbs_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput)
{
  THCUNN_assertSameGPU(state, 3, input, gradOutput, gradInput);
  THCudaTensor_resizeAs(state, gradInput, input);
  THC_pointwiseApply3(state, gradInput, input, gradOutput, absupdateGradInput_functor());
}
