#include "hip/hip_runtime.h"
#include "THCUNN.h"
#include "common.h"
#ifdef CURAND_PATH
#include <curand.h>
#include <curand_kernel.h>
#else
#include <hip/hip_hcc.h>
#include "MTGP/hiprand_mtgp32.h"
#endif

// copied from cutorch/lib/THC/THCTensorRandom.cu
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256
#define NUM_BLOCKS(n) min((int)THCCeilDiv(n, (long) BLOCK_SIZE), MAX_NUM_BLOCKS)

#ifdef CURAND_PATH
__global__ void rreluUpdateOutputTrain( int n, curandStateMtgp32 *state,
  float *input, float* noise, float *output, double a, double b)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    if (input[i] <= 0)
    {
      float r = curand_uniform(&state[hipBlockIdx_x]);
      r = r * (b-a) + a;
      output[i] = input[i] * r;
      noise[i] = r;
    }
    else
    {
      output[i] = input[i];
      noise[i] = 1;
    }
  }
}
#else

struct user_uniform_functor {
  double _a;
  double _b;
  user_uniform_functor(double a, double b) __attribute__((hc, cpu)) : _a(a), _b(b) {}
  inline double operator()(const float& x) const __attribute__((hc, cpu)) {
    return x * (_b - _a) + _a;
  }
  // User should provide copy ctor
  user_uniform_functor(const user_uniform_functor&other) __attribute__((hc, cpu)) : _a(other._a), _b(other._b) { }
  // User should provide copy assign ctor
  user_uniform_functor& operator = (const user_uniform_functor&other) __attribute__((hc, cpu)) {
    _a = other._a;
    _b = other._b;
    return *this;
  }
};

__global__ void rreluUpdateOutputTrain( int n, HipRandStateMtgp32 *state,
  float *input, float* noise, float *output, double a, double b)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    if (input[i] <= 0) {
      output[i] = input[i] * noise[i];
    }
    else
    {
      output[i] = input[i];
      noise[i] = 1;
    }
  }
}
#endif

struct RReLUUpdateOutputEval_functor
{
  const float negSlope_;

 __host__ __device__  RReLUUpdateOutputEval_functor(float negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(float *out, float *in)
  {
    const float x = *in;
    const float r = x <= 0 ? negSlope_ : 1;
    *out = x * r;
  }
};

struct RReLUUpdateOutputEvalIP_functor
{
  const float negSlope_;

  __host__ __device__ RReLUUpdateOutputEvalIP_functor(float negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(float *x)
  {
    if (*x <= 0)
    {
      *x = *x * negSlope_;
    }
  }
};

void THNN_CudaRReLU_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output,
  THCudaTensor *noise, double lower, double upper, bool train, bool inplace, void *generator)
{
  THCUNN_assertSameGPU(state, 3, input, output, noise);
#ifdef CURAND_PATH
  struct curandStateMtgp32* gen_states = THCRandom_generatorStates(state);
#else
  struct HipRandStateMtgp32* gen_states = THCRandom_generatorStates(state);
#endif
  if (train)
  {
    input = THCudaTensor_newContiguous(state, input);
    THCudaTensor_resizeAs(state, noise, input);
    float *input_data = THCudaTensor_data(state, input);
    float *noise_data = THCudaTensor_data(state, noise);
    long n = THCudaTensor_nElement(state, input);
    if (inplace)
    {
#ifdef CURAND_PATH
      hipLaunchKernelGGL((rreluUpdateOutputTrain), dim3(NUM_BLOCKS(n)), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state),
        n, gen_states, input_data, noise_data, input_data, lower, upper);
      THCudaTensor_set(state, output, input);
#else
      hipStream_t currentStream = THCState_getCurrentStream(state);
      hc::accelerator_view* current_accl_view;
      hipHccGetAcceleratorView(currentStream, &current_accl_view);
      user_uniform_kernel(*current_accl_view, gen_states, noise_data, user_uniform_functor(lower, upper));
      hipLaunchKernelGGL((rreluUpdateOutputTrain), dim3(NUM_BLOCKS(n)), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state),
        n, gen_states, input_data, noise_data, input_data, lower, upper);
      THCudaTensor_set(state, output, input);
#endif
    }
    else
    {
      THCudaTensor_resizeAs(state, output, input);
      float *output_data = THCudaTensor_data(state, output);
#ifdef CURAND_PATH
      hipLaunchKernelGGL((rreluUpdateOutputTrain), dim3(NUM_BLOCKS(n)), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state),
        n, gen_states, input_data, noise_data, output_data, lower, upper);
#else
      hipStream_t currentStream = THCState_getCurrentStream(state);
      hc::accelerator_view* current_accl_view;
      hipHccGetAcceleratorView(currentStream, &current_accl_view);
      user_uniform_kernel(*current_accl_view, gen_states, noise_data, user_uniform_functor(lower, upper));
      hipLaunchKernelGGL((rreluUpdateOutputTrain), dim3(NUM_BLOCKS(n)), dim3(BLOCK_SIZE), 0, THCState_getCurrentStream(state),
        n, gen_states, input_data, noise_data, output_data, lower, upper);
#endif
    }
    THCudaCheck(hipGetLastError());
    THCudaTensor_free(state, input);
  }
  else
  {
    const double negSlope = (lower + upper) / 2;
    if (inplace)
    {
      THC_pointwiseApply1(state, input, RReLUUpdateOutputEvalIP_functor(negSlope));
      THCudaTensor_set(state, output, input);
    }
    else
    {
      THCudaTensor_resizeAs(state, output, input);
      THC_pointwiseApply2(state, output, input, RReLUUpdateOutputEval_functor(negSlope));
    }
  }
}

struct RReLUupdateGradInputEval_functor
{
  const float negSlope_;

  __host__ __device__ RReLUupdateGradInputEval_functor(float negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(float *gradIn, float *gradOut, float *in)
  {
    *gradIn = (*in) <= 0 ? (*gradOut) * negSlope_ : (*gradOut);
  }
};

struct RReLUupdateGradInputEvalIP_functor
{
  const float negSlope_;

  __host__ __device__ RReLUupdateGradInputEvalIP_functor(float negSlope)
    : negSlope_(negSlope)
  {}

  __device__ __forceinline__ void operator()(float *gradOut, float *in)
  {
    if (*in <= 0)
    {
      *gradOut = (*gradOut) * negSlope_;
    }
  }
};

void THNN_CudaRReLU_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput,
  THCudaTensor *gradInput, THCudaTensor *noise, double lower, double upper, bool train, bool inplace)
{
  THCUNN_assertSameGPU(state, 4, input, gradOutput, gradInput, noise);

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  if (train && upper - lower > 1E-6)    // e.g. if upper == lower, RReLU behaves like LeakyReLU
  {
    // multiply the gradient by the noise tensor
    if (inplace)
    {
      THCudaTensor_cmul(state, gradOutput, gradOutput, noise);
      THCudaTensor_set(state, gradInput, gradOutput);
    }
    else
    {
      THCudaTensor_resizeAs(state, gradInput, input);
      THCudaTensor_cmul(state, gradInput, gradOutput, noise);
    }
  }
  else
  {
    // use constant factor for negative input values
    const double negSlope = (lower + upper) / 2;
    if (inplace)
    {
      THC_pointwiseApply2(state, gradOutput, input, RReLUupdateGradInputEvalIP_functor(negSlope));
      THCudaTensor_set(state, gradInput, gradOutput);
    }
    else
    {
      THCudaTensor_resizeAs(state, gradInput, input);
      THC_pointwiseApply3(state, gradInput, gradOutput, input, RReLUupdateGradInputEval_functor(negSlope));
    }
  }

  THCudaTensor_free(state, gradOutput);
}
