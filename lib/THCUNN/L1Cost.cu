#include "THCUNN.h"
#include "common.h"

#if THRUST_PATH
    #include <thrust/device_ptr.h>
    #include <thrust/reduce.h>
    #include <thrust/transform.h>
#else
    #include <bolt/amp/functional.h>
    #include <bolt/amp/reduce.h>
    #include <bolt/amp/transform.h>
#endif

struct l1cost_functor
{
  __host__ __device__ 
  l1cost_functor() = default;

  __device__
  float operator()(float x, float y) const
  {
#ifdef __HIP_PLATFORM_HCC__
    return fabsf(x) + fabsf(y);
#else
    return std::abs(x) + std::abs(y);
#endif
  }

  l1cost_functor(const l1cost_functor& fun) = default;

  __host__ __device__ 
  ~l1cost_functor() {}
};

void THNN_CudaL1Cost_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THCUNN_assertSameGPU(state, 1, input);
  float sum;
  long size = THCudaTensor_nElement(state, input);
  input = THCudaTensor_newContiguous(state, input);
#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  sum = thrust::reduce(input_data, input_data+size, (float) 0, l1cost_functor());
#else
  auto input_data = THCudaTensor_data(state, input);
  auto input_data_end = input_data + size;
   sum = bolt::amp::reduce(input_data, 
                           //input_data+size, 
                           input_data_end, 
                           0.0f,
                           l1cost_functor());
#endif

  THCudaTensor_free(state, input);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct l1cost_updateGradInput_functor
{
  __host__ __device__ float operator()(float x) const
  {
    if (x > 0)
      return 1;
    else if (x < 0)
      return -1;
    else
      return 0;
  }
};

void THNN_CudaL1Cost_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput)
{
  THCUNN_assertSameGPU(state, 2, input, gradInput);
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, gradInput, input);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(input_data, input_data+size, gradInput_data, l1cost_updateGradInput_functor());
#else
  auto input_data = THCudaTensor_data(state, input);
  auto gradInput_data = THCudaTensor_data(state, gradInput);

  bolt::amp::transform(input_data, 
                       input_data+size, 
                       gradInput_data, 
                       l1cost_updateGradInput_functor());
#endif

  THCudaTensor_free(state, input);
}
