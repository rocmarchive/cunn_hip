#include "THCUNN.h"
#include "common.h"

#if THRUST_PATH
    #include <thrust/fill.h>
    #include <thrust/functional.h>
    #include <thrust/device_ptr.h>
    #include <thrust/reduce.h>
    #include <thrust/inner_product.h>
    #if CUDA_VERSION >= 7000
        #include <thrust/system/cuda/execution_policy.h>
    #endif
#else
    #include <bolt/amp/functional.h>
    #include <bolt/amp/inner_product.h>
#endif

struct mse_functor
{
  __host__ __device__
  mse_functor() {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    float z = x-y;
    return z*z;
  }

  __host__ __device__
  ~mse_functor() {}
};

void THNN_CudaMSECriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, bool sizeAverage)
{
  THCUNN_assertSameGPU(state, 2, input, target);
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  float sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), mse_functor());
#else
  auto input_data = THCudaTensor_data(state, input);
  auto target_data = THCudaTensor_data(state, target);
  float sum = bolt::amp::inner_product(input_data, 
                                       input_data+size, 
                                       target_data, 0.0f,
                                       bolt::amp::plus<float>(), 
                                       mse_functor());
#endif
  if (sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct mse_updateGradInput_functor
{
  float norm;

  __host__ __device__ 
  mse_updateGradInput_functor() = default;

  __host__ __device__ 
  mse_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  mse_updateGradInput_functor(const mse_updateGradInput_functor& fun) = default;

  __host__ __device__ 
  float operator()(const float &x, const float &y) const
  {
    return norm * (x - y);
  }
};

void THNN_CudaMSECriterion_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *gradInput, bool sizeAverage)
{
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);
  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );

  long size = THCudaTensor_nElement(state, input);
  float norm = sizeAverage ? 2.f/size : 2.f;

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    mse_updateGradInput_functor(norm));
#else
  auto input_data = THCudaTensor_data(state, input);
  auto target_data = THCudaTensor_data(state, target);
  auto gradInput_data = THCudaTensor_data(state, gradInput);

  bolt::amp::transform(input_data, 
                       input_data+size, 
                       target_data, 
                       gradInput_data,
                       mse_updateGradInput_functor(norm));
#endif

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}
