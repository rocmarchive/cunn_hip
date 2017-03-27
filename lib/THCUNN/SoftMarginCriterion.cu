#include "THCUNN.h"
#include "common.h"

// WSTHORNTON
#define THRUST_PATH 0

#if THRUST_PATH
    #include <thrust/fill.h>
    #include <thrust/functional.h>
    #include <thrust/device_ptr.h>
    #include <thrust/reduce.h>
    #include <thrust/inner_product.h>
#else
    #include <bolt/amp/functional.h>
    #include <bolt/amp/inner_product.h>
#endif

struct softmargin_functor
{
  __host__ __device__
  softmargin_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return log(1 + exp(-x*y));
  }

  __host__ __device__
  ~softmargin_functor() {}
};


void THNN_CudaSoftMarginCriterion_updateOutput(THCState *state,
                                               THCudaTensor *input,
                                               THCudaTensor *target,
                                               THCudaTensor *output,
                                               int sizeAverage
                                              )
{
  THCUNN_assertSameGPU(state, 2, input, target);
  float sum;

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (float) 0, thrust::plus<float>(), softmargin_functor());
#else
  auto input_data = THCudaTensor_data(state, input);
  auto target_data = THCudaTensor_data(state, target);
  sum = 0.0f;
//  sum = bolt::amp::inner_product(input_data, 
//                                 input_data+size, 
//                                 target_data, 0.0f, 
//                                 bolt::amp::plus<float>(), 
//                                 softmargin_functor());
#endif

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}


struct softmargin_updateGradInput_functor
{
  float norm;

  __host__ __device__ 
  softmargin_updateGradInput_functor() = default;
  __host__ __device__ 
  softmargin_updateGradInput_functor(float norm_) :
    norm(norm_) {}

  softmargin_updateGradInput_functor(const softmargin_updateGradInput_functor& fun) = default;

  __host__ __device__ 
  float operator()(const float& x, const float& y) const
    {
      float temp = exp(-x*y);
      return -y*temp*norm/(1.f + temp);
    }
};

void THNN_CudaSoftMarginCriterion_updateGradInput(THCState *state,
                                                  THCudaTensor *input,
                                                  THCudaTensor *target,
                                                  THCudaTensor *gradInput,
                                                  int sizeAverage
                                                 )
{
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 1./size : 1.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, softmargin_updateGradInput_functor(norm));
#else
  auto input_data = THCudaTensor_data(state, input);
  auto target_data = THCudaTensor_data(state, target);
  auto gradInput_data = THCudaTensor_data(state, gradInput);

  bolt::amp::transform(input_data, 
                       input_data+size, 
                       target_data, 
                       gradInput_data, 
                       softmargin_updateGradInput_functor(norm));
#endif

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}
