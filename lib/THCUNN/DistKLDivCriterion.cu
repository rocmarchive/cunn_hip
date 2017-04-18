#include "THCUNN.h"
#include "common.h"

#if THRUST_PATH
    #include <thrust/fill.h>
    #include <thrust/functional.h>
    #include <thrust/device_ptr.h>
    #include <thrust/reduce.h>
    #include <thrust/inner_product.h>
#else
    #include <bolt/amp/functional.h>
    #include <bolt/amp/inner_product.h>
    #include <bolt/amp/iterator/ubiquitous_iterator.h>
#endif

struct kl_functor
{
  __host__ __device__
  kl_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
      return y > 0 ? y * (log(y) - x) : 0;
  }

  __host__ __device__
  ~kl_functor() {}
};

void THNN_CudaDistKLDivCriterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *output, bool sizeAverage)
{
  THCUNN_assertSameGPU(state, 2, input, target);

  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
             "input and target need to have the same number of elements");

  float sum;

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  sum = thrust::inner_product(input_data, input_data+size, target_data, (float) 0, thrust::plus<float>(), kl_functor());
#else
  auto input_data =
      bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, input));
  auto target_data =
      bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, target));
  sum = bolt::amp::inner_product(input_data,
                                 input_data+size,
                                 target_data, 0.0f,
                                 bolt::amp::plus<float>(),
                                 kl_functor());
#endif

  if (sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct kl_updateGradInput_functor
{
  float norm;

  __host__ __device__
  explicit
  kl_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  kl_updateGradInput_functor(const kl_updateGradInput_functor& fun) = default;

  __host__ __device__
  float operator()(float x, float y) const
  {
      return y > 0 ? norm * (-y) : 0;
  }
};

void THNN_CudaDistKLDivCriterion_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *gradInput, bool sizeAverage)
{
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  THArgCheck(THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
             "input and target need to have the same number of elements");

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 1./size : 1.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

#if THRUST_PATH
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, kl_updateGradInput_functor(norm));
#else
  auto input_data =
      bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, input));
  auto target_data =
      bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, target));
  auto gradInput_data =
      bolt::amp::make_ubiquitous_iterator(THCudaTensor_data(state, gradInput));

  bolt::amp::transform(input_data,
                       input_data + size,
                       target_data,
                       gradInput_data,
                       kl_updateGradInput_functor(norm));
#endif

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}
