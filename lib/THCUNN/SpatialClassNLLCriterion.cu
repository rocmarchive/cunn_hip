#include "hip/hip_runtime.h"
#include "THCUNN.h"
#include "common.h"

#include <stdio.h>
#include <assert.h>

// WSTHORNTON
#define THRUST_PATH 0
#if THRUST_PATH
    #include <thrust/functional.h>
#else
    #include <bolt/amp/functional.h>
#endif

#include "/root/grid_launch_variadic/headers/implementation/functions/grid_launch.hpp"

__global__ void cunn_SpatialClassNLLCriterion_updateOutput_kernel(hipLaunchParm lp, 
          float *output,
          float *total_weight,
          float *input,
          long *target,
          float *weights,
          int size_average,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample)
{
  __shared__ float partial_sums[CUDA_NUM_THREADS];

  int i, t;
  float cur_weight;
  float input_sum = 0;
  float acc_weight = 0;

  int sample = hipBlockIdx_x / blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  int step = hipBlockDim_x * blocks_per_sample;
  for (i = (hipBlockIdx_x % blocks_per_sample) * hipBlockDim_x + hipThreadIdx_x;
       i < map_nelem;
       i += step) {
    t = target[toffset + i] - TH_INDEX_BASE;
    assert(t >= 0 && t < n_classes);
    cur_weight = weights ? weights[t] : 1.0f;
    input_sum -= input[ioffset + i + map_nelem * t] * cur_weight;
    acc_weight += cur_weight;
  }

  __syncthreads();

#if THRUST_PATH
  input_sum = reduceBlock(partial_sums, hipBlockDim_x, input_sum, thrust::plus<float>(), 0.0f);
  acc_weight = reduceBlock(partial_sums, hipBlockDim_x, acc_weight, thrust::plus<float>(), 0.0f);
#else
  input_sum = reduceBlock(partial_sums, hipBlockDim_x, input_sum, bolt::amp::plus<float>(), 0.0f);
  acc_weight = reduceBlock(partial_sums, hipBlockDim_x, acc_weight, bolt::amp::plus<float>(), 0.0f);
#endif

  if (hipThreadIdx_x == 0) {
    atomicAdd(total_weight, acc_weight);
    if (size_average && acc_weight > 0)
      atomicAdd(output, input_sum / acc_weight / hipGridDim_x);
    else
      atomicAdd(output, input_sum);
  }
}

__global__ void cunn_SpatialClassNLLCriterion_updateGradInput_kernel(hipLaunchParm lp, 
          float *gradInput,
          long *target,
          float *weights,
          float *total_weight,
          int size_average,
          int batch_size,
          int n_classes,
          int map_nelem,
          int blocks_per_sample)
{
  if (*total_weight <= 0)
    return;

  int i, t;
  float norm = size_average ? (1.0f / *total_weight) : 1.0f;

  int sample = hipBlockIdx_x / blocks_per_sample;
  int step = hipBlockDim_x * blocks_per_sample;
  int toffset = sample * map_nelem;
  int ioffset = sample * map_nelem * n_classes;
  for (i = (hipBlockIdx_x % blocks_per_sample) * hipBlockDim_x + hipThreadIdx_x;
       i < map_nelem;
       i += step) {
    t = (int)target[toffset + i] - TH_INDEX_BASE;
    assert(t >= 0 && t < n_classes);
    gradInput[ioffset + i + map_nelem * t] = -(weights ? weights[t] : 1.0f) * norm;
  }
}

void THNN_CudaSpatialClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaLongTensor *target,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight)
{
  THArgCheck(THCudaLongTensor_nDimension(state, target) == 3, 1,
               "only batches of spatial targets supported (3D tensors)");
  THArgCheck(THCudaTensor_nDimension(state, input) == 4, 2,
               "only batches of spatial inputs supported (4D tensors)");
  if (weights && THCudaTensor_nElement(state, weights) != THCudaTensor_size(state, input, 1)) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (weights)
    THCUNN_assertSameGPU(state, 5, input, target, weights, output, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, output, total_weight);

  input = THCudaTensor_newContiguous(state, input);
  weights = weights ? THCudaTensor_newContiguous(state, weights) : NULL;
  target = THCudaLongTensor_newContiguous(state, target);

  float *input_data = THCudaTensor_data(state, input);
  float *weights_data = weights ? THCudaTensor_data(state, weights) : NULL;
  long  *target_data = THCudaLongTensor_data(state, target);
  float *output_data = THCudaTensor_data(state, output);
  float *total_weight_data = THCudaTensor_data(state, total_weight);

  long batch_size = THCudaLongTensor_size(state, target, 0);
  long map_nelem = THCudaLongTensor_nElement(state, target) / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  THCudaTensor_fill(state, output, 0);
  THCudaTensor_fill(state, total_weight, 0);

  hipLaunchKernelV2(HIP_KERNEL_NAME(cunn_SpatialClassNLLCriterion_updateOutput_kernel), dim3(total_blocks), dim3(CUDA_NUM_THREADS), 0, THCState_getCurrentStream(state), 
      output_data,
      total_weight_data,
      input_data,
      target_data,
      weights_data,
      sizeAverage,
      THCudaTensor_size(state, input, 0),
      THCudaTensor_size(state, input, 1),
      THCudaTensor_size(state, input, 2) * THCudaTensor_size(state, input, 3),
      blocks_per_sample
  );
  THCudaCheck(hipGetLastError());

  if (weights)
    THCudaTensor_free(state, weights);
  THCudaLongTensor_free(state, target);
  THCudaTensor_free(state, input);
}

void THNN_CudaSpatialClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaLongTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight)
{
  THArgCheck(THCudaLongTensor_nDimension(state, target) == 3, 1,
               "only batches of spatial targets supported (3D tensors)");
  THArgCheck(THCudaTensor_nDimension(state, input) == 4, 2,
               "only batches of spatial inputs supported (4D tensors)");
  THArgCheck(THCudaTensor_isContiguous(state, gradInput), 4,
               "gradInput must be contiguous");
  if (weights && THCudaTensor_nElement(state, weights) != THCudaTensor_size(state, input, 1)) {
    THError("weight tensor should be defined either for all or no classes");
  }

  if (weights)
    THCUNN_assertSameGPU(state, 5, weights, input, target, gradInput, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, gradInput, total_weight);

  input = THCudaTensor_newContiguous(state, input);
  weights = weights ? THCudaTensor_newContiguous(state, weights) : NULL;
  target = THCudaLongTensor_newContiguous(state, target);

  float *weights_data = weights ? THCudaTensor_data(state, weights) : NULL;
  float *gradInput_data = THCudaTensor_data(state, gradInput);
  long *target_data = THCudaLongTensor_data(state, target);
  float *total_weight_data = THCudaTensor_data(state, total_weight);

  long batch_size = THCudaLongTensor_size(state, target, 0);
  long map_nelem = THCudaLongTensor_nElement(state, target) / batch_size;
  int blocks_per_sample = GET_BLOCKS(map_nelem) / 128;
  blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
  int total_blocks = blocks_per_sample * batch_size;

  hipLaunchKernelV2(HIP_KERNEL_NAME(cunn_SpatialClassNLLCriterion_updateGradInput_kernel), dim3(total_blocks), dim3(CUDA_NUM_THREADS), 0, THCState_getCurrentStream(state), 
      gradInput_data,
      target_data,
      weights_data,
      total_weight_data,
      sizeAverage,
      THCudaTensor_size(state, input, 0),
      THCudaTensor_size(state, input, 1),
      THCudaTensor_size(state, input, 2) *THCudaTensor_size(state, input, 3),
      blocks_per_sample
  );
  THCudaCheck(hipGetLastError());

  if (weights)
    THCudaTensor_free(state, weights);
  THCudaLongTensor_free(state, target);
  THCudaTensor_free(state, input);
}
