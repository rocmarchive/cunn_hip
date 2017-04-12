#include "hip/hip_runtime.h"
#ifndef THCUNN_COMMON_H
#define THCUNN_COMMON_H

#ifdef __NVCC__
#define CURAND_PATH 1
#endif
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; i < (n); i += hipBlockDim_x * hipGridDim_x)

// #define THCUNN_assertSameGPU(...) THAssertMsg(THCudaTensor_checkGPU(__VA_ARGS__), \
//   "Some of weight/gradient/input tensors are located on different GPUs. Please move them to a single one.")

#define THCUNN_assertSameGPU(...) /* whitespace */

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#endif
