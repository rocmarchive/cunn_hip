#include "hip/hip_runtime.h"
#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <THC/THCApply.cuh>

#include "common.h"

template <typename T>
struct PReLUUpdateOutput
{
  T* weight_;

  __host__ __device__
  PReLUUpdateOutput() {}

  __host__ __device__
  PReLUUpdateOutput(T* weight)
    : weight_(weight)
  {}

  __device__ __forceinline__ void operator()(T *out, T *in)
  {
    T x = *in;
    *out = (x > 0) ? x : weight_[0] * x;
  }

  __host__ __device__
  ~PReLUUpdateOutput() {}
};

template <typename T>
__global__ void preluForward(T *output, const T *input, const T *weight, int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    output[i] = input[i] > 0 ? input[i] : input[i] * weight[mapNumber];
  }
}

template <typename T>
struct PReLUUpdateGradInput
{
  T *weight_;

  __host__ __device__
  PReLUUpdateGradInput() {}

  __host__ __device__
  PReLUUpdateGradInput(T *weight)
    : weight_(weight)
  {}

  __device__ __forceinline__ void operator()(T *gradInput, T *gradOutput, T *input)
  {
    *gradInput = *input > 0 ? *gradOutput : *gradOutput * *weight_;
  }

  __host__ __device__
  ~PReLUUpdateGradInput() {}

};

template <typename T>
__global__ void preluBackward(
  T *gradInput,
  const T *input,
  const T *weight,
  const T *gradOutput,
  int n, int nElemsPerSample, int mapSize)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int positionInSample = i % nElemsPerSample;
    int mapNumber = positionInSample / mapSize;
    gradInput[i] = input[i] > 0 ? gradOutput[i] : gradOutput[i] * weight[mapNumber];
  }
}

template <typename T>
struct PReLUAccGradParametersShared
{
  __device__ __forceinline__ void operator()(T *gradInput, T  *input, T *gradOutput)
  {
    *gradInput = (*input) * (*gradOutput) * (*input <= 0);
  }
};

template <typename T>
struct PReLUAccGradParameters
{
  T scale;

  __host__ __device__
  PReLUAccGradParameters() = default;

  __host__ __device__
  PReLUAccGradParameters(T scale)
    : scale(scale)
  {}

  __host__ __device__
  PReLUAccGradParameters(const PReLUAccGradParameters& f) = default;

  __device__ __forceinline__ void operator()(T *gradInput, T *input, T *gradOutput)
  {
    *gradInput = (*input) * (*gradOutput) * scale * (*input <= 0);
  }

  __host__ __device__
  ~PReLUAccGradParameters() {}
};

template <typename T>
struct PReLUAccGradParameters1to1
{
  T scale;

  __host__ __device__
  PReLUAccGradParameters1to1() = default;

  __host__ __device__
  PReLUAccGradParameters1to1(T scale)
    : scale(scale)
  {}

  __host__ __device__
  PReLUAccGradParameters1to1(const PReLUAccGradParameters1to1& f) = default;

  __device__ __forceinline__ void operator()(T *gradWeight, T *input, T *gradOutput)
  {
    *gradWeight += (*input) * (*gradOutput) * scale * (*input <= 0);
  }

  __host__ __device__
  ~PReLUAccGradParameters1to1() {}
};

#include "generic/PReLU.cu"
#include "THCGenerateFloatTypes.h"
