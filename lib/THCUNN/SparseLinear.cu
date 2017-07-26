#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"
#include <hipsparse.h>

static hipsparseHandle_t hipsparse_handle = 0;

static void init_hipsparse() {
  if (hipsparse_handle == 0) {
    hipsparseStatus_t status = hipsparseCreate(&hipsparse_handle);
    if (status != HIPSPARSE_STATUS_SUCCESS) {
      THError("HIPSPARSE Library initialization failed");
    }
  }
}

#ifdef CUDA_HALF_TENSOR
void THNN_CudaHalfSparseLinear_updateOutput(
          THCState *state,
          THCudaHalfTensor *input,
          THCudaHalfTensor *output,
          THCudaHalfTensor *weight,
          THCudaHalfTensor *bias) {
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfSparseLinear_accGradParameters(
          THCState *state,
          THCudaHalfTensor *input,
          THCudaHalfTensor *gradOutput,
          THCudaHalfTensor *gradWeight,
          THCudaHalfTensor *gradBias,
          THCudaHalfTensor *weight,
          THCudaHalfTensor *bias,
          float weightDecay,
          float scale) {
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfSparseLinear_legacyUpdateOutput(
          THCState *state,
          THCudaHalfTensor *input,
          THCudaHalfTensor *output,
          THCudaHalfTensor *weight,
          THCudaHalfTensor *bias) {
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfSparseLinear_legacyAccGradParameters(
          THCState *state,
          THCudaHalfTensor *input,
          THCudaHalfTensor *gradOutput,
          THCudaHalfTensor *gradWeight,
          THCudaHalfTensor *gradBias,
          THCudaHalfTensor *weight,
          THCudaHalfTensor *bias,
          float weightDecay,
          float scale) {
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfSparseLinear_zeroGradParameters(
          THCState *state,
          THCudaHalfTensor *gradWeight,
          THCudaHalfTensor *gradBias,
          THCudaHalfTensor *lastInput) {
  THError("THCudaHalfTensor not supported with SparseLinear");
}

void THNN_CudaHalfSparseLinear_updateParameters(
          THCState *state,
          THCudaHalfTensor *weight,
          THCudaHalfTensor *bias,
          THCudaHalfTensor *gradWeight,
          THCudaHalfTensor *gradBias,
          THCudaHalfTensor *lastInput,
          float learningRate) {
  THError("THCudaHalfTensor not supported with SparseLinear");
}
#endif

#include "generic/SparseLinear.cu"
#include "THCGenerateFloatType.h"
#include "generic/SparseLinear.cu"
#include "THCGenerateDoubleType.h"
