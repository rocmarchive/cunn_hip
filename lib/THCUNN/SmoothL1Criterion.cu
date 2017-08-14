#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "THCThrustAllocator.cuh"
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename Dtype, typename Acctype>
struct smoothl1_functor
{
  __host__ __device__
  smoothl1_functor() {}

  __host__ __device__ Acctype operator()(const Dtype &x, const Dtype &y) const
  {
     Acctype z = ScalarConvert<Dtype, Acctype>::to(THCNumerics<Dtype>::abs(x-y));
     return z < Acctype(1) ? 0.5f*z*z : z - 0.5f;
    return Acctype(0.0);
  }

  __host__ __device__
  ~smoothl1_functor() {}
};

template <typename Dtype>
struct smoothl1_updateGradInput_functor
{
  Dtype norm;

  __host__ __device__
  smoothl1_updateGradInput_functor() = default;

  __host__ __device__
  smoothl1_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  __host__ __device__
  smoothl1_updateGradInput_functor(const smoothl1_updateGradInput_functor& f) = default;

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    Dtype z = x - y;
    if (z < ScalarConvert<int, Dtype>::to(-1))
      return -norm;
    else if (z > ScalarConvert<int, Dtype>::to(1))
      return norm;
    else
      return norm * z;
  }

  __host__ __device__
  ~smoothl1_updateGradInput_functor() {}

};

#include "generic/SmoothL1Criterion.cu"
#include "THCGenerateFloatTypes.h"
