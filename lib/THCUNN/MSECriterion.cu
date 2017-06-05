#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#if THRUST_PATH
#include "THCThrustAllocator.cuh"
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
  #include <bolt/amp/iterator/ubiquitous_iterator.h>
#endif

template <typename Dtype, typename Acctype>
struct mse_functor
{
  __host__ __device__
  mse_functor() {}

  __host__ __device__ Acctype operator()(const Dtype &x, const Dtype &y) const
  {
     Acctype z = ScalarConvert<Dtype, Acctype>::to(x)-y;
     return z*z;
    return Acctype(0.0);
  }
};

template <typename Dtype, typename Acctype>
struct mse_updateGradInput_functor
{
  Acctype norm;

  __host__ __device__
  mse_updateGradInput_functor() = default;

  __host__ __device__
  mse_updateGradInput_functor(Acctype norm_)
    : norm(norm_)
  {}

  __host__ __device__
  mse_updateGradInput_functor(const mse_updateGradInput_functor& f) = default;

  __host__ __device__ Dtype operator()(const Dtype &x, const Dtype &y) const
  {
    return ScalarConvert<Acctype, Dtype>::to(norm * (ScalarConvert<Dtype, Acctype>::to(x) - y));
    return Dtype(0.0);
  }

  __host__ __device__
  ~mse_updateGradInput_functor() {}
};

#include "generic/MSECriterion.cu"
#include "THCGenerateFloatTypes.h"
