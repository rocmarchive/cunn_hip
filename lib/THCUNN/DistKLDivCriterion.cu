#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

  #include <thrust/fill.h>
  #include <thrust/functional.h>
  #include <thrust/device_ptr.h>
  #include <thrust/reduce.h>
  #include <thrust/inner_product.h>

template <typename Dtype, typename Acctype>
struct kl_functor
{
  __host__ __device__ Acctype operator()(const Dtype& x, const Dtype& y) const
  {
      Acctype yAcc = ScalarConvert<Dtype, Acctype>::to(y);
      return y > 0 ? yAcc * (THCNumerics<Acctype>::log(yAcc) - x) : Acctype(0);
  }

  __host__ __device__
  ~kl_functor() {}
};

template <typename Dtype>
struct kl_updateGradInput_functor
{
  Dtype norm;

  __host__ __device__
  kl_updateGradInput_functor() = default;

  __host__ __device__
  kl_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  __host__ __device__
  kl_updateGradInput_functor(const kl_updateGradInput_functor& f) = default;

  __host__ __device__ Dtype operator()(const Dtype& x, const Dtype& y) const
  {
      return y > 0 ? norm * (-y) : ScalarConvert<int, Dtype>::to(0);
  }

  __host__ __device__
  ~kl_updateGradInput_functor() {}
};

#include "generic/DistKLDivCriterion.cu"
#include "THCGenerateFloatTypes.h"
