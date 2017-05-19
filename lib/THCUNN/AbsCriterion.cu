#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

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

template <typename Dtype, typename Acctype>
struct abs_functor
{
  __host__ __device__ Acctype operator()(const Dtype& x, const Dtype& y) const
  {
    Dtype z = x-y;
    return ScalarConvert<Dtype, Acctype>::to(z >= 0 ? z : -z);
  }
};

template <typename Dtype>
struct abs_updateGradInput_functor
{
  Dtype norm;

  __host__ __device__
  abs_updateGradInput_functor() = default;

  __host__ __device__
  abs_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  __host__ __device__
  abs_updateGradInput_functor(const abs_updateGradInput_functor& f) = default;

  __host__ __device__ Dtype operator()(const Dtype& x, const Dtype& y) const
  {
    return (x - y) >= 0 ? norm : -norm;
  }

  __host__ __device__
  ~abs_updateGradInput_functor() {}
};

#include "generic/AbsCriterion.cu"
#include "THCGenerateFloatTypes.h"
