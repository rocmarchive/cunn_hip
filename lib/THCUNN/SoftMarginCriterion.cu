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
struct softmargin_functor
{
  __host__ __device__ Acctype operator()(const Dtype& x, const Dtype& y) const
  {
    return log(1 + exp(ScalarConvert<Dtype, Acctype>::to(-x)*y));
  }

  __host__ __device__
  ~softmargin_functor() {}
};

template <typename Dtype, typename Acctype>
struct softmargin_updateGradInput_functor
{
  Acctype norm;

  __host__ __device__
  softmargin_updateGradInput_functor() = default;

  __host__ __device__
  softmargin_updateGradInput_functor(Acctype norm_) :
    norm(norm_) {}

  __host__ __device__
  softmargin_updateGradInput_functor(const softmargin_updateGradInput_functor& f) = default;

  __host__ __device__ Dtype operator()(const Dtype& x, const Dtype& y) const
    {
      Acctype temp = exp(ScalarConvert<Dtype, Acctype>::to(-x)*y);
      return ScalarConvert<Acctype, Dtype>::to(-y*temp*norm/(ScalarConvert<int, Acctype>::to(1) + temp));
    }

  __host__ __device__
  ~softmargin_updateGradInput_functor() {}

};

#include "generic/SoftMarginCriterion.cu"
#include "THCGenerateFloatTypes.h"
