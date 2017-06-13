#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#if THRUST_PATH
  #include <thrust/functional.h>
  #include <thrust/device_ptr.h>
  #include <thrust/iterator/zip_iterator.h>
  #include <thrust/transform.h>
  #include <thrust/transform_reduce.h>
#else
  #include <bolt/amp/iterator/ubiquitous_iterator.h>
  #include <bolt/amp/transform.h>
  #include <bolt/amp/reduce.h>
#endif

template <typename T>
inline __device__ T eps();

template <>
inline __device__ float eps() { return 1e-12f; }

template <>
inline __device__ double eps() { return 1e-12; }

#if THRUST_PATH
#else
  template <typename RealType, typename AccType>
  __global__
  void hipTorch_apply_bce(RealType* in1, 
                          RealType* in2, 
                          AccType* out, 
                          long numElements) 
  {
    CUDA_KERNEL_LOOP(index, numElements) {
      RealType o = in1[index];
      RealType t = in2[index];
      out[index] =
        - (t * THCNumerics<AccType>::log(o + eps<AccType>()) + (AccType(1)- t) * THCNumerics<AccType>::log(AccType(1) - o + eps<AccType>()));
    } 
  }
  
  template <typename RealType, typename AccType>
  __global__
  void hipTorch_apply_bce_weights(RealType* in1, 
                                  RealType* in2, 
                                  RealType* in3, 
                                  AccType* out, 
                                  long numElements) 
  {
    CUDA_KERNEL_LOOP(index, numElements) {
      RealType o = in1[index];
      RealType t = in2[index];
      RealType w = in3[index];
      out[index] =
        - w * (t * THCNumerics<AccType>::log(o + eps<AccType>()) + (AccType(1) - t) * THCNumerics<AccType>::log(AccType(1) - o + eps<AccType>()));
    } 
  }
  
  template <typename RealType, typename AccType>
  __global__
  void hipTorch_apply_updateGradInput(RealType* in1, 
                                      RealType* in2, 
                                      AccType* out, 
                                      long numElements,
                                      RealType norm) 
  {
    CUDA_KERNEL_LOOP(index, numElements) {
      RealType o = in1[index];
      RealType t = in2[index];
      out[index] =
        ScalarConvert<AccType,RealType>::to(- (t - o) / ((AccType(1) - o + eps<AccType>()) * (o + eps<AccType>())) * norm);
    } 
  }
  
  template <typename RealType, typename AccType>
  __global__
  void hipTorch_apply_updateGradInput_weights(RealType* in1, 
                                              RealType* in2, 
                                              RealType* in3, 
                                              AccType* out, 
                                              long numElements,
                                              RealType norm) 
  {
    CUDA_KERNEL_LOOP(index, numElements) {
      RealType o = in1[index];
      RealType t = in2[index];
      RealType w = in3[index];
      out[index] =
        ScalarConvert<AccType, RealType>::to(- (t - o) / ((AccType(1) - o + eps<AccType>()) * (o + eps<AccType>())) * norm * w);
    } 
  }
#endif

template <typename Dtype, typename Acctype>
struct bce_functor
{
#if THRUST_PATH
  template <class Tuple>
  __host__ __device__
  Acctype operator()(Tuple x)
  {
    Dtype o = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    return - (t * THCNumerics<Acctype>::log(o + eps<Acctype>()) + (Acctype(1)- t) * THCNumerics<Acctype>::log(Acctype(1) - o + eps<Acctype>()));
  }
#else
  __host__ __device__
  Acctype operator()(const Dtype& o, const Dtype& t) const
  {
    return - (t * THCNumerics<Acctype>::log(o + eps<Acctype>()) + (Acctype(1)- t) * THCNumerics<Acctype>::log(Acctype(1) - o + eps<Acctype>()));
  }
#endif
};

template <typename Dtype, typename Acctype>
struct bce_functor_weights
{
  template <class Tuple>
  __host__ __device__
  Acctype operator()(Tuple x)
  {
#if THRUST_PATH
    Dtype o = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    Dtype w = thrust::get<2>(x);
    return - w * (t * THCNumerics<Acctype>::log(o + eps<Acctype>()) + (Acctype(1) - t) * THCNumerics<Acctype>::log(Acctype(1) - o + eps<Acctype>()));
#else
    return Acctype(0);
#endif
  }
};

template <typename Dtype, typename Acctype>
struct bce_updateGradInput_functor
{
  const Dtype norm;

  bce_updateGradInput_functor(Dtype norm_)
    : norm(norm_)
  {}

  template <class Tuple>
  __host__ __device__
  Dtype operator()(Tuple x)
  {
#if THRUST_PATH
    Dtype o = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    return ScalarConvert<Acctype,Dtype>::to(- (t - o) / ((Acctype(1) - o + eps<Acctype>()) * (o + eps<Acctype>())) * norm);
#else
    return Acctype(0);
#endif
  }
};

template <typename Dtype, typename Acctype>
struct bce_updateGradInput_functor_weights
{
  const Dtype norm;

  bce_updateGradInput_functor_weights(Dtype norm_)
    : norm(norm_)
  {}

  template <class Tuple>
  __host__ __device__
  Dtype operator()(Tuple x)
  {
#if THRUST_PATH
    Dtype o = thrust::get<0>(x);
    Dtype t = thrust::get<1>(x);
    Dtype w = thrust::get<2>(x);
    return ScalarConvert<Acctype, Dtype>::to(- (t - o) / ((Acctype(1) - o + eps<Acctype>()) * (o + eps<Acctype>())) * norm * w);
#else
    return Dtype(0);
#endif
  }
};

#include "generic/BCECriterion.cu"
#include "THCGenerateFloatTypes.h"
