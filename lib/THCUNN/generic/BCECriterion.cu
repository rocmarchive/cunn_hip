#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/BCECriterion.cu"
#else

void THNN_(BCECriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage,
           THCTensor *weights)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_check_dim_size(state, output, 1, 0, 1);
  THCUNN_assertSameGPU(state, 3, input, target, weights);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

#ifdef THRUST_PATH
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));

  accreal sum;
  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<real> weights_data(THCTensor_(data)(state, weights));
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      bce_functor_weights<real, accreal>(),
      (accreal) 0,
      thrust::plus<accreal>()
    );
    THCTensor_(free)(state, weights);
  } else {
    sum = thrust::transform_reduce(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      bce_functor<real, accreal>(),
      (accreal) 0,
      thrust::plus<accreal>()
    );
  }

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
#else
  bolt::amp::Ubiquitous_iterator<real> input_data(THCTensor_(data)(state, input));
  bolt::amp::Ubiquitous_iterator<real> target_data(THCTensor_(data)(state, target));
  real* tresult = nullptr; 
  hipMalloc((void**)&tresult, size*sizeof(real));
  bolt::amp::Ubiquitous_iterator<real> tresult_data(tresult);
  accreal sum;
  if (weights) {
    // #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
    //   weights = THCTensor_(newContiguous)(state, weights);
    //   hipLaunchKernelGGL(hipTorch_apply4<real,accreal>, 64, 64, 0, 0, 
    //     THCTensor_(data)(state, input),
    //     THCTensor_(data)(state, weights),
    //     THCTensor_(data)(state, target),
    //     tresult,
    //     size,
    //     bce_functor_weights<real,accreal>());
    //   sum = bolt::amp::reduce(tresult_data, tresult_data+size, (accreal) 0, bolt::amp::plus<accreal>());
    //   THCTensor_(free)(state, weights);
    // #endif
  } else {
    #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
      hipLaunchKernelGGL((hipTorch_apply3<real,accreal,bce_functor<real,accreal> >), dim3(64), dim3(64), 0, 0, 
        THCTensor_(data)(state, input),
        THCTensor_(data)(state, target),
        tresult,
        size,
        bce_functor<real,accreal>());
      sum = bolt::amp::reduce(tresult_data, tresult_data+size, (accreal) 0, bolt::amp::plus<accreal>());
    #endif
  }
  hipFree(tresult);

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
#endif
}

void THNN_(BCECriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage,
           THCTensor *weights)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_nElement(state, input, weights);
  THCUNN_assertSameGPU(state, 4, input, target, gradInput, weights);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<accreal, real>::to(sizeAverage ? accreal(1)/size : accreal(1));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

#ifdef THRUST_PATH
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    thrust::device_ptr<real> weights_data(THCTensor_(data)(state, weights));
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data, weights_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, weights_data+size)),
      gradInput_data,
      bce_updateGradInput_functor_weights<real, accreal>(norm)
    );
    THCTensor_(free)(state, weights);
  } else {
    thrust::transform(
      thrust::make_zip_iterator(thrust::make_tuple(input_data, target_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size)),
      gradInput_data,
      bce_updateGradInput_functor<real, accreal>(norm)
    );
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
#else
  if (weights) {
    weights = THCTensor_(newContiguous)(state, weights);
    // #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
    //   hipLaunchKernelGGL(hipTorch_apply4<real,accreal>, 64, 64, 0, 0, 
    //     THCTensor_(data)(state, input),
    //     THCTensor_(data)(state, target),
    //     THCTensor_(data)(state, weights),
    //     THCTensor_(data)(state, gradInput),
    //     size,
    //     bce_updateGradInput_functor_weights<real, accreal>(norm));
    //   THCTensor_(free)(state, weights);
    // #endif
  } else {
    // #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
    //   hipLaunchKernelGGL(hipTorch_apply3<real,accreal>, 64, 64, 0, 0, 
    //     THCTensor_(data)(state, input),
    //     THCTensor_(data)(state, target),
    //     THCTensor_(data)(state, gradInput),
    //     size,
    //     bce_updateGradInput_functor<real, accreal>(norm));
    // #endif
  }

  // THCTensor_(free)(state, input);
  // THCTensor_(free)(state, target);
#endif
}

#endif
