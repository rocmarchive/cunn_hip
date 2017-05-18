#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/AbsCriterion.cu"
#else

void THNN_(AbsCriterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 2, input, target);

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

#if THRUST_PATH
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  accreal sum = thrust::inner_product(input_data, input_data+size, target_data, (accreal)0, thrust::plus<accreal>(), abs_functor<real, accreal>());
#else
  auto input_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, input));
  auto target_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, target));
  accreal sum = bolt::amp::inner_product(input_data, input_data+size, target_data, (accreal)0, bolt::amp::plus<accreal>(), abs_functor<real, accreal>());
#endif

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(AbsCriterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<double, real>::to(sizeAverage ? 1./size : 1.);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

#if THRUST_PATH
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(input_data, input_data+size, target_data, gradInput_data, abs_updateGradInput_functor<real>(norm));
#else
  auto input_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, input));
  auto target_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, target));
  auto gradInput_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, gradInput));

  bolt::amp::transform(input_data, input_data+size, target_data, gradInput_data, abs_updateGradInput_functor<real>(norm));
#endif

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
