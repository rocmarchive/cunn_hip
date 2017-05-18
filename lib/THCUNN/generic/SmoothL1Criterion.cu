#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SmoothL1Criterion.cu"
#else

void THNN_(SmoothL1Criterion_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *output,
           bool sizeAverage)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_check_dim_size(state, output, 1, 0, 1);
  THCUNN_assertSameGPU(state, 2, input, target);
  THArgCheck(
    THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
    "input and target need to have the same number of elements"
  );

  ptrdiff_t size = THCTensor_(nElement)(state, input);

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

#if THRUST_PATH
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  accreal sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (accreal) 0,
    thrust::plus<accreal>(), smoothl1_functor<real, accreal>()
  );
#else
  auto input_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, input));
  auto target_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, target));
  accreal sum = bolt::amp::inner_product(
    input_data, input_data+size, target_data, (accreal) 0,
    bolt::amp::plus<accreal>(), smoothl1_functor<real, accreal>()
  );
#endif

  if (sizeAverage)
    sum /= size;

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);

  THCTensor_(set1d)(state, output, 0, ScalarConvert<accreal, real>::to(sum));
}

void THNN_(SmoothL1Criterion_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *target,
           THCTensor *gradInput,
           bool sizeAverage)
{
  THCUNN_check_nElement(state, input, target);
  THCUNN_assertSameGPU(state, 3, input, target, gradInput);
  THArgCheck(
    THCTensor_(nElement)(state, input) == THCTensor_(nElement)(state, target), 2,
    "input and target need to have the same number of elements"
  );

  ptrdiff_t size = THCTensor_(nElement)(state, input);
  real norm = ScalarConvert<accreal, real>::to(sizeAverage ? accreal(1)/size : accreal(1));

  input = THCTensor_(newContiguous)(state, input);
  target = THCTensor_(newContiguous)(state, target);

  THCTensor_(resizeAs)(state, gradInput, input);

#if THRUST_PATH
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<real> input_data(THCTensor_(data)(state, input));
  thrust::device_ptr<real> target_data(THCTensor_(data)(state, target));
  thrust::device_ptr<real> gradInput_data(THCTensor_(data)(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    smoothl1_updateGradInput_functor<real>(norm)
  );
#else
  auto input_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, input));
  auto target_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, target));
  auto gradInput_data = bolt::amp::make_ubiquitous_iterator(THCTensor_(data)(state, gradInput));

  bolt::amp::transform(
    input_data, input_data+size, target_data, gradInput_data,
    smoothl1_updateGradInput_functor<real>(norm)
  );
#endif

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, target);
}

#endif
