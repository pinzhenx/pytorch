#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>


#if !AT_DNNL_ENABLED()

namespace at { namespace native {

Tensor dnnl_relu(const Tensor& input) {
  AT_ERROR("dnnl_relu: ATen not compiled with DNNL support");
}

Tensor& dnnl_relu_(Tensor& input) {
  AT_ERROR("dnnl_relu_: ATen not compiled with DNNL support");
}

}}

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at { namespace native {

Tensor dnnl_relu(const Tensor& input) {
  const ideep::tensor& x = itensor_from_dnnl(input);
  ideep::tensor y;
  ideep::eltwise_forward::compute<AllocForDNNL>(
      x, y, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return new_with_itensor_dnnl(std::move(y), input.options());
}

Tensor& dnnl_relu_(Tensor& input) {
  ideep::tensor& x = itensor_from_dnnl(input);
  ideep::eltwise_forward::compute<AllocForDNNL>(
      x, x, ideep::algorithm::eltwise_relu, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
  return input;
}

}}

#endif // AT_DNNL_EBABLED
