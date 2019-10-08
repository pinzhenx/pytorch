#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor dnnl_sigmoid(const Tensor& self) {
  AT_ERROR("dnnl_sigmoid: ATen not compiled with DNNL support");
}

Tensor& dnnl_sigmoid_(Tensor& self) {
  AT_ERROR("dnnl_sigmoid_: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

Tensor dnnl_sigmoid(const Tensor& self) {
  ideep::tensor& x = itensor_from_dnnl(self);
  ideep::tensor y;
  ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return new_with_itensor_dnnl(std::move(y), self.options());
}

Tensor& dnnl_sigmoid_(Tensor& self) {
  ideep::tensor& x = itensor_from_dnnl(self);
  ideep::eltwise_forward::compute(
      x, x, ideep::algorithm::eltwise_logistic, ideep::prop_kind::forward);
  return self;
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
