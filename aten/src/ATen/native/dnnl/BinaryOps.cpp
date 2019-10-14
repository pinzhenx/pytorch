#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor& dnnl_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  AT_ERROR("dnnl_add_out: ATen not compiled with DNNL support");
}

Tensor dnnl_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("dnnl_add: ATen not compiled with DNNL support");
}

Tensor& dnnl_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  AT_ERROR("dnnl_add_: ATen not compiled with DNNL support");
}

Tensor& dnnl_mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  AT_ERROR("dnnl_mul_out: ATen not compiled with DNNL support");
}

Tensor dnnl_mul(const Tensor& self, const Tensor& other) {
  AT_ERROR("dnnl_mul: ATen not compiled with DNNL support");
}

Tensor& dnnl_mul_(Tensor& self, const Tensor& other) {
  AT_ERROR("dnnl_mul_: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

Tensor& dnnl_add_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  auto& x = itensor_from_dnnl(self);
  auto& y = itensor_from_dnnl(other);
  auto& z = itensor_from_dnnl(result);
  ideep::sum::compute(/*scales*/{1.0, alpha.to<float>()}, {x, y}, z);
  return result;
}

Tensor dnnl_add(const Tensor& self, const Tensor& other, Scalar alpha) {
  auto& x = itensor_from_dnnl(self);
  auto& y = itensor_from_dnnl(other);
  ideep::tensor z;
  ideep::sum::compute(/*scales*/{1.0, alpha.to<float>()}, {x, y}, z);
  return new_with_itensor_dnnl(std::move(z), self.options());
}

Tensor& dnnl_add_(Tensor& self, const Tensor& other, Scalar alpha) {
  return native::dnnl_add_out(self, self, other, alpha);
}

Tensor& dnnl_mul_out(Tensor& result, const Tensor& self, const Tensor& other) {
  AT_ASSERTM(result.sizes() == self.sizes(),
             "dnnl_mul_out: the output size should be same as input size");
  auto& x = itensor_from_dnnl(self);
  auto& z = itensor_from_dnnl(result);

  // for zero_dim tensor
  if (other.ndimension() == 0) {
    ideep::eltwise_forward::compute(
      x, z, ideep::algorithm::eltwise_linear,
      ideep::prop_kind::forward_inference, /*alpha*/other.item().to<float>());
  } else {
    AT_ASSERTM(self.sizes() == other.sizes(),
               "dnnl_mul_out: currently dnnl not support broadcasting");
    auto& y = itensor_from_dnnl(other);
    ideep::binary::compute(x, y, z, dnnl::algorithm::binary_mul);
  }
  return result;
}

Tensor dnnl_mul(const Tensor& self, const Tensor& other) {
  Tensor result = empty_dnnl(self.sizes(), self.options());
  return native::dnnl_mul_out(result, self, other);
}

Tensor& dnnl_mul_(Tensor& self, const Tensor& other) {
  return native::dnnl_mul_out(self, self, other);
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
