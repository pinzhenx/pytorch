#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor dnnl_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  AT_ERROR("dnnl_linear: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

Tensor dnnl_linear(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias) {
  TORCH_CHECK(self.dim() >= 2,
      "dnnl_linear: input needs to has dim at least 2, input dim ", self.dim());
  TORCH_CHECK(self.is_dnnl(),
      "dnnl_linear: input needs to be dnnl layout");
  TORCH_CHECK(weight.is_dnnl() && bias.is_dnnl(),
      "dnnl_linear: weight and bias need to be dnnl layout");

  // reshape first if input dim is greater than 2 and the reshape will cost a memory copy.
  auto self_reshaped = self.dim() > 2 ? self.reshape({-1, self.size(self.dim() - 1)}) : self;
  const ideep::tensor x = itensor_from_dnnl(self_reshaped);
  const ideep::tensor w = itensor_from_dnnl(weight);

  ideep::tensor y;
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_dnnl(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() > 2) {
    return new_with_itensor_dnnl(std::move(y), self.options()).reshape(output_size);
  }
  return new_with_itensor_dnnl(std::move(y), self.options());
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
