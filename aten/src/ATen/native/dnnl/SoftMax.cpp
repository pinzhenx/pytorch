#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor dnnl_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  AT_ERROR("dnnl_softmax: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

Tensor dnnl_softmax(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  AT_ASSERTM(
      !half_to_float,
      "softmax with half to float conversion is not supported on Dnnl");
  const int64_t wrapped_dim = maybe_wrap_dim(dim, self.dim());
  auto& x = itensor_from_dnnl(self);
  ideep::tensor y;
  ideep::softmax_forward::compute(x, y, wrapped_dim);
  return new_with_itensor_dnnl(std::move(y), self.options());
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
