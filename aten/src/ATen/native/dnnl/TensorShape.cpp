#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor dnnl_view(const Tensor& self, IntArrayRef size) {
  AT_ERROR("dnnl_reshape: ATen not compiled with DNNL support");
}

Tensor dnnl_reshape(const Tensor& self, IntArrayRef size) {
  AT_ERROR("dnnl_reshape: ATen not compiled with DNNL support");
}

Tensor dnnl_clone(const Tensor& self) {
  AT_ERROR("dnnl_clone: ATen not compiled with DNNL support");
}

Tensor dnnl_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("dnnl_transpose: ATen not compiled with DNNL support");
}

Tensor& dnnl_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("dnnl_transpose_: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

Tensor dnnl_view(const Tensor& self, IntArrayRef size) {
  AT_ERROR(
      "Currently Dnnl tensor does not support view. Change to use reshape instead");
}

Tensor dnnl_reshape(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const ideep::tensor& x = itensor_from_dnnl(self);
  ideep::tensor y{x};
  y.reshape<AllocForDNNL>({inferred_size.cbegin(), inferred_size.cend()});
  return new_with_itensor_dnnl(std::move(y), self.options());
}

Tensor dnnl_clone(const Tensor& self) {
  ideep::tensor& src = itensor_from_dnnl(self);
  ideep::tensor dst;
  ideep::direct_copy::compute<AllocForDNNL>(src, dst);
  return new_with_itensor_dnnl(std::move(dst), self.options());
}

Tensor dnnl_transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
  const ideep::tensor& x = itensor_from_dnnl(self);
  ideep::tensor y;
  std::vector<int> axes(x.ndims());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[dim0], axes[dim1]);
  y.transpose_from<AllocForDNNL>(x, axes);
  return new_with_itensor_dnnl(std::move(y), self.options());
}

Tensor& dnnl_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  AT_ERROR("dnnl_transpose_: in-place dnnl operations are not supported yet");
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
