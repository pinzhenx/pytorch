#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor& dnnl_zero_(Tensor& self) {
  AT_ERROR("dnnl_zero_: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

Tensor& dnnl_zero_(Tensor& self) {
  using Vec = vec256::Vec256<float>;

  ideep::tensor& x = itensor_from_dnnl(self);

  auto n = x.get_nelems();
  auto* x_ = static_cast<float*>(x.get_data_handle());
  parallel_for(0, n, 2048, [x_](int64_t begin, int64_t end) {
    vec256::map(
        [](Vec /* unused */) { return 0.0; },
        x_ + begin,
        x_ + begin,
        end - begin);
  });

  return self;
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
