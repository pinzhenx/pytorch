#include <ATen/native/dnnl/DNNLCommon.h>
#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>

#if AT_DNNL_ENABLED()

#include <ideep.hpp>

namespace at { namespace native {

/**
 * `IntrusivePtrTargetWrapper` wraps a custom storage handle  of a tensor
*  (as template param) and inherits `c10::intrusive_ptr_target` so that it
*  can be used with `c10::intrusive_ptr`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 *
 * NOTE: if this is generally useful we may want to move this to its own header.
 */
template <typename T>
struct CAFFE2_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
private:
  T target_;

public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target): target_(target) {}
  IntrusivePtrTargetWrapper(T&& target): target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<ideep::tensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using DNNLTensorImpl = OpaqueTensorImpl<IDeepTensorWrapperPtr>;
using DNNLTensor = Tensor;

Tensor new_with_itensor_dnnl(ideep::tensor&& it, const TensorOptions& options) {
  auto dims = it.get_dims();
  IDeepTensorWrapperPtr handle =
      c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  return detail::make_tensor<DNNLTensorImpl>(
      TensorTypeSet(TensorTypeId::DnnlCPUTensorId), options.dtype(),
      options.device(), handle, dims);
}

ideep::tensor& itensor_from_dnnl(const DNNLTensor& dnnl_tensor) {
  AT_ASSERTM(dnnl_tensor.is_dnnl(), "dnnl_to_dense expects DNNL tensor input");
  AT_ASSERTM(!dnnl_tensor.is_variable(),
             "_internal_get_DNNLImpl: should not be a variable");
  DNNLTensorImpl* mklimpl =
      static_cast<DNNLTensorImpl*>(dnnl_tensor.unsafeGetTensorImpl());
  return mklimpl->unsafe_opaque_handle()->get_target();
}

ideep::tensor empty_dnnl_tensor_like(const ideep::tensor& t) {
  ideep::tensor ret;
  ret.reinit_like(t);
  return ret;
}

ideep::tensor itensor_view_from_dense(const Tensor& tensor) {
  AT_ASSERTM(
      tensor.device().type() == DeviceType::CPU,
      "itensor_view_from_dense expects CPU tensor input");
  AT_ASSERTM(
      tensor.layout() == Layout::Strided,
      "itensor_view_from_dense expects dense tensor input");
  AT_ASSERTM(tensor.scalar_type() == ScalarType::Float,
             "itensor_view_from_dense expects float tensor input");
  AT_ASSERTM(
      !tensor.is_variable(),
      "itensor_view_from_dense: should not be a variable");
  // XPZ: use default cpu engine?
  return {tensor.sizes().vec(),
          ideep::tensor::data_type::f32,
          tensor.template data_ptr<float>(),
          ideep::engine::cpu_engine()};
}
}}

#endif // AT_DNNL_ENABLED()
