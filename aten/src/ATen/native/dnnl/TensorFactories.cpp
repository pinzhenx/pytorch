#include <ATen/native/dnnl/DNNLCommon.h>

namespace at { namespace native {

#if AT_DNNL_ENABLED()

Tensor empty_dnnl(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(!optional_memory_format.has_value(),
              "'memory_format' argument is incompatible with dnnl tensor");
  ideep::tensor it{sizes.vec(), ideep::tensor::data_type::f32};
  return new_with_itensor_dnnl(std::move(it), options);
}

#else

Tensor empty_dnnl(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR("empty_dnnl: DNNL build is disabled");
}

#endif // AT_DNNL_ENABLED()

}}
