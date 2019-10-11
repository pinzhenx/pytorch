#include <ATen/native/dnnl/DNNLCommon.h>

namespace at { namespace native {

#if AT_DNNL_ENABLED()

Tensor empty_dnnl(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK( 
     !optional_memory_format.has_value(),
     "'memory_format' argument is incompatible with dnnl tensor");
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  // TODO: support int64_t dims in ideep::tensor to avoid extra conversion
  ideep::tensor::dims dst_dims (sizes.begin(), sizes.end());
  ideep::tensor it {dst_dims, ideep::tensor::data_type::f32};
  return new_with_itensor_dnnl(std::move(it), options);
}

#else

Tensor empty_dnnl(IntArrayRef sizes, const TensorOptions& options, c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR("empty_dnnl: DNNL build is disabled");
}

#endif // AT_DNNL_ENABLED()

}}
