#pragma once

#include <ATen/ATen.h>
#include <ATen/Config.h>

#if AT_DNNL_ENABLED()
#include <ideep.hpp>

namespace at { namespace native {

// Custom allocator using c10 CPU allocator for `ideep::tensor`
struct AllocForDNNL {
  static char* malloc(size_t size) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    return (char*)allocator->raw_allocate(size);
  }

  static void free(void* p) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    allocator->raw_deallocate(p);
  }
};

// Construct aten DNNL tensor given an ideep tensor
Tensor new_with_itensor_dnnl(ideep::tensor&& it, const TensorOptions& options);

// Retrieve `ideep::tensor` from DNNL tensor
ideep::tensor& itensor_from_dnnl(const Tensor& dnnl_tensor);

// Construct an `ideep::tensor` "view" from dense tensor, note the
// ideep::tensor will share the underlying buffer
ideep::tensor itensor_view_from_dense(const Tensor& tensor);

ideep::tensor empty_dnnl_tensor_like(const ideep::tensor& t);
}}

#endif // AT_DNNL_ENABLED
