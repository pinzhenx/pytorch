#pragma once

#include <ATen/ATen.h>

namespace at {
namespace native {

Tensor dnnl_view(const Tensor& self, IntArrayRef size);

Tensor dnnl_clone(const Tensor& self);

} // namespace native
} // namespace at
