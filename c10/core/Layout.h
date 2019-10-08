#pragma once

#include <c10/core/Backend.h>
#include <c10/util/Exception.h>

#include <iostream>

namespace c10 {
enum class Layout : int8_t { Strided, Sparse, Dnnl };

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kDnnl = Layout::Dnnl;

inline Layout layout_from_backend(Backend backend) {
  switch (backend) {
    case Backend::SparseCPU:
    case Backend::SparseCUDA:
    case Backend::SparseHIP:
      return Layout::Sparse;
    case Backend::DnnlCPU:
      return Layout::Dnnl;
    default:
      return Layout::Strided;
  }
}

inline std::ostream& operator<<(std::ostream& stream, at::Layout layout) {
  switch (layout) {
    case at::kStrided:
      return stream << "Strided";
    case at::kSparse:
      return stream << "Sparse";
    case at::kDnnl:
      return stream << "Dnnl";
    default:
      AT_ERROR("Unknown layout");
  }
}

} // namespace c10
