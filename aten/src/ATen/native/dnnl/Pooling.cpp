#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/utils/ParamUtils.h>
#include <tuple>


#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

Tensor dnnl_max_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  AT_ERROR(
      "dnnl_max_pool2d: ATen not compiled with DNNL support");
}

Tensor dnnl_avg_pool2d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  AT_ERROR("dnnl_avg_pool2d: ATen not compiled with DNNL support");
}

Tensor& dnnl_avg_pool2d_out(
    Tensor& output,
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  AT_ERROR("dnnl_avg_pool2d_out: ATen not compiled with DNNL support");
}

Tensor dnnl_adaptive_avg_pool2d(Tensor const& input, IntArrayRef output_size) {
  AT_ERROR("dnnl_adaptive_avg_pool2d: ATen not compiled with DNNL support");
}

Tensor& dnnl_adaptive_avg_pool2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  AT_ERROR(
      "dnnl_adaptive_avg_pool2d_out: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_ENABLED

#include <ATen/native/dnnl/DNNLCommon.h>
#include <ATen/native/dnnl/Utils.h>

namespace at {
namespace native {

static Tensor _dnnl_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    ideep::algorithm algo) {
  auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", 2);
  auto stride_vec = expand_param_if_needed(stride, "stride", 2);
  auto padding_vec = expand_param_if_needed(padding, "padding", 2);
  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", 2);

  const ideep::tensor& x = itensor_from_dnnl(input);
  std::vector<int64_t> output_sizes;

  if (ceil_mode) {
    // DNNL does not support ceil mode, so we adjust padding
    // on the right side to match behavior. Adjust output size
    // accordingly.
    const std::vector<int64_t> output_sizes_ceil = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        true /* ceil_mode */);

    // adjust padding until output sizes agree
    bool all_equal = false;
    while (!all_equal) {
      output_sizes = pool_output_sizes(
          input.sizes(),
          kernel_size_vec,
          stride_vec,
          padding_vec_l,
          padding_vec_r,
          dilation_vec,
          false /*ceil_mode */);

      all_equal = true;
      for (size_t i = 2; i < input.sizes().size(); ++i) {
        if (output_sizes[i] < output_sizes_ceil[i]) {
           padding_vec_r[i - 2]++;
           all_equal = false;
        }
      }
    }
  } else {
    output_sizes = pool_output_sizes(
        input.sizes(),
        kernel_size_vec,
        stride_vec,
        padding_vec_l,
        padding_vec_r,
        dilation_vec,
        false /*ceil_mode */);
  }

  // XPZ: TODO: forward training?
  ideep::tensor y {output_sizes, ideep::tensor::data_type::f32, nullptr};
  ideep::pooling_forward::compute(
      x,
      y,
      stride_vec,
      kernel_size_vec,
      padding_vec_l,
      padding_vec_r,
      algo,
      ideep::prop_kind::forward_inference);

  return new_with_itensor_dnnl(std::move(y), input.options());
}

Tensor dnnl_max_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  return _dnnl_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      ideep::algorithm::pooling_max);
}

Tensor dnnl_avg_pool2d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  TORCH_CHECK(!divisor_override.has_value(),
           "dnnl_avg_pool2d operator does not support divisor");
  return _dnnl_pool2d(
      input,
      kernel_size,
      stride,
      padding,
      /*dilation*/ {1, 1},
      ceil_mode,
      count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                        : ideep::algorithm::pooling_avg_exclude_padding);
}

Tensor& dnnl_avg_pool2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  AT_ERROR(
      "dnnl_avg_pool2d_out: in-place dnnl operations are not supported yet");
}

Tensor dnnl_adaptive_avg_pool2d(
    Tensor const& input,
    IntArrayRef output_size) {
  AT_ASSERTM(input.dim() == 4, "dnnl_adaptive_avg_pool2d: Expect 2D input");

  auto output_size_vec =
      expand_param_if_needed(output_size, "output_size", input.dim() - 2);
  std::vector<int64_t> kernel_size(input.dim() - 2);
  for (int64_t i = 2; i < input.dim(); ++i) {
    auto s1 = input.size(i);
    auto s2 = output_size_vec[i - 2];
    AT_ASSERTM(s2 != 0, "output size can not be zero");
    AT_ASSERTM(
        s1 % s2 == 0,
        "input size is not divisible by the output size is not supported yet");
    kernel_size[i - 2] = s1 / s2;
  }
  return _dnnl_pool2d(
      input,
      kernel_size,
      /*stride*/ kernel_size,
      /*padding*/ {0, 0},
      /*dilation*/ {1, 1},
      /*ceil_mode*/ false,
      /*algo*/ ideep::algorithm::pooling_avg);
}

Tensor& dnnl_adaptive_avg_pool2d_out(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size) {
  AT_ERROR(
      "dnnl_adaptive_avg_pool2d_out: in-place dnnl operations are not supported yet");
}


} // namespace native
} // namespace at

#endif // AT_DNNL_ENABLED
