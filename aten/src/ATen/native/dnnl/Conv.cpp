#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_DNNL_ENABLED()

namespace at { namespace native {

at::Tensor dnnl_convolution(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  AT_ERROR("dnnl_convolution_forward: ATen not compiled with DNNL support");
}

at::Tensor dnnl_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("dnnl_convolution_backward_input: ATen not compiled with DNNL support");
}

std::tuple<at::Tensor,at::Tensor> dnnl_convolution_backward_weights(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("dnnl_convolution_backward_weights: ATen not compiled with DNNL support");
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> dnnl_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  AT_ERROR("dnnl_convolution_backward: ATen not compiled with DNNL support");
}

}}

#else // AT_DNNL_EBABLED

#include <ATen/dnnl/Runtime.h>
#include <ATen/native/dnnl/DNNLCommon.h>
#include <ATen/native/dnnl/Utils.h>

using namespace dnnl;

namespace {
// Helper function for getting an ideep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the retured ideep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the ideep tensor.
inline ideep::tensor get_dnnl_tensor(const at::Tensor& tensor) {
  if (tensor.is_dnnl()) {
    return at::native::itensor_from_dnnl(tensor);
  } else {
    return at::native::itensor_view_from_dense(tensor);
  }
}
}

namespace at { namespace native {

ideep::tensor _dnnl_conv2d(
    const ideep::tensor& x,
    const ideep::tensor& w,
    const c10::optional<ideep::tensor>& b,
    at::IntArrayRef padding,
    at::IntArrayRef stride,
    at::IntArrayRef dilation,
    int64_t groups) {
  std::vector<int64_t> kernel_size(x.ndims());
  // dnnl conv2d weights could have been re-ordered to 5d by
  // dnnl_reorder_conv2d_weight
  if (w.ndims() == x.ndims() + 1) {
    AT_ASSERTM(
        groups > 1,
        "Only group _dnnl_conv2d weights could have been reordered to 5d");
    kernel_size[0] = w.get_dim(0) * w.get_dim(1);
    std::copy_n(
        w.get_dims().cbegin() + 2, x.ndims() - 1, kernel_size.begin() + 1);
  } else {
    std::copy_n(w.get_dims().cbegin(), x.ndims(), kernel_size.begin());
  }

  auto input_size = x.get_dims();
  auto output_sizes =
      conv_output_size(input_size, kernel_size, padding, stride, dilation);

  ideep::tensor y {output_sizes, ideep::tensor::data_type::f32};
  if (b.has_value()) {
    ideep::convolution_forward::compute(
        x,
        w,
        b.value(),
        y,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  } else {
    ideep::convolution_forward::compute(
        x,
        w,
        y,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  }
  return y;
}

at::Tensor dnnl_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  const ideep::tensor dnnl_input = get_dnnl_tensor(input);
  const ideep::tensor dnnl_weight = get_dnnl_tensor(weight);
  c10::optional<ideep::tensor> dnnl_bias{c10::nullopt};
  if (bias.defined()) {
    dnnl_bias = get_dnnl_tensor(bias);
  }

  ideep::tensor dnnl_output = _dnnl_conv2d(
      dnnl_input,
      dnnl_weight,
      dnnl_bias,
      padding,
      stride,
      dilation,
      groups);

  if (input.is_dnnl()) {
    return new_with_itensor_dnnl(std::move(dnnl_output), input.options());
  } else {
    return dnnl_to_dense(
        new_with_itensor_dnnl(std::move(dnnl_output), input.options()));
  }
}

Tensor dnnl_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // auto grad_input = at::empty(input_size, grad_output.options());

  // auto cpu_engine = CpuEngine::Instance().get_engine();

  // int32_t g = groups;

  // int32_t n = grad_input.size(0);
  // int32_t ic = grad_input.size(1);
  // int32_t ih = grad_input.size(2);
  // int32_t iw = grad_input.size(3);

  // int32_t oc = grad_output.size(1);
  // int32_t oh = grad_output.size(2);
  // int32_t ow = grad_output.size(3);

  // int32_t kh = weight.size(2);
  // int32_t kw = weight.size(3);

  // int32_t sh = stride[0];
  // int32_t sw = stride[1];
  // int32_t ph = padding[0];
  // int32_t pw = padding[1];

  // auto data_t = memory::data_type::f32;
  // auto format_any = memory::format_tag::any;
  // auto format_nchw = memory::format_tag::nchw;
  // auto format_weight = (g!= 1) ? memory::format_tag::goihw : memory::format_tag::oihw;

  // memory::dims input_tz = {n, ic, ih, iw};
  // memory::dims weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
  // memory::dims bias_tz = {oc};
  // memory::dims output_tz = {n, oc, oh, ow};
  // memory::dims _stride = {sh, sw};
  // memory::dims _padding = {ph, pw};

  // auto input_md = memory::desc({input_tz}, data_t, format_any);
  // auto weight_md = memory::desc({weight_tz}, data_t, format_any);
  // auto bias_md = memory::desc({bias_tz}, data_t, format_any);
  // auto output_md = memory::desc({output_tz}, data_t, format_any);

  // // need to re-create conv_forward_pd to feed conv_backward_data_pd
  // std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  // if (bias_defined) {
  //   conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
  //     convolution_direct, input_md, weight_md, bias_md, output_md,
  //     _stride, _padding, _padding));
  // } else {
  //   conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
  //     convolution_direct, input_md, weight_md, output_md,
  //     _stride, _padding, _padding));
  // }

  // std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  // conv_forward_pd.reset(new convolution_forward::primitive_desc(
  //   *conv_forward_desc, cpu_engine));

  // std::shared_ptr<convolution_backward_data::desc> conv_backward_data_desc;
  // conv_backward_data_desc.reset(new convolution_backward_data::desc(
  //   convolution_direct, input_md, weight_md, output_md,
  //   _stride, _padding, _padding));

  // std::shared_ptr<convolution_backward_data::primitive_desc> conv_backward_data_pd;
  // conv_backward_data_pd.reset(new convolution_backward_data::primitive_desc(
  //   *conv_backward_data_desc, cpu_engine, *conv_forward_pd));

  // auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, cpu_engine},
  //   grad_output.data_ptr());
  // auto weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, cpu_engine},
  //   weight.data_ptr());
  // auto grad_input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, cpu_engine},
  //   grad_input.data_ptr());

  // std::vector<primitive> net;

  // auto grad_output_pd = conv_backward_data_pd->diff_dst_primitive_desc();
  // auto grad_output_memory = grad_output_usr_memory;
  // if (grad_output_usr_memory.get_primitive_desc() != memory::primitive_desc(grad_output_pd)) {
  //   grad_output_memory = memory(grad_output_pd);
  //   net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
  // }

  // auto weight_pd = conv_backward_data_pd->weights_primitive_desc();
  // auto weight_memory = weight_usr_memory;
  // if (weight_usr_memory.get_primitive_desc() != memory::primitive_desc(weight_pd)) {
  //   weight_memory = memory(weight_pd);
  //   net.push_back(reorder(weight_usr_memory, weight_memory));
  // }

  // auto grad_input_pd = conv_backward_data_pd->diff_src_primitive_desc();
  // auto grad_input_memory = grad_input_usr_memory;
  // if (grad_input_memory.get_primitive_desc() != memory::primitive_desc(grad_input_pd)) {
  //   grad_input_memory = memory(grad_input_pd);
  // }

  // std::shared_ptr<convolution_backward_data> conv_backward_data;
  // conv_backward_data.reset(new convolution_backward_data(*conv_backward_data_pd,
  //   grad_output_memory, weight_memory, grad_input_memory));
  // net.push_back(*conv_backward_data);

  // if (grad_input_memory != grad_input_usr_memory) {
  //   net.push_back(reorder(grad_input_memory, grad_input_usr_memory));
  // }

  // Stream::Instance().get_stream().submit(net);

  // return grad_input;
  return grad_output;
}

std::tuple<at::Tensor, at::Tensor> dnnl_convolution_backward_weights(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  // auto grad_weight = at::empty(weight_size, grad_output.options());

  // Tensor grad_bias;
  // if (bias_defined) {
  //   grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  // }

  // auto cpu_engine = CpuEngine::Instance().get_engine();

  // int32_t g = groups;

  // int32_t n = input.size(0);
  // int32_t ic = input.size(1);
  // int32_t ih = input.size(2);
  // int32_t iw = input.size(3);

  // int32_t oc = grad_output.size(1);
  // int32_t oh = grad_output.size(2);
  // int32_t ow = grad_output.size(3);

  // int32_t kh = grad_weight.size(2);
  // int32_t kw = grad_weight.size(3);

  // int32_t sh = stride[0];
  // int32_t sw = stride[1];
  // int32_t ph = padding[0];
  // int32_t pw = padding[1];

  // auto data_t = memory::data_type::f32;
  // auto format_any = memory::format_tag::any;
  // auto format_nchw = memory::format_tag::nchw;
  // auto format_weight = (g!= 1) ? memory::format_tag::goihw : memory::format_tag::oihw;
  // auto format_x = memory::format_tag::x;

  // memory::dims input_tz = {n, ic, ih, iw};
  // memory::dims weight_tz = (g!= 1) ? memory::dims{g, oc/g, ic/g, kh, kw} : memory::dims{oc, ic, kh, kw};
  // memory::dims bias_tz = {oc};
  // memory::dims output_tz = {n, oc, oh, ow};
  // memory::dims _stride = {sh, sw};
  // memory::dims _padding = {ph, pw};

  // memory::desc input_md({input_tz}, data_t, format_any);
  // memory::desc weight_md({weight_tz}, data_t, format_any);
  // memory::desc bias_md({bias_tz}, data_t, format_any);
  // memory::desc output_md({output_tz}, data_t, format_any);

  // // need to re-create conv_forward_pd to feed conv_backward_weight_pd
  // std::shared_ptr<convolution_forward::desc> conv_forward_desc;
  // if (bias_defined) {
  //   conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
  //     convolution_direct, input_md, weight_md, bias_md, output_md,
  //     _stride, _padding, _padding));
  // } else {
  //   conv_forward_desc.reset(new convolution_forward::desc(prop_kind::forward,
  //     convolution_direct, input_md, weight_md, output_md,
  //     _stride, _padding, _padding));
  // }

  // std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  // conv_forward_pd.reset(new convolution_forward::primitive_desc(
  //   *conv_forward_desc, cpu_engine));

  // std::shared_ptr<convolution_backward_weights::desc> conv_backward_weight_desc;
  // if (bias_defined) {
  //   conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
  //     convolution_direct, input_md, weight_md, bias_md, output_md,
  //     _stride, _padding, _padding));
  // } else {
  //   conv_backward_weight_desc.reset(new convolution_backward_weights::desc(
  //     convolution_direct, input_md, weight_md, output_md,
  //     _stride, _padding, _padding));
  // }

  // std::shared_ptr<convolution_backward_weights::primitive_desc> conv_backward_weight_pd;
  // conv_backward_weight_pd.reset(new convolution_backward_weights::primitive_desc(
  //   *conv_backward_weight_desc, cpu_engine, *conv_forward_pd));

  // auto input_usr_memory = memory({{{input_tz}, data_t, format_nchw}, cpu_engine},
  //   input.data_ptr());
  // auto grad_output_usr_memory = memory({{{output_tz}, data_t, format_nchw}, cpu_engine},
  //   grad_output.data_ptr());
  // auto grad_weight_usr_memory = memory({{{weight_tz}, data_t, format_weight}, cpu_engine},
  //   grad_weight.data_ptr());
  // std::shared_ptr<memory> grad_bias_memory;

  // std::vector<primitive> net;

  // auto input_pd = conv_backward_weight_pd->src_primitive_desc();
  // auto input_memory = input_usr_memory;
  // if (input_usr_memory.get_primitive_desc() != memory::primitive_desc(input_pd)) {
  //   input_memory = memory(input_pd);
  //   net.push_back(reorder(input_usr_memory, input_memory));
  // }

  // auto grad_output_pd = conv_backward_weight_pd->diff_dst_primitive_desc();
  // auto grad_output_memory = grad_output_usr_memory;
  // if (grad_output_usr_memory.get_primitive_desc() != memory::primitive_desc(grad_output_pd)) {
  //   grad_output_memory = memory(grad_output_pd);
  //   net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
  // }

  // auto grad_weight_pd = conv_backward_weight_pd->diff_weights_primitive_desc();
  // auto grad_weight_memory = grad_weight_usr_memory;
  // if (grad_weight_usr_memory.get_primitive_desc() != memory::primitive_desc(grad_weight_pd)) {
  //   grad_weight_memory = memory(grad_weight_pd);
  // }

  // std::shared_ptr<convolution_backward_weights> conv_backward_weight;
  // if (bias_defined) {
  //   grad_bias_memory.reset(new memory({{{bias_tz}, data_t, format_x}, cpu_engine},
  //     grad_bias.data_ptr()));
  //   conv_backward_weight.reset(new convolution_backward_weights(*conv_backward_weight_pd,
  //     input_memory, grad_output_memory, grad_weight_memory, *grad_bias_memory));
  // } else {
  //   conv_backward_weight.reset(new convolution_backward_weights(*conv_backward_weight_pd,
  //     input_memory, grad_output_memory, grad_weight_memory));
  // }

  // net.push_back(*conv_backward_weight);

  // if (grad_weight_memory != grad_weight_usr_memory) {
  //   net.push_back(reorder(grad_weight_memory, grad_weight_usr_memory));
  // }

  // Stream::Instance().get_stream().submit(net);

  // return std::tuple<at::Tensor, at::Tensor>{grad_weight, grad_bias};
  return std::tuple<at::Tensor, at::Tensor>{input, input};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> dnnl_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output_t, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::dnnl_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::dnnl_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native

#endif
