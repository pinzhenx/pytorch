#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/dnnl/DNNLCommon.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at { namespace native {

#if AT_DNNL_ENABLED()

Tensor dnnl_to_dense(const Tensor& dnnl_tensor) {
  auto& stensor = itensor_from_dnnl(dnnl_tensor);
  auto cpu_tensor = at::empty(stensor.get_dims(),
                              dnnl_tensor.options().layout(c10::kStrided));
  stensor.to_public(cpu_tensor.template data_ptr<float>());
  return cpu_tensor;
}

Tensor dense_to_dnnl(const Tensor& cpu_tensor) {
  AT_ASSERTM(cpu_tensor.device().type() == DeviceType::CPU,
             "dense_to_dnnl expects CPU tensor input");
  AT_ASSERTM(cpu_tensor.layout() == Layout::Strided,
             "dense_to_dnnl expects strided tensor input");
  AT_ASSERTM(cpu_tensor.scalar_type() == ScalarType::Float,
             "dense_to_dnnl expects float tensor input");
  AT_ASSERTM(cpu_tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto dnnl_tensor =
      empty_dnnl(cpu_tensor_cont.sizes(), cpu_tensor_cont.options());
  auto& dtensor = itensor_from_dnnl(dnnl_tensor);
  dtensor.feed_from({dtensor.get_dims(), ideep::tensor::data_type::f32,
                     cpu_tensor_cont.template data_ptr<float>()});
  return dnnl_tensor;
}

// Dnnl tensor has special non-public format for conv2d weights
// (dense_to_dnnl only converts dense tensor to dnnl tensor with
// public format). Ideep conv kernel will do implicit reorder if the
// weight is not already in this optimized format. By the time I'm
// writing this note, we are seeing ~20% perf cost of doing the
// on-the-fly reorder.
Tensor dnnl_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {

  // auto stride_vec = expand_param_if_needed(stride, "stride", 2);
  // auto padding_vec = expand_param_if_needed(padding, "padding", 2);
  // auto dilation_vec = expand_param_if_needed(dilation, "dilation", 2);

  // ideep::tensor w = itensor_from_dnnl(self).as_weights();
  // w.make_group(groups);
  // ideep::tensor::descriptor desc =
  //     ideep::convolution_forward::expected_weights_descriptor(
  //         w.get_dims(),
  //         w.get_data_type(),
  //         {stride_vec.cbegin(), stride_vec.cend()},
  //         {padding_vec.cbegin(), padding_vec.cend()},
  //         {padding_vec.cbegin(), padding_vec.cend()},
  //         {dilation_vec.cbegin(), dilation_vec.cend()},
  //         groups,
  //         ideep::algorithm::convolution_direct);
  // ideep::tensor result;
  // result.init<AllocForDNNL>(desc);
  // result.feed_from(w);

  // return new_with_itensor_dnnl(std::move(result), self.options());
  return self;
}

#else

Tensor dnnl_to_dense(const Tensor& dnnl_tensor) {
  AT_ERROR("DNNL build is disabled");
}

Tensor dense_to_dnnl(const Tensor& cpu_tensor) {
  AT_ERROR("DNNL build is disabled");
}

Tensor dnnl_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("dnnl_reorder_conv2d_weight: DNNL build is disabled");
}

#endif // AT_DNNL_ENABLED()

}}
