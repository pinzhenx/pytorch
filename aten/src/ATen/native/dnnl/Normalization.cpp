#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_DNNL_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> dnnl_batch_norm(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  AT_ERROR("dnnl_batch_norm: ATen not compiled with DNNL support");
}

} // namespace native
} // namespace at

#else // AT_DNNL_EBABLED

#include <ATen/native/dnnl/DNNLCommon.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> dnnl_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  auto& x = itensor_from_dnnl(input);
  auto& w = itensor_from_dnnl(weight);
  auto& b = itensor_from_dnnl(bias);
  auto& m = itensor_from_dnnl(running_mean);
  auto& v = itensor_from_dnnl(running_var);

  ideep::tensor y;

  if (train) {
    // TODO: support training
    AT_ERROR("dnnl_batch_norm: dnnl training is not supported in yet.");

    // ideep::tensor saved_mean;
    // ideep::tensor saved_var;
    // ideep::batch_normalization_forward_training::compute<AllocForDNNL>(
    //     x, w, b, y, saved_mean, saved_var, m, v, momentum, eps);
    // return std::make_tuple(
    //     new_with_itensor_dnnl(std::move(y), input.options()),
    //     new_with_itensor_dnnl(std::move(saved_mean), input.options()),
    //     new_with_itensor_dnnl(std::move(saved_var), input.options()));
  } else {
    AT_ASSERTM(input.dim() == 4 || input.dim() == 5,
               "dnnl_batch_norm: currently dnnl only support 2d and 3d batchnorm");
    ideep::batch_normalization_forward_inference::compute(
        x, m, v, w, b, y, eps);
    return std::make_tuple(
        new_with_itensor_dnnl(std::move(y), input.options()),
        new_with_itensor_dnnl(ideep::tensor{}, input.options()),
        new_with_itensor_dnnl(ideep::tensor{}, input.options()));
  }
}

} // namespace native
} // namespace at

#endif // AT_DNNL_EBABLED
