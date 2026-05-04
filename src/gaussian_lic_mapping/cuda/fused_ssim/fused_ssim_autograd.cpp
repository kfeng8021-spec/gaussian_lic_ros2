// SPDX-License-Identifier: GPL-3.0-or-later

#include "gaussian_lic_mapping/cuda/fused_ssim.hpp"

#include <torch/torch.h>

namespace gaussian_lic_mapping::cuda_ops
{
namespace
{

class FusedSsimMapFunction : public torch::autograd::Function<FusedSsimMapFunction>
{
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext * context,
    torch::Tensor image1,
    torch::Tensor image2)
  {
    auto image1_contiguous = image1.contiguous();
    auto image2_contiguous = image2.contiguous();
    auto result = fused_ssim_forward_map(
      kFusedSsimC1, kFusedSsimC2, image1_contiguous, image2_contiguous, true);

    context->save_for_backward({
      image1_contiguous.detach(),
      image2_contiguous.detach(),
      std::get<1>(result),
      std::get<2>(result),
      std::get<3>(result)});
    return std::get<0>(result);
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext * context,
    torch::autograd::variable_list grad_outputs)
  {
    const auto saved = context->get_saved_variables();
    auto grad_map = grad_outputs[0].contiguous();
    auto grad_image1 = fused_ssim_backward_map(
      kFusedSsimC1,
      kFusedSsimC2,
      saved[0],
      saved[1],
      grad_map,
      saved[2],
      saved[3],
      saved[4]);
    return {grad_image1, torch::Tensor()};
  }
};

}  // namespace

torch::Tensor fused_ssim_map(const torch::Tensor & image1, const torch::Tensor & image2)
{
  return FusedSsimMapFunction::apply(image1, image2);
}

torch::Tensor fused_ssim(const torch::Tensor & image1, const torch::Tensor & image2)
{
  return fused_ssim_map(image1, image2).mean();
}

}  // namespace gaussian_lic_mapping::cuda_ops
