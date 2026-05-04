// SPDX-License-Identifier: GPL-3.0-or-later

#include "gaussian_lic_mapping/cuda/fused_ssim.hpp"

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{

torch::Tensor gaussian_window(const torch::Tensor & image)
{
  static const std::vector<float> weights{
    0.001028380123898387F,
    0.0075987582094967365F,
    0.036000773310661316F,
    0.10936068743467331F,
    0.21300552785396576F,
    0.26601171493530273F,
    0.21300552785396576F,
    0.10936068743467331F,
    0.036000773310661316F,
    0.0075987582094967365F,
    0.001028380123898387F};
  auto g = torch::tensor(weights, torch::TensorOptions().dtype(torch::kFloat32))
    .to(image.device())
    .view({11, 1});
  auto window = torch::mm(g, g.t()).view({1, 1, 11, 11});
  return window.expand({image.size(1), 1, 11, 11}).contiguous();
}

torch::Tensor reference_ssim_map(const torch::Tensor & image1, const torch::Tensor & image2)
{
  namespace F = torch::nn::functional;
  const auto channel_count = image1.size(1);
  const auto window = gaussian_window(image1);
  const auto conv_options = F::Conv2dFuncOptions().padding(5).groups(channel_count);

  const auto mu1 = F::conv2d(image1, window, conv_options);
  const auto mu2 = F::conv2d(image2, window, conv_options);
  const auto mu1_sq = mu1.pow(2);
  const auto mu2_sq = mu2.pow(2);
  const auto mu1_mu2 = mu1 * mu2;

  const auto sigma1_sq = F::conv2d(image1 * image1, window, conv_options) - mu1_sq;
  const auto sigma2_sq = F::conv2d(image2 * image2, window, conv_options) - mu2_sq;
  const auto sigma12 = F::conv2d(image1 * image2, window, conv_options) - mu1_mu2;

  constexpr float c1 = gaussian_lic_mapping::cuda_ops::kFusedSsimC1;
  constexpr float c2 = gaussian_lic_mapping::cuda_ops::kFusedSsimC2;
  return ((2.0F * mu1_mu2 + c1) * (2.0F * sigma12 + c2)) /
    ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2));
}

float max_abs_diff(const torch::Tensor & lhs, const torch::Tensor & rhs)
{
  return (lhs.detach() - rhs.detach()).abs().max().item<float>();
}

float abs_value_diff(const torch::Tensor & lhs, const torch::Tensor & rhs)
{
  return std::abs(lhs.detach().item<float>() - rhs.detach().item<float>());
}

}  // namespace

int main()
{
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available\n";
    return 2;
  }

  torch::manual_seed(20260504);
  const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  constexpr int full_height = 960;
  constexpr int full_width = 1280;
  auto image1 = torch::rand({1, 3, full_height, full_width}, options).contiguous();
  auto image2 = torch::rand({1, 3, full_height, full_width}, options).contiguous();

  const auto fused_map = gaussian_lic_mapping::cuda_ops::fused_ssim_map(image1, image2);
  const auto reference_map = reference_ssim_map(image1, image2);
  const auto forward_map_error = max_abs_diff(fused_map, reference_map);
  const auto forward_mean_map_error = (fused_map.detach() - reference_map.detach()).abs().mean().item<float>();
  const auto forward_value_error = abs_value_diff(fused_map.mean(), reference_map.mean());

  auto grad_image1 = torch::rand({1, 3, 32, 40}, options).contiguous();
  grad_image1.set_requires_grad(true);
  auto grad_image2 = torch::rand({1, 3, 32, 40}, options).contiguous();

  auto fused_value = gaussian_lic_mapping::cuda_ops::fused_ssim(grad_image1, grad_image2);
  fused_value.backward();
  const auto fused_grad = grad_image1.grad().detach().clone();

  auto reference_image1 = grad_image1.detach().clone();
  reference_image1.set_requires_grad(true);
  auto reference_image2 = grad_image2.detach().clone();
  auto reference_value = reference_ssim_map(reference_image1, reference_image2).mean();
  reference_value.backward();
  const auto gradient_error = max_abs_diff(fused_grad, reference_image1.grad());

  cudaDeviceSynchronize();
  std::cout << "fused_ssim_probe full=" << full_width << "x" << full_height
            << " forward_map_error=" << forward_map_error
            << " forward_mean_map_error=" << forward_mean_map_error
            << " forward_value_error=" << forward_value_error
            << " gradient_error=" << gradient_error << "\n";

  if (!std::isfinite(forward_mean_map_error) || forward_mean_map_error > 1.0e-4F) {
    std::cerr << "fused SSIM forward mean map mismatch exceeds tolerance\n";
    return 1;
  }
  if (!std::isfinite(forward_value_error) || forward_value_error > 1.0e-3F) {
    std::cerr << "fused SSIM scalar mismatch exceeds tolerance\n";
    return 1;
  }
  if (!std::isfinite(gradient_error) || gradient_error > 1.0e-2F) {
    std::cerr << "fused SSIM gradient mismatch exceeds tolerance\n";
    return 1;
  }

  std::cout << "fused_ssim_probe OK\n";
  return 0;
}
