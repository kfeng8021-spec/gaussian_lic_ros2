// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <torch/torch.h>

#include <tuple>

namespace gaussian_lic_mapping::cuda_ops
{

constexpr float kFusedSsimC1 = 0.0001F;
constexpr float kFusedSsimC2 = 0.0009F;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_ssim_forward_map(
  float c1,
  float c2,
  const torch::Tensor & image1,
  const torch::Tensor & image2,
  bool train);

torch::Tensor fused_ssim_backward_map(
  float c1,
  float c2,
  const torch::Tensor & image1,
  const torch::Tensor & image2,
  const torch::Tensor & grad_map,
  const torch::Tensor & dmap_dmu1,
  const torch::Tensor & dmap_dsigma1_sq,
  const torch::Tensor & dmap_dsigma12);

torch::Tensor fused_ssim_map(const torch::Tensor & image1, const torch::Tensor & image2);

torch::Tensor fused_ssim(const torch::Tensor & image1, const torch::Tensor & image2);

}  // namespace gaussian_lic_mapping::cuda_ops
