// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <torch/torch.h>

namespace gaussian_lic_mapping::cuda_ops
{

torch::Tensor dist_cuda2(const torch::Tensor & points);

}  // namespace gaussian_lic_mapping::cuda_ops
