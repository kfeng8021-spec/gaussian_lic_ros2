// SPDX-License-Identifier: GPL-3.0-or-later

#include "gaussian_lic_mapping/cuda/simple_knn.hpp"

#include "simple_knn.hpp"

#include <c10/cuda/CUDAException.h>

namespace gaussian_lic_mapping::cuda_ops
{

torch::Tensor dist_cuda2(const torch::Tensor & points)
{
  TORCH_CHECK(points.is_cuda(), "dist_cuda2 expects a CUDA tensor");
  TORCH_CHECK(points.dtype() == torch::kFloat32, "dist_cuda2 expects float32 points");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "dist_cuda2 expects shape [N, 3]");
  TORCH_CHECK(points.size(0) >= 4, "dist_cuda2 requires at least four points");

  const auto contiguous_points = points.contiguous();
  const int point_count = static_cast<int>(contiguous_points.size(0));
  auto mean_distances = torch::empty({point_count}, contiguous_points.options().dtype(torch::kFloat32));

  SimpleKNN::knn(
    point_count,
    reinterpret_cast<float3 *>(contiguous_points.data_ptr<float>()),
    mean_distances.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return mean_distances;
}

}  // namespace gaussian_lic_mapping::cuda_ops
