// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cuda_runtime.h>

namespace gaussian_lic_mapping::cuda_ops
{

class SimpleKNN
{
public:
  static void knn(int point_count, float3 * points, float * mean_distances);
};

}  // namespace gaussian_lic_mapping::cuda_ops
