// SPDX-License-Identifier: GPL-3.0-or-later

#include "gaussian_lic_mapping/cuda/simple_knn.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{

struct Point
{
  float x;
  float y;
  float z;
};

float squared_distance(const Point & lhs, const Point & rhs)
{
  const float dx = lhs.x - rhs.x;
  const float dy = lhs.y - rhs.y;
  const float dz = lhs.z - rhs.z;
  return dx * dx + dy * dy + dz * dz;
}

std::vector<float> reference_distances(const std::vector<Point> & points)
{
  std::vector<float> output(points.size(), 0.0F);
  for (size_t i = 0; i < points.size(); ++i) {
    float best[3] = {
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity()};
    for (size_t j = 0; j < points.size(); ++j) {
      if (i == j) {
        continue;
      }
      float distance = squared_distance(points[i], points[j]);
      for (float & candidate : best) {
        if (candidate > distance) {
          std::swap(candidate, distance);
        }
      }
    }
    output[i] = (best[0] + best[1] + best[2]) / 3.0F;
  }
  return output;
}

}  // namespace

int main()
{
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available\n";
    return 2;
  }

  constexpr int point_count = 5000;
  torch::manual_seed(20260504);
  auto cpu_points = torch::rand({point_count, 3}, torch::TensorOptions().dtype(torch::kFloat32));
  auto cuda_points = cpu_points.to(torch::kCUDA);
  auto cuda_distances = gaussian_lic_mapping::cuda_ops::dist_cuda2(cuda_points);
  auto cpu_distances = cuda_distances.to(torch::kCPU).contiguous();

  const auto point_accessor = cpu_points.accessor<float, 2>();
  std::vector<Point> points;
  points.reserve(point_count);
  for (int i = 0; i < point_count; ++i) {
    points.push_back(Point{point_accessor[i][0], point_accessor[i][1], point_accessor[i][2]});
  }
  const auto reference = reference_distances(points);

  const auto distance_accessor = cpu_distances.accessor<float, 1>();
  float max_abs_error = 0.0F;
  float mean_abs_error = 0.0F;
  for (int i = 0; i < point_count; ++i) {
    const float error = std::abs(distance_accessor[i] - reference[static_cast<size_t>(i)]);
    max_abs_error = std::max(max_abs_error, error);
    mean_abs_error += error;
  }
  mean_abs_error /= static_cast<float>(point_count);

  std::cout << "simple_knn_probe points=" << point_count
            << " max_abs_error=" << max_abs_error
            << " mean_abs_error=" << mean_abs_error << "\n";
  if (!std::isfinite(max_abs_error) || max_abs_error > 1.0e-5F) {
    std::cerr << "dist_cuda2 mismatch exceeds tolerance\n";
    return 1;
  }
  std::cout << "simple_knn_probe OK\n";
  return 0;
}
