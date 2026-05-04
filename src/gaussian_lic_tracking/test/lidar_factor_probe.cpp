// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_tracking/lidar_factor.hpp>

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
  gaussian_lic_tracking::LidarFactorConfig config;
  config.min_points = 16;
  config.max_frame_points = 256;
  config.max_map_points = 1024;
  config.nearest_distance_m = 0.5;
  config.correction_gain = 1.0;
  config.max_correction_m = 1.0;
  gaussian_lic_tracking::LidarFactor factor(config);

  std::vector<Eigen::Vector3d> scan;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      scan.emplace_back(0.2 * static_cast<double>(x), 0.2 * static_cast<double>(y), 1.0);
    }
  }

  gaussian_lic_tracking::TrajectoryPose identity;
  identity.stamp_ns = 0;
  identity.q_w_i = Eigen::Quaterniond::Identity();
  factor.insert_keyframe(scan, identity);
  if (factor.map_size() != scan.size()) {
    std::cerr << "LiDAR map insert size mismatch\n";
    return 1;
  }

  gaussian_lic_tracking::TrajectoryPose predicted;
  predicted.stamp_ns = 100000000LL;
  predicted.q_w_i = Eigen::Quaterniond::Identity();
  predicted.p_w_i = Eigen::Vector3d(0.05, 0.0, 0.0);
  const auto correction = factor.compute_translation_correction(scan, predicted);
  const double x_error = std::abs(correction.delta_p_w.x() + 0.05);
  const double yz_error = std::abs(correction.delta_p_w.y()) + std::abs(correction.delta_p_w.z());
  std::cout << "lidar_factor_probe matched=" << correction.matched_points
            << " mean_residual_m=" << correction.mean_residual_m
            << " delta=" << correction.delta_p_w.transpose()
            << " map_size=" << factor.map_size() << "\n";
  if (!correction.applied || correction.matched_points < config.min_points) {
    std::cerr << "LiDAR factor did not produce a correction\n";
    return 1;
  }
  if (x_error > 1.0e-12 || yz_error > 1.0e-12) {
    std::cerr << "LiDAR translation correction is wrong\n";
    return 1;
  }

  factor.clear();
  const auto empty_correction = factor.compute_translation_correction(scan, predicted);
  if (empty_correction.applied) {
    std::cerr << "LiDAR factor corrected without a map\n";
    return 1;
  }

  std::cout << "lidar_factor_probe OK\n";
  return 0;
}
