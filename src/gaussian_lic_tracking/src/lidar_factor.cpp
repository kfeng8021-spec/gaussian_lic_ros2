// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_tracking/lidar_factor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gaussian_lic_tracking
{

LidarFactor::LidarFactor(LidarFactorConfig config)
{
  set_config(config);
}

void LidarFactor::set_config(const LidarFactorConfig & config)
{
  if (config.min_points == 0U) {
    throw std::runtime_error("LiDAR factor min_points must be positive");
  }
  if (config.nearest_distance_m <= 0.0) {
    throw std::runtime_error("LiDAR factor nearest distance must be positive");
  }
  if (config.correction_gain < 0.0 || config.correction_gain > 1.0) {
    throw std::runtime_error("LiDAR factor correction_gain must be in [0, 1]");
  }
  if (config.max_correction_m <= 0.0) {
    throw std::runtime_error("LiDAR factor max correction must be positive");
  }
  config_ = config;
}

void LidarFactor::clear()
{
  map_points_w_.clear();
}

std::vector<Eigen::Vector3d> LidarFactor::sample_points(
  const std::vector<Eigen::Vector3d> & points,
  const size_t max_points)
{
  if (max_points == 0U || points.size() <= max_points) {
    return points;
  }
  const size_t stride = static_cast<size_t>(
    std::ceil(static_cast<double>(points.size()) / static_cast<double>(max_points)));
  std::vector<Eigen::Vector3d> sampled;
  sampled.reserve(max_points);
  for (size_t index = 0; index < points.size() && sampled.size() < max_points; index += stride) {
    sampled.push_back(points[index]);
  }
  return sampled;
}

LidarCorrection LidarFactor::compute_translation_correction(
  const std::vector<Eigen::Vector3d> & frame_points_i,
  const TrajectoryPose & predicted_pose) const
{
  LidarCorrection correction;
  if (frame_points_i.size() < config_.min_points || map_points_w_.size() < config_.min_points) {
    return correction;
  }

  const auto sampled = sample_points(frame_points_i, config_.max_frame_points);
  const double max_distance_sq = config_.nearest_distance_m * config_.nearest_distance_m;
  Eigen::Vector3d residual_sum = Eigen::Vector3d::Zero();
  double residual_norm_sum = 0.0;
  size_t matches = 0U;

  for (const auto & point_i : sampled) {
    const Eigen::Vector3d point_w = predicted_pose.q_w_i * point_i + predicted_pose.p_w_i;
    double best_distance_sq = std::numeric_limits<double>::infinity();
    Eigen::Vector3d best_point_w = Eigen::Vector3d::Zero();
    for (const auto & map_point_w : map_points_w_) {
      const double distance_sq = (map_point_w - point_w).squaredNorm();
      if (distance_sq < best_distance_sq) {
        best_distance_sq = distance_sq;
        best_point_w = map_point_w;
      }
    }
    if (best_distance_sq <= max_distance_sq) {
      const Eigen::Vector3d residual = best_point_w - point_w;
      residual_sum += residual;
      residual_norm_sum += std::sqrt(best_distance_sq);
      ++matches;
    }
  }

  if (matches < config_.min_points) {
    return correction;
  }

  correction.applied = true;
  correction.matched_points = matches;
  correction.mean_residual_m = residual_norm_sum / static_cast<double>(matches);
  correction.delta_p_w = config_.correction_gain * residual_sum / static_cast<double>(matches);
  const double norm = correction.delta_p_w.norm();
  if (norm > config_.max_correction_m) {
    correction.delta_p_w *= config_.max_correction_m / norm;
  }
  return correction;
}

void LidarFactor::insert_keyframe(
  const std::vector<Eigen::Vector3d> & frame_points_i,
  const TrajectoryPose & pose)
{
  if (frame_points_i.size() < config_.min_points) {
    return;
  }
  const auto sampled = sample_points(frame_points_i, config_.max_frame_points);
  map_points_w_.reserve(std::min(config_.max_map_points, map_points_w_.size() + sampled.size()));
  for (const auto & point_i : sampled) {
    map_points_w_.push_back(pose.q_w_i * point_i + pose.p_w_i);
  }
  if (config_.max_map_points > 0U && map_points_w_.size() > config_.max_map_points) {
    const size_t overflow = map_points_w_.size() - config_.max_map_points;
    map_points_w_.erase(map_points_w_.begin(), map_points_w_.begin() + static_cast<std::ptrdiff_t>(overflow));
  }
}

}  // namespace gaussian_lic_tracking
