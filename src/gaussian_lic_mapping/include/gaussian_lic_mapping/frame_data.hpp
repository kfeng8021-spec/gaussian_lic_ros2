// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace gaussian_lic_mapping
{

struct AlignedRosFrame
{
  sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr pose;
  sensor_msgs::msg::Image::ConstSharedPtr image;
  sensor_msgs::msg::Image::ConstSharedPtr depth;
};

struct MapperPoint
{
  Eigen::Vector3f xyz_world{Eigen::Vector3f::Zero()};
  Eigen::Vector3f color_rgb{Eigen::Vector3f::Ones()};
  float depth_m{0.0F};
};

struct MapperFrameData
{
  builtin_interfaces::msg::Time stamp;
  uint64_t frame_index{0};
  bool is_keyframe{false};
  int width{0};
  int height{0};
  cv::Mat image_rgb_float;
  cv::Mat depth_m_float;
  Eigen::Quaterniond q_wc{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d t_wc{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d r_wc{Eigen::Matrix3d::Identity()};
  std::vector<MapperPoint> points;
  size_t skipped_points_nonpositive_depth{0};
};

MapperFrameData convert_aligned_frame(
  const AlignedRosFrame & frame,
  uint64_t frame_index,
  int select_every_k_frame);

}  // namespace gaussian_lic_mapping
