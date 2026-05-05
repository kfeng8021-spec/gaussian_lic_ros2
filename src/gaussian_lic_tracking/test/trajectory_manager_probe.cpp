// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_tracking/time.hpp>
#include <gaussian_lic_tracking/trajectory_manager.hpp>

#include <cmath>
#include <iostream>

int main()
{
  constexpr int64_t dt_ns = 50000000LL;
  gaussian_lic_tracking::TrajectoryManager trajectory(dt_ns);
  const Eigen::Vector3d velocity(2.0, -0.5, 0.25);

  for (int i = 0; i < 8; ++i) {
    const int64_t stamp_ns = static_cast<int64_t>(i) * dt_ns;
    gaussian_lic_tracking::TrajectoryPose pose;
    pose.stamp_ns = stamp_ns;
    pose.p_w_i = velocity * (static_cast<double>(stamp_ns) / 1.0e9);
    pose.q_w_i = Eigen::Quaterniond::Identity();
    trajectory.add_control_pose(pose);
  }

  double max_position_error = 0.0;
  double max_velocity_error = 0.0;
  for (int i = 2; i <= 8; ++i) {
    const int64_t stamp_ns = static_cast<int64_t>(i) * dt_ns / 2;
    gaussian_lic_tracking::TrajectoryPose query;
    if (!trajectory.query_pose(stamp_ns, query)) {
      std::cerr << "failed to query trajectory at " << stamp_ns << " ns\n";
      return 1;
    }
    const Eigen::Vector3d expected = velocity * (static_cast<double>(stamp_ns) / 1.0e9);
    max_position_error = std::max(max_position_error, (query.p_w_i - expected).norm());
    max_velocity_error = std::max(max_velocity_error, (query.v_w_i - velocity).norm());
    if (std::abs(query.q_w_i.w() - 1.0) > 1.0e-12) {
      std::cerr << "orientation drifted for constant-orientation trajectory\n";
      return 1;
    }
  }

  builtin_interfaces::msg::Time negative_stamp;
  negative_stamp.sec = -1;
  negative_stamp.nanosec = 500000000U;
  const int64_t negative_ns = gaussian_lic_tracking::stamp_to_nanoseconds(negative_stamp);
  const auto roundtrip = gaussian_lic_tracking::nanoseconds_to_stamp(negative_ns);
  if (negative_ns != -500000000LL || roundtrip.sec != -1 || roundtrip.nanosec != 500000000U) {
    std::cerr << "signed nanosecond ROS2 time roundtrip failed\n";
    return 1;
  }

  std::cout << "trajectory_manager_probe controls=" << trajectory.size()
            << " dt_ns=" << trajectory.control_interval_ns()
            << " max_position_error=" << max_position_error
            << " max_velocity_error=" << max_velocity_error
            << " negative_ns=" << negative_ns << "\n";
  if (max_position_error > 1.0e-9 || max_velocity_error > 1.0e-9) {
    std::cerr << "constant-velocity cubic B-spline query error is too large\n";
    return 1;
  }
  std::cout << "trajectory_manager_probe OK\n";
  return 0;
}
