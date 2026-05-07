// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_tracking/imu_propagator.hpp>
#include <gaussian_lic_tracking/time.hpp>

#include <cmath>
#include <iostream>

int main()
{
  gaussian_lic_tracking::ImuPropagator propagator;
  gaussian_lic_tracking::ImuState initial;
  initial.stamp_ns = 0;
  propagator.reset(initial);

  constexpr int steps = 100;
  constexpr int64_t dt_ns = 10000000LL;
  const Eigen::Vector3d omega(0.0, 0.0, M_PI * 0.5);
  const Eigen::Vector3d zero_accel(0.0, 0.0, 0.0);
  for (int i = 1; i <= steps; ++i) {
    propagator.add_measurement(static_cast<int64_t>(i) * dt_ns, omega, zero_accel);
  }

  const auto yaw_state = propagator.state();
  const double yaw = Eigen::AngleAxisd(yaw_state.q_w_i).angle();
  const double yaw_error = std::abs(yaw - M_PI * 0.5);
  if (yaw_state.stamp_ns != steps * dt_ns) {
    std::cerr << "IMU yaw propagation stamp is wrong\n";
    return 1;
  }

  propagator.reset(initial);
  const Eigen::Vector3d zero_omega(0.0, 0.0, 0.0);
  const Eigen::Vector3d accel(1.0, 0.0, 0.0);
  for (int i = 1; i <= steps; ++i) {
    propagator.add_measurement(static_cast<int64_t>(i) * dt_ns, zero_omega, accel);
  }
  const auto accel_state = propagator.state();
  const double position_x_error = std::abs(accel_state.p_w_i.x() - 0.5);
  const double velocity_x_error = std::abs(accel_state.v_w_i.x() - 1.0);
  gaussian_lic_tracking::ImuState midpoint_state;
  if (!propagator.query_state(500000000LL, midpoint_state)) {
    std::cerr << "IMU history query failed\n";
    return 1;
  }
  const double midpoint_position_error = std::abs(midpoint_state.p_w_i.x() - 0.125);
  const double midpoint_velocity_error = std::abs(midpoint_state.v_w_i.x() - 0.5);

  std::cout << "imu_propagator_probe stamp_ns=" << accel_state.stamp_ns
            << " yaw_error=" << yaw_error
            << " position_x_error=" << position_x_error
            << " velocity_x_error=" << velocity_x_error
            << " midpoint_position_x_error=" << midpoint_position_error
            << " midpoint_velocity_x_error=" << midpoint_velocity_error
            << " history_size=" << propagator.history_size() << "\n";
  if (accel_state.stamp_ns != steps * dt_ns) {
    std::cerr << "IMU acceleration propagation stamp is wrong\n";
    return 1;
  }
  if (yaw_error > 1.0e-12 || position_x_error > 1.0e-12 || velocity_x_error > 1.0e-12 ||
    midpoint_position_error > 1.0e-12 || midpoint_velocity_error > 1.0e-12)
  {
    std::cerr << "IMU propagation drift exceeds tolerance\n";
    return 1;
  }

  propagator.set_gravity_w(Eigen::Vector3d{0.0, 0.0, -9.80665});
  gaussian_lic_tracking::ImuState gravity_initial;
  gravity_initial.stamp_ns = 0;
  propagator.reset_with_measurement(
    gravity_initial,
    zero_omega,
    Eigen::Vector3d{0.0, 0.0, 9.80665});
  for (int i = 1; i <= steps; ++i) {
    propagator.add_measurement(
      static_cast<int64_t>(i) * dt_ns,
      zero_omega,
      Eigen::Vector3d{0.0, 0.0, 9.80665});
  }
  const auto gravity_state = propagator.state();
  const double gravity_position_error = gravity_state.p_w_i.norm();
  const double gravity_velocity_error = gravity_state.v_w_i.norm();
  if (gravity_position_error > 1.0e-10 || gravity_velocity_error > 1.0e-10) {
    std::cerr << "IMU gravity compensation failed to keep a stationary sample fixed\n";
    return 1;
  }

  gaussian_lic_tracking::ImuState tilted_initial;
  tilted_initial.stamp_ns = 0;
  const Eigen::Quaterniond true_q_w_i =
    (Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitX()) *
    Eigen::AngleAxisd(-0.11, Eigen::Vector3d::UnitY())).normalized();
  const Eigen::Vector3d tilted_stationary_accel =
    true_q_w_i.inverse() * Eigen::Vector3d{0.0, 0.0, 9.80665};
  tilted_initial.q_w_i = Eigen::Quaterniond::FromTwoVectors(
    tilted_stationary_accel.normalized(),
    Eigen::Vector3d{0.0, 0.0, 1.0}).normalized();
  propagator.reset_with_measurement(tilted_initial, zero_omega, tilted_stationary_accel);
  for (int i = 1; i <= steps; ++i) {
    propagator.add_measurement(
      static_cast<int64_t>(i) * dt_ns,
      zero_omega,
      tilted_stationary_accel);
  }
  const auto tilted_gravity_state = propagator.state();
  if (tilted_gravity_state.p_w_i.norm() > 1.0e-10 ||
    tilted_gravity_state.v_w_i.norm() > 1.0e-10)
  {
    std::cerr << "IMU tilted gravity alignment failed to keep a stationary sample fixed\n";
    return 1;
  }

  propagator.set_gravity_w(Eigen::Vector3d::Zero());
  propagator.set_max_history_size(200);
  propagator.reset(initial);
  for (int i = 1; i <= steps; ++i) {
    propagator.add_measurement(static_cast<int64_t>(i) * dt_ns, zero_omega, accel);
  }
  gaussian_lic_tracking::ImuState corrected_midpoint;
  if (!propagator.query_state(500000000LL, corrected_midpoint)) {
    std::cerr << "IMU delayed feedback query failed\n";
    return 1;
  }
  corrected_midpoint.p_w_i.setZero();
  corrected_midpoint.v_w_i.setZero();
  if (!propagator.rebase_from_state(corrected_midpoint)) {
    std::cerr << "IMU delayed feedback rebase failed\n";
    return 1;
  }
  const auto rebased_state = propagator.state();
  const double rebased_position_x_error = std::abs(rebased_state.p_w_i.x() - 0.125);
  const double rebased_velocity_x_error = std::abs(rebased_state.v_w_i.x() - 0.5);
  if (rebased_position_x_error > 1.0e-12 || rebased_velocity_x_error > 1.0e-12 ||
    propagator.measurement_history_size() == 0U ||
    propagator.measurements_after(500000000LL).empty())
  {
    std::cerr << "IMU delayed feedback rebase did not replay future measurements\n";
    return 1;
  }

  std::cout << "imu_propagator_probe OK\n";
  return 0;
}
