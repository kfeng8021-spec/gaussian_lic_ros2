// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_tracking/sliding_window_optimizer.hpp>

#include <cmath>
#include <iostream>

int main()
{
  gaussian_lic_tracking::SlidingWindowConfig config;
  config.max_iterations = 4;
  gaussian_lic_tracking::SlidingWindowOptimizer optimizer(config);

  gaussian_lic_tracking::SlidingWindowState state;
  state.stamp_ns = 100;
  state.p_w_i = Eigen::Vector3d{0.0, 0.0, 0.0};
  optimizer.add_or_update_state(state);

  gaussian_lic_tracking::SlidingWindowVisualAlignmentFactor factor;
  factor.stamp_ns = state.stamp_ns;
  factor.reference_p_w_i = state.p_w_i;
  factor.measured_shift_px = Eigen::Vector2d{12.0, -7.0};
  factor.meters_per_pixel = 0.01;
  factor.weight = 50.0;
  optimizer.add_visual_alignment_factor(factor);

  const auto summary = optimizer.optimize();
  gaussian_lic_tracking::SlidingWindowState optimized;
  if (!optimizer.get_state(state.stamp_ns, optimized)) {
    std::cerr << "optimized visual-factor state is missing\n";
    return 1;
  }

  const Eigen::Vector3d expected{0.12, -0.07, 0.0};
  const double xy_error = (optimized.p_w_i.head<2>() - expected.head<2>()).norm();
  std::cout << "sliding_window_visual_factor_probe visual_factors="
            << summary.visual_factor_count
            << " initial_cost=" << summary.initial_cost
            << " final_cost=" << summary.final_cost
            << " xy_error=" << xy_error
            << " optimized_p=" << optimized.p_w_i.transpose() << "\n";

  if (!summary.converged || summary.visual_factor_count != 1U ||
    summary.final_cost >= summary.initial_cost || xy_error > 1.0e-8)
  {
    std::cerr << "sliding window visual alignment factor failed to recover planar correction\n";
    return 1;
  }
  std::cout << "sliding_window_visual_factor_probe OK\n";
  return 0;
}
