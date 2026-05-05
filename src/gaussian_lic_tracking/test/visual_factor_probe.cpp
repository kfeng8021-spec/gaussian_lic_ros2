// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_tracking/visual_factor.hpp>

#include <cmath>
#include <iostream>

int main()
{
  gaussian_lic_tracking::VisualFrame reference;
  reference.stamp_ns = 0;
  reference.width = 4;
  reference.height = 4;
  reference.gray.resize(16);
  for (size_t index = 0; index < reference.gray.size(); ++index) {
    reference.gray[index] = static_cast<float>(index) / 32.0F;
  }

  gaussian_lic_tracking::VisualFrame candidate = reference;
  gaussian_lic_tracking::VisualFactor factor(16);
  const auto identical = factor.evaluate(reference, candidate);
  if (!identical.valid || identical.rmse != 0.0 || identical.mean_abs_error != 0.0) {
    std::cerr << "identical visual frames should have zero residual\n";
    return 1;
  }

  for (auto & value : candidate.gray) {
    value += 0.1F;
  }
  const auto shifted = factor.evaluate(reference, candidate);
  std::cout << "visual_factor_probe compared=" << shifted.compared_pixels
            << " mae=" << shifted.mean_abs_error
            << " rmse=" << shifted.rmse << "\n";
  if (!shifted.valid) {
    std::cerr << "shifted visual residual is invalid\n";
    return 1;
  }
  if (std::abs(shifted.mean_abs_error - 0.1) > 1.0e-6 ||
    std::abs(shifted.rmse - 0.1) > 1.0e-6)
  {
    std::cerr << "shifted visual residual is wrong\n";
    return 1;
  }

  gaussian_lic_tracking::VisualFrame alignment_reference;
  alignment_reference.width = 8;
  alignment_reference.height = 8;
  alignment_reference.gray.assign(alignment_reference.width * alignment_reference.height, 0.0F);
  for (size_t y = 0; y < alignment_reference.height; ++y) {
    for (size_t x = 0; x < alignment_reference.width; ++x) {
      alignment_reference.gray[y * alignment_reference.width + x] =
        static_cast<float>(0.1 * static_cast<double>(x) + 0.03 * static_cast<double>(y));
    }
  }
  gaussian_lic_tracking::VisualFrame alignment_candidate = alignment_reference;
  for (size_t y = 0; y < alignment_reference.height; ++y) {
    for (size_t x = 0; x + 1 < alignment_reference.width; ++x) {
      alignment_candidate.gray[y * alignment_candidate.width + x + 1] =
        alignment_reference.gray[y * alignment_reference.width + x];
    }
  }
  const auto alignment = factor.estimate_translation(alignment_reference, alignment_candidate, 2);
  std::cout << " visual_alignment dx=" << alignment.dx
            << " dy=" << alignment.dy
            << " rmse=" << alignment.rmse;
  if (!alignment.valid || alignment.dx != 1 || alignment.dy != 0 || alignment.rmse > 1.0e-6) {
    std::cerr << "visual alignment failed to recover integer translation\n";
    return 1;
  }

  candidate.width = 2;
  if (factor.evaluate(reference, candidate).valid) {
    std::cerr << "shape-mismatched visual frames should be invalid\n";
    return 1;
  }

  std::cout << "visual_factor_probe OK\n";
  return 0;
}
