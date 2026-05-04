// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <stdexcept>

#include <builtin_interfaces/msg/time.hpp>

namespace gaussian_lic_tracking
{

constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

inline int64_t stamp_to_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.nanosec >= static_cast<uint32_t>(kNanosecondsPerSecond)) {
    throw std::runtime_error("ROS2 stamp nanosec field must be < 1e9");
  }
  return static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond +
    static_cast<int64_t>(stamp.nanosec);
}

inline builtin_interfaces::msg::Time nanoseconds_to_stamp(const int64_t stamp_ns)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(stamp_ns / kNanosecondsPerSecond);
  int64_t nanosec = stamp_ns % kNanosecondsPerSecond;
  if (nanosec < 0) {
    --stamp.sec;
    nanosec += kNanosecondsPerSecond;
  }
  stamp.nanosec = static_cast<uint32_t>(nanosec);
  return stamp;
}

}  // namespace gaussian_lic_tracking
