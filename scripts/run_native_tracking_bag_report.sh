#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
BAG_PATH="/home/frank/data/fast_livo/Bright_Screen_Wall_frontend_raw"
OUTPUT_DIR="${ROOT_DIR}/results/fastlivo2/Bright_Screen_Wall_native_tracking_12s"
PLAYBACK_DURATION=12
PLAYBACK_RATE=1.0
TIMEOUT_SEC=30
MIN_POSES=20
MIN_STATUS_SAMPLES=1
MIN_POINT_FRAMES=10
LIDAR_MIN_POINTS=32
REQUIRE_BA_FEEDBACK=false
ENABLE_VISUAL_FACTORS=false

usage() {
  cat <<'EOF'
Usage: scripts/run_native_tracking_bag_report.sh [OPTIONS]

Run the native ROS2 tracking frontend against a frontend-raw rosbag2 dataset,
record tracking outputs, extract reproducibility artifacts, and gate tracking
health counters. This is a real-bag native tracking evidence path; it does not
replace the strict mapper-contract/CUDA paper gate.

Options:
  --bag DIR                    Frontend-raw rosbag2 directory.
  --output DIR                 Output report directory.
  --playback-duration SEC      rosbag2 playback duration. Default: 12.
  --rate RATE                  rosbag2 playback rate. Default: 1.0.
  --timeout SEC                Topic/report timeout. Default: 30.
  --min-poses N                Minimum recorded frontend odometry poses. Default: 20.
  --min-status-samples N       Minimum TrackingStatus samples. Default: 1.
  --min-point-frames N         Minimum recorded /points_for_gs frames. Default: 10.
  --lidar-min-points N         Tracking LiDAR frame minimum. Default: 32.
  --require-ba-feedback        Require accepted sliding-window feedback.
  --enable-visual-factors      Require mapper-rendered-image visual factors to be present externally.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      BAG_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --playback-duration)
      PLAYBACK_DURATION="$2"
      shift 2
      ;;
    --rate)
      PLAYBACK_RATE="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --min-poses)
      MIN_POSES="$2"
      shift 2
      ;;
    --min-status-samples)
      MIN_STATUS_SAMPLES="$2"
      shift 2
      ;;
    --min-point-frames)
      MIN_POINT_FRAMES="$2"
      shift 2
      ;;
    --lidar-min-points)
      LIDAR_MIN_POINTS="$2"
      shift 2
      ;;
    --require-ba-feedback)
      REQUIRE_BA_FEEDBACK=true
      shift
      ;;
    --enable-visual-factors)
      ENABLE_VISUAL_FACTORS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${BAG_PATH}/metadata.yaml" ]]; then
  echo "frontend-raw bag metadata not found: ${BAG_PATH}" >&2
  exit 2
fi

if [[ -n "${GAUSSIAN_LIC_NATIVE_TRACKING_ROS_DOMAIN_ID:-}" ]]; then
  export ROS_DOMAIN_ID="${GAUSSIAN_LIC_NATIVE_TRACKING_ROS_DOMAIN_ID}"
elif [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  export ROS_DOMAIN_ID="$((170 + (BASHPID % 40)))"
fi

cd "${ROOT_DIR}"
set +u
source "/opt/ros/${ROS_DISTRO}/setup.bash"
source install/setup.bash
set -u

OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
ARTIFACT_DIR="${OUTPUT_DIR}/artifacts"
REPORT_JSON="${OUTPUT_DIR}/native_tracking_report.json"
LOG_DIR="${OUTPUT_DIR}/logs"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

launch_log="${LOG_DIR}/tracking_launch.log"
play_log="${LOG_DIR}/rosbag_play.log"
recorder_log="${LOG_DIR}/native_tracking_recorder.log"

stop_process_group() {
  local pid="$1"
  local signal_name="$2"
  if [[ -z "${pid}" ]]; then
    return
  fi
  kill "-${signal_name}" "${pid}" 2>/dev/null || true
  kill "-${signal_name}" -- -"${pid}" 2>/dev/null || true
  for _ in $(seq 1 200); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return
    fi
    sleep 0.1
  done
  kill -KILL -- -"${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  if [[ -n "${play_pid:-}" ]]; then
    kill "${play_pid}" 2>/dev/null || true
    wait "${play_pid}" 2>/dev/null || true
  fi
  if [[ -n "${record_pid:-}" ]]; then
    stop_process_group "${record_pid}" TERM
  fi
  if [[ -n "${launch_pid:-}" ]]; then
    stop_process_group "${launch_pid}" TERM
  fi
}
trap cleanup EXIT

setsid ros2 launch gaussian_lic_bringup tracking.launch.py \
  enable_sliding_window_optimizer:=true \
  enable_lidar_plane_factor:=true \
  enable_visual_factor:="${ENABLE_VISUAL_FACTORS}" \
  enable_visual_alignment_window_factor:="${ENABLE_VISUAL_FACTORS}" \
  enable_se3_photometric_window_factor:="${ENABLE_VISUAL_FACTORS}" \
  lidar_min_points:="${LIDAR_MIN_POINTS}" \
  lidar_keyframe_translation_m:=0.0 \
  lidar_nearest_distance_m:=1.0 \
  lidar_max_frame_points:=4000 \
  lidar_max_map_points:=40000 \
  sliding_window_max_iterations:=3 \
  sliding_window_max_normal_equation_condition:=10000000000000.0 \
  >"${launch_log}" 2>&1 &
launch_pid=$!

sleep 2

setsid ros2 run gaussian_lic_tools native_tracking_recorder \
  --ros-args \
  -p output_dir:="${ARTIFACT_DIR}" \
  -p odometry_topic:=/gaussian_lic/frontend/odometry \
  -p pointcloud_topic:=/points_for_gs \
  -p status_topic:=/gaussian_lic/frontend/status \
  >"${recorder_log}" 2>&1 &
record_pid=$!

sleep 2

ros2 bag play "${BAG_PATH}" \
  --clock \
  --rate "${PLAYBACK_RATE}" \
  --read-ahead-queue-size 100 \
  --playback-duration "${PLAYBACK_DURATION}" \
  --disable-keyboard-controls \
  >"${play_log}" 2>&1 &
play_pid=$!

play_timeout="$(
  python3 - "${PLAYBACK_DURATION}" "${TIMEOUT_SEC}" <<'PY'
import math
import sys
print(int(math.ceil(float(sys.argv[1]) + float(sys.argv[2]))))
PY
)"
if ! timeout "${play_timeout}" tail --pid="${play_pid}" -f /dev/null; then
  echo "rosbag2 playback did not finish before timeout" >&2
  exit 1
fi
wait "${play_pid}"
unset play_pid

sleep 2
stop_process_group "${record_pid}" TERM
unset record_pid

stop_process_group "${launch_pid}" TERM
unset launch_pid

python3 - "${ARTIFACT_DIR}/metrics.json" "${REPORT_JSON}" \
  "${MIN_POSES}" "${MIN_STATUS_SAMPLES}" "${MIN_POINT_FRAMES}" "${REQUIRE_BA_FEEDBACK}" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
min_poses = int(sys.argv[3])
min_status = int(sys.argv[4])
min_point_frames = int(sys.argv[5])
require_ba_feedback = sys.argv[6].lower() == "true"

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
topic_counts = metrics.get("topic_counts", {})
status = metrics.get("tracking_status", {})
last = status.get("last") or {}
errors = []

if metrics.get("trajectory_poses", 0) < min_poses:
    errors.append(f"trajectory poses {metrics.get('trajectory_poses', 0)} < {min_poses}")
if status.get("samples", 0) < min_status:
    errors.append(f"tracking status samples {status.get('samples', 0)} < {min_status}")
if topic_counts.get("/points_for_gs", 0) < min_point_frames:
    errors.append(f"/points_for_gs frames {topic_counts.get('/points_for_gs', 0)} < {min_point_frames}")

for key in (
    "image_stamp_regressions",
    "pointcloud_stamp_regressions",
    "imu_stamp_regressions",
    "imu_invalid_measurements",
    "lidar_invalid_frames",
    "lidar_invalid_points",
    "lidar_invalid_point_times",
    "lidar_out_of_range_point_times",
    "sliding_window_invalid_candidate_steps",
    "sliding_window_linearization_failure_count",
    "sliding_window_linear_solve_failure_count",
    "sliding_window_invalid_optimized_states",
    "sliding_window_numeric_jacobian_blocks",
    "sliding_window_numeric_jacobian_columns",
    "sliding_window_orphan_factors",
):
    if int(last.get(key, 0)) != 0:
        errors.append(f"{key} is {last.get(key)}")

if int(last.get("sliding_window_normal_equation_rows", 0)) <= 0:
    errors.append("sliding_window_normal_equation_rows is zero")
if int(last.get("sliding_window_total_imu_factors", 0)) <= 0:
    errors.append("sliding_window_total_imu_factors is zero")
if int(last.get("sliding_window_point_factors", 0)) <= 0:
    errors.append("sliding_window_point_factors is zero")
if int(last.get("sliding_window_smoothness_factors", 0)) <= 0:
    errors.append("sliding_window_smoothness_factors is zero")
if require_ba_feedback and int(last.get("sliding_window_feedback_updates", 0)) <= 0:
    errors.append("sliding_window_feedback_updates is zero")
if require_ba_feedback and int(last.get("sliding_window_accepted_steps", 0)) <= 0:
    errors.append("sliding_window_accepted_steps is zero")

report = {
    "ok": not errors,
    "errors": errors,
    "metrics": metrics,
}
report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)
print(
    "native tracking report OK: "
    f"poses={metrics.get('trajectory_poses', 0)} "
    f"points_frames={topic_counts.get('/points_for_gs', 0)} "
    f"status_samples={status.get('samples', 0)} "
    f"imu_factors={last.get('sliding_window_total_imu_factors', 0)}"
)
PY

echo "[native-tracking] report: ${REPORT_JSON}"
