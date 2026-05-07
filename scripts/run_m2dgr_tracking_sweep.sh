#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_PATH="/home/frank/data/m2dgr/room_01_frontend_raw"
REFERENCE_TUM="/home/frank/data/m2dgr/room_01/room_01_gt.tum"
OUTPUT_ROOT="${ROOT_DIR}/results/m2dgr/room_01_tracking_sweep"
PLAYBACK_DURATION=30
PLAYBACK_RATE=0.2
TIMEOUT_SEC=90
POST_PLAY_SETTLE_SEC=8
MAX_RUNS=0
CONFIG_LIST=""

usage() {
  cat <<'EOF'
Usage: scripts/run_m2dgr_tracking_sweep.sh [OPTIONS]

Run a small M2DGR room_01 native-tracking parameter sweep against the local
ground-truth TUM. Each run keeps trajectory comparison non-gating so the sweep
can collect failed configurations and rank them by RMSE/path drift.

Options:
  --bag DIR                 Frontend-raw M2DGR rosbag2 directory.
  --reference-tum FILE      Ground-truth/reference TUM trajectory.
  --output-root DIR         Sweep output root.
  --playback-duration SEC   rosbag2 playback duration. Default: 30.
  --rate RATE               rosbag2 playback rate. Default: 0.2.
  --timeout SEC             Per-run timeout margin. Default: 90.
  --post-play-settle SEC    Per-run callback drain time. Default: 8.
  --max-runs N              Limit number of configs. Default: 0 for all.
  --configs LIST            Comma-separated config ids to run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      BAG_PATH="$2"
      shift 2
      ;;
    --reference-tum)
      REFERENCE_TUM="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
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
    --post-play-settle)
      POST_PLAY_SETTLE_SEC="$2"
      shift 2
      ;;
    --max-runs)
      MAX_RUNS="$2"
      shift 2
      ;;
    --configs)
      CONFIG_LIST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${BAG_PATH}/metadata.yaml" ]]; then
  echo "frontend-raw bag metadata not found: ${BAG_PATH}" >&2
  exit 2
fi
if [[ ! -s "${REFERENCE_TUM}" ]]; then
  echo "reference TUM is missing or empty: ${REFERENCE_TUM}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"

configs=(
  "all_frames_default"
  "keyframe_025_tighter_nn"
  "low_gain_tighter_step"
  "stronger_imu_pose"
)
if [[ -n "${CONFIG_LIST}" ]]; then
  IFS=',' read -r -a configs <<<"${CONFIG_LIST}"
fi

run_config() {
  local id="$1"
  local output="${OUTPUT_ROOT}/${id}_${PLAYBACK_DURATION}s"
  local extra=()

  case "${id}" in
    all_frames_default)
      extra+=(
        --lidar-keyframe-translation-m 0.0
        --lidar-nearest-distance-m 1.0
        --lidar-correction-gain 0.7
        --lidar-max-correction-m 0.25
        --tracking-max-pose-step-m 0.08
      )
      ;;
    keyframe_025_tighter_nn)
      extra+=(
        --lidar-keyframe-translation-m 0.25
        --lidar-nearest-distance-m 0.65
        --lidar-correction-gain 0.6
        --lidar-max-correction-m 0.20
        --tracking-max-pose-step-m 0.08
      )
      ;;
    low_gain_tighter_step)
      extra+=(
        --lidar-keyframe-translation-m 0.10
        --lidar-nearest-distance-m 0.55
        --lidar-correction-gain 0.35
        --lidar-max-correction-m 0.12
        --tracking-max-pose-step-m 0.08
        --sliding-window-max-translation-step-m 0.5
      )
      ;;
    stronger_imu_pose)
      extra+=(
        --lidar-keyframe-translation-m 0.20
        --lidar-nearest-distance-m 0.70
        --sliding-window-imu-position-weight 2.0
        --sliding-window-pose-translation-weight 4.0
        --sliding-window-pose-rotation-weight 4.0
        --sliding-window-smoothness-position-weight 0.3
        --sliding-window-smoothness-velocity-weight 0.3
        --tracking-max-pose-step-m 0.08
      )
      ;;
    *)
      echo "unknown sweep config: ${id}" >&2
      return 2
      ;;
  esac

  echo "[m2dgr-sweep] run ${id} -> ${output}"
  set +e
  "${ROOT_DIR}/scripts/run_native_tracking_bag_report.sh" \
    --bag "${BAG_PATH}" \
    --output "${output}" \
    --playback-duration "${PLAYBACK_DURATION}" \
    --rate "${PLAYBACK_RATE}" \
    --timeout "${TIMEOUT_SEC}" \
    --post-play-settle "${POST_PLAY_SETTLE_SEC}" \
    --reference-tum "${REFERENCE_TUM}" \
    --reference-trajectory-align yaw \
    --reference-max-association-dt 0.2 \
    --reference-min-coverage 0.0 \
    --reference-max-rmse-m 2.0 \
    --reference-max-mean-m 1.5 \
    --reference-max-error-m 5.0 \
    --reference-max-path-drift 10.0 \
    --min-poses 20 \
    --min-point-frames 10 \
    "${extra[@]}"
  local status=$?
  set -e
  echo "[m2dgr-sweep] ${id} exit=${status}"
  return 0
}

run_count=0
for id in "${configs[@]}"; do
  if [[ "${MAX_RUNS}" -gt 0 && "${run_count}" -ge "${MAX_RUNS}" ]]; then
    break
  fi
  run_config "${id}"
  run_count=$((run_count + 1))
done

python3 - "${OUTPUT_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for report_path in sorted(root.glob("*/native_tracking_report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    compare = report.get("trajectory_compare", {})
    translation = compare.get("translation", {})
    path_length = compare.get("path_length", {})
    rmse = translation.get("rmse_m")
    sort_rmse = rmse if isinstance(rmse, (int, float)) and math.isfinite(float(rmse)) else math.inf
    rows.append(
        {
            "id": report_path.parent.name,
            "ok": bool(report.get("ok", False)),
            "poses": metrics.get("trajectory_poses", 0),
            "points": metrics.get("topic_counts", {}).get("/points_for_gs", 0),
            "path_m": metrics.get("trajectory_path_length_m", 0.0),
            "rmse_m": rmse if sort_rmse != math.inf else None,
            "mean_m": translation.get("mean_m"),
            "max_m": translation.get("max_m"),
            "path_drift": path_length.get("relative_drift"),
            "matched_poses": compare.get("matched_poses"),
            "coverage": compare.get("coverage"),
            "report": str(report_path),
        }
    )
rows.sort(key=lambda item: (math.inf if item["rmse_m"] is None else item["rmse_m"]))
summary = {"runs": rows, "best": rows[0] if rows else None}
(root / "sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
