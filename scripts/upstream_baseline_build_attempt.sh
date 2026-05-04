#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GAUSSIAN_LIC_NOETIC_IMAGE:-gaussian_lic_noetic_baseline:latest}"
SOFTWARE_DIR="${GAUSSIAN_LIC_SOFTWARE_DIR:-/home/frank/Software}"
HOST_CUDA="${GAUSSIAN_LIC_HOST_CUDA:-/usr/local/cuda-12.8}"
CONTAINER_CUDA="${GAUSSIAN_LIC_CONTAINER_CUDA:-/usr/local/cuda-11.7}"
OPENCV_DIR="${GAUSSIAN_LIC_OPENCV_DIR:-${SOFTWARE_DIR}/opencv/opencv-4.7.0/build}"
WORKDIR_IN_CONTAINER="${GAUSSIAN_LIC_NOETIC_WORKDIR:-/tmp/catkin_gaussian}"
LOG_PATH="${ROOT_DIR}/log/noetic_gaussian_lic_build_attempt.log"
DOCKER_DNS="${GAUSSIAN_LIC_DOCKER_DNS:-8.8.8.8}"
PATCH_CFLOAT=true

usage() {
  cat <<'EOF'
Usage: scripts/upstream_baseline_build_attempt.sh [options]

Runs a reproducible ROS1 Noetic upstream Gaussian-LIC/Gaussian-LIC2 build attempt
inside a prepared Docker image. The script builds a copy under /tmp and leaves
the repository checkout untouched.

Options:
  --image NAME        Docker image. Default: gaussian_lic_noetic_baseline:latest
  --software DIR      Host third-party directory mounted as ~/Software.
                      Default: /home/frank/Software
  --host-cuda DIR     Host CUDA directory mounted as /usr/local/cuda-11.7.
                      Default: /usr/local/cuda-12.8
  --opencv-dir DIR    Host OpenCV build directory to inject into the copied
                      upstream CMakeLists.txt. Default:
                      /home/frank/Software/opencv/opencv-4.7.0/build
  --log PATH          Build log path. Default: log/noetic_gaussian_lic_build_attempt.log
  --no-cfloat-patch   Do not add <cfloat> to copied CUDA files before building.
  -h, --help          Show this help.

Environment:
  GAUSSIAN_LIC_DOCKER_DNS controls the docker --dns value. Set it empty to omit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --software)
      SOFTWARE_DIR="$2"
      shift 2
      ;;
    --host-cuda)
      HOST_CUDA="$2"
      shift 2
      ;;
    --opencv-dir)
      OPENCV_DIR="$2"
      shift 2
      ;;
    --log)
      LOG_PATH="$2"
      shift 2
      ;;
    --no-cfloat-patch)
      PATCH_CFLOAT=false
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

cd "${ROOT_DIR}"
LOG_PATH="$(realpath -m "${LOG_PATH}")"
case "${LOG_PATH}" in
  "${ROOT_DIR}"/*)
    LOG_IN_WORK="/work/${LOG_PATH#"${ROOT_DIR}/"}"
    ;;
  *)
    echo "Log path must be inside the repository so the container can write it: ${LOG_PATH}" >&2
    exit 2
    ;;
esac

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Missing Docker image: ${IMAGE}" >&2
  echo "Create the Noetic dependency image before running the upstream baseline build." >&2
  exit 2
fi
if [[ ! -d "${SOFTWARE_DIR}" ]]; then
  echo "Missing third-party directory: ${SOFTWARE_DIR}" >&2
  exit 2
fi
if [[ ! -d "${HOST_CUDA}" ]]; then
  echo "Missing host CUDA directory: ${HOST_CUDA}" >&2
  exit 2
fi
OPENCV_DIR="$(realpath -m "${OPENCV_DIR}")"
SOFTWARE_DIR="$(realpath -m "${SOFTWARE_DIR}")"
if [[ ! -d "${OPENCV_DIR}" ]]; then
  echo "Missing OpenCV build directory: ${OPENCV_DIR}" >&2
  exit 2
fi
case "${OPENCV_DIR}" in
  "${SOFTWARE_DIR}"/*)
    OPENCV_CONTAINER_DIR="/root/Software/${OPENCV_DIR#"${SOFTWARE_DIR}/"}"
    ;;
  *)
    echo "OpenCV directory must be under ${SOFTWARE_DIR} so the container can mount it: ${OPENCV_DIR}" >&2
    exit 2
    ;;
esac
if [[ ! -d "${ROOT_DIR}/external/Gaussian-LIC" ]]; then
  echo "Missing upstream checkout: ${ROOT_DIR}/external/Gaussian-LIC" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_PATH}")"

docker_args=(--rm)
if [[ -n "${DOCKER_DNS}" ]]; then
  docker_args+=(--dns "${DOCKER_DNS}")
fi
docker_args+=(
  -v "${ROOT_DIR}:/work"
  -v "${SOFTWARE_DIR}:/root/Software:ro"
  -v "${SOFTWARE_DIR}:/home/frank/Software:ro"
  -v "${HOST_CUDA}:${CONTAINER_CUDA}:ro"
  -w /work
)

patch_flag=0
if [[ "${PATCH_CFLOAT}" == "true" ]]; then
  patch_flag=1
fi

docker run "${docker_args[@]}" "${IMAGE}" bash -lc "set -euo pipefail
rm -rf '${WORKDIR_IN_CONTAINER}'
mkdir -p '${WORKDIR_IN_CONTAINER}/src'
cp -a /work/external/Gaussian-LIC '${WORKDIR_IN_CONTAINER}/src/Gaussian-LIC'
cd '${WORKDIR_IN_CONTAINER}/src/Gaussian-LIC'
sed -i 's|^set(OpenCV_DIR .*)|set(OpenCV_DIR ${OPENCV_CONTAINER_DIR})|' CMakeLists.txt
if [[ '${patch_flag}' == '1' ]]; then
  for file in \
    src/simple-knn/simple_knn.cu \
    src/rasterizer/cuda_rasterizer/rasterizer_impl.cu; do
    if ! grep -q '#include <cfloat>' \"\${file}\"; then
      sed -i '1i#include <cfloat>' \"\${file}\"
    fi
  done
fi
cd '${WORKDIR_IN_CONTAINER}'
source /opt/ros/noetic/setup.bash
export PATH=/usr/local/bin:${CONTAINER_CUDA}/bin:\${PATH}
export LD_LIBRARY_PATH=${CONTAINER_CUDA}/lib64:\${LD_LIBRARY_PATH:-}
catkin_make \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_TOOLKIT_ROOT_DIR=${CONTAINER_CUDA} \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  2>&1 | tee '${LOG_IN_WORK}'
"
