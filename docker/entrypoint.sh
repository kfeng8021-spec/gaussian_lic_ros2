#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

workspace="${GAUSSIAN_LIC_WS:-/ws}"

set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "${workspace}/install/setup.bash" ]]; then
  source "${workspace}/install/setup.bash"
fi
set -u

cd "${workspace}"
exec "$@"
