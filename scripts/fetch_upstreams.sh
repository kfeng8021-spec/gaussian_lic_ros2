#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"

clone_or_fetch() {
    local name="$1"
    local url="$2"
    local dest="${EXTERNAL_DIR}/${name}"

    if [[ -d "${dest}/.git" ]]; then
        echo "[fetch] ${name} already exists; fetching tags and remote refs"
        git -C "${dest}" fetch --tags origin
    else
        echo "[fetch] cloning ${name}"
        git clone --recursive "${url}" "${dest}"
    fi
}

mkdir -p "${EXTERNAL_DIR}"
clone_or_fetch "Gaussian-LIC" "https://github.com/APRIL-ZJU/Gaussian-LIC.git"
clone_or_fetch "Coco-LIC" "https://github.com/APRIL-ZJU/Coco-LIC.git"

echo "[fetch] done"

