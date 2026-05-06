#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/fetch_sharepoint_file.sh --url URL --output FILE [OPTIONS]

Fetch a public SharePoint/OneDrive file with browser-like cookies and resumable
curl output. This is used for M2DGR and other strict-parity datasets whose
official links redirect through SharePoint short URLs.

Options:
  --url URL             SharePoint sharing URL. Required.
  --output FILE         Final output file. Required.
  --expected-bytes N    Optional exact byte count gate after download.
  --min-free-gb N       Require this much free space at FILE's directory.
                        Default: 5.
  --cookie FILE         Cookie jar. Default: /tmp/sharepoint_fetch_cookies.txt.
  --help                Show this help.
EOF
}

url=""
output=""
expected_bytes=""
min_free_gb=5
cookie="/tmp/sharepoint_fetch_cookies.txt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      url="$2"
      shift 2
      ;;
    --output)
      output="$2"
      shift 2
      ;;
    --expected-bytes)
      expected_bytes="$2"
      shift 2
      ;;
    --min-free-gb)
      min_free_gb="$2"
      shift 2
      ;;
    --cookie)
      cookie="$2"
      shift 2
      ;;
    --help|-h)
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

if [[ -z "$url" || -z "$output" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"
touch "$cookie"

free_kb="$(df -Pk "$(dirname "$output")" | awk 'NR==2 {print $4}')"
need_kb="$(python3 - <<PY
print(int(float("$min_free_gb") * 1024 * 1024))
PY
)"
if (( free_kb < need_kb )); then
  echo "insufficient free space at $(dirname "$output"): ${free_kb} KiB < ${need_kb} KiB" >&2
  exit 3
fi

tmp="${output}.part"
download_url="$url"
case "$download_url" in
  *\?*) download_url="${download_url}&download=1" ;;
  *) download_url="${download_url}?download=1" ;;
esac

curl --http1.1 \
  -L \
  --fail \
  --retry 50 \
  --retry-delay 15 \
  --retry-all-errors \
  --connect-timeout 30 \
  -C - \
  -A "Mozilla/5.0" \
  -c "$cookie" \
  -b "$cookie" \
  "$download_url" \
  -o "$tmp"

mv "$tmp" "$output"

if [[ -n "$expected_bytes" ]]; then
  actual_bytes="$(stat -c '%s' "$output")"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "byte-count mismatch for $output: got $actual_bytes expected $expected_bytes" >&2
    exit 4
  fi
fi

python3 - <<PY
from pathlib import Path
p = Path("$output")
print({"path": str(p), "bytes": p.stat().st_size})
PY
