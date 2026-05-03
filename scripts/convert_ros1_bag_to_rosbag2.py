#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a ROS1 .bag into rosbag2/mcap for Gaussian-LIC reproduction."
    )
    parser.add_argument("--input", required=True, help="Input ROS1 .bag")
    parser.add_argument("--output", required=True, help="Output rosbag2 directory")
    parser.add_argument(
        "--storage",
        default="mcap",
        choices=("mcap", "sqlite3"),
        help="rosbag2 storage backend to request when the converter backend supports it",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="Topic to include. Repeat for multiple topics. Default: all supported topics.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    if not input_path.exists():
        print(f"input bag does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        import rosbags  # noqa: F401
    except ImportError:
        print(
            "ROS1 bag conversion requires the optional Python package 'rosbags'.\n"
            "Install it in the conversion environment, then rerun this command.\n"
            "This script is intentionally dependency-gated so the Jazzy runtime path "
            "does not depend on ROS1 or rosbags.",
            file=sys.stderr,
        )
        return 3

    print(
        "The converter backend dependency is present, but message-level ROS1->ROS2 "
        "type mapping is not implemented yet. M1 tracks this as the standalone bag "
        "converter package deliverable.\n"
        f"input={input_path}\noutput={output_path}\nstorage={args.storage}\n"
        f"topics={args.topics or 'all'}",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
