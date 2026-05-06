#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Normalize tabular reference trajectories to TUM format.

Accepted input rows:
  - TUM:              t x y z qx qy qz qw
  - MCD pose_inW.csv: num t x y z qx qy qz qw

Non-numeric headers and comment lines are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_pose_row(line: str, line_number: int) -> tuple[float, ...] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = text.replace(",", " ").split()
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return None
    if len(values) == 8:
        return values
    if len(values) == 9:
        return values[1:]
    raise ValueError(f"line {line_number}: expected 8 or 9 numeric columns, got {len(values)}")


def convert(input_path: Path, output_path: Path) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line_number, line in enumerate(source, start=1):
            pose = parse_pose_row(line, line_number)
            if pose is None:
                continue
            sink.write(
                f"{pose[0]:.9f} {pose[1]:.12g} {pose[2]:.12g} {pose[3]:.12g} "
                f"{pose[4]:.12g} {pose[5]:.12g} {pose[6]:.12g} {pose[7]:.12g}\n"
            )
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-poses", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        count = convert(args.input.expanduser().resolve(), args.output.expanduser().resolve())
    except Exception as exc:  # noqa: BLE001 - report malformed data uniformly.
        print(f"trajectory conversion failed: {exc}", file=sys.stderr)
        return 2
    if count < args.min_poses:
        print(f"trajectory conversion produced {count} poses < min_poses {args.min_poses}", file=sys.stderr)
        return 3
    print({"input": str(args.input), "output": str(args.output), "poses": count})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
