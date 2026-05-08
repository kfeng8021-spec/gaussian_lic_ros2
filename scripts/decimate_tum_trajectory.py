#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decimate a TUM trajectory by timestamp spacing."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input TUM trajectory.")
    parser.add_argument("--output", type=Path, required=True, help="Output decimated TUM trajectory.")
    parser.add_argument(
        "--min-dt",
        type=float,
        default=0.09,
        help="Minimum time spacing in seconds between retained poses. Default: 0.09.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_dt <= 0.0:
        raise SystemExit("--min-dt must be positive")
    if not args.input.is_file():
        raise SystemExit(f"missing input trajectory: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    last_time: float | None = None
    total = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 8:
                raise SystemExit(f"invalid TUM line {total + 1}: {stripped}")
            stamp = float(fields[0])
            total += 1
            if last_time is None or stamp - last_time >= args.min_dt:
                kept.append(stripped)
                last_time = stamp

    args.output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    print(f"decimated {args.input}: kept {len(kept)}/{total} poses at min_dt={args.min_dt:g}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
