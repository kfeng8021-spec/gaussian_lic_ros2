#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse
import json
from pathlib import Path
import sys


DEFAULT_KEYS = (
    "tracking_hz",
    "mapping_hz",
    "mean_iteration_ms",
)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def lookup(metrics, key):
    value = metrics
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(key)
        value = value[part]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} is {type(value).__name__}, expected number")
    return float(value)


def is_latency_key(key):
    token = key.lower()
    return "latency" in token or "time" in token or token.endswith("_ms")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fail when Gaussian-LIC metrics regress.")
    parser.add_argument("--metrics", required=True, help="Current metrics.json")
    parser.add_argument("--baseline", required=True, help="Baseline metrics.json")
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="Metric key to compare. Dot paths are supported. Default: tracking_hz, mapping_hz, mean_iteration_ms.",
    )
    parser.add_argument("--max-regression", type=float, default=0.15, help="Allowed relative regression")
    args = parser.parse_args(argv)

    current = load_json(args.metrics)
    baseline = load_json(args.baseline)
    keys = tuple(args.keys) if args.keys else DEFAULT_KEYS

    failed = False
    for key in keys:
        try:
            cur = lookup(current, key)
            base = lookup(baseline, key)
        except (KeyError, TypeError) as exc:
            print(f"[perf] skip {key}: {exc}", file=sys.stderr)
            continue

        if base == 0:
            print(f"[perf] skip {key}: baseline is zero", file=sys.stderr)
            continue

        if is_latency_key(key):
            regression = (cur - base) / abs(base)
            ok = regression <= args.max_regression
        else:
            regression = (base - cur) / abs(base)
            ok = regression <= args.max_regression

        status = "OK" if ok else "FAIL"
        print(f"[perf] {key}: current={cur:.6g} baseline={base:.6g} regression={regression:.2%} {status}")
        failed = failed or not ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
