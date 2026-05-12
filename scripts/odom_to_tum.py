#!/usr/bin/env python3
"""Subscribes to a `nav_msgs/Odometry` topic and writes each message to a
TUM-format trajectory file: `stamp_s tx ty tz qx qy qz qw`.

Used by the continuous-time native reference parity smoke as a robust
alternative to the in-node TUM writer when poses can take extreme magnitudes.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class TumLogger(Node):
    def __init__(self, topic: str, path: str, position_bound: float) -> None:
        super().__init__("odom_to_tum")
        self.fp = open(path, "w", buffering=1)
        self.fp.write("# stamp_s tx ty tz qx qy qz qw\n")
        self.position_bound = float(position_bound)
        self.received = 0
        self.written = 0
        self.create_subscription(Odometry, topic, self._on, 200)

    def _on(self, msg: Odometry) -> None:
        self.received += 1
        stamp_s = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w):
            if not math.isfinite(v):
                return
        if any(abs(v) > self.position_bound for v in (p.x, p.y, p.z)):
            return
        self.fp.write(
            f"{stamp_s:.9f} {p.x:.6f} {p.y:.6f} {p.z:.6f} "
            f"{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n"
        )
        self.written += 1

    def close(self) -> None:
        try:
            self.fp.flush()
            os.fsync(self.fp.fileno())
            self.fp.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/continuous_time/odometry")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--duration", type=float, required=True,
        help="Capture duration in wall seconds.",
    )
    parser.add_argument("--position-bound", type=float, default=1.0e6)
    args = parser.parse_args()

    rclpy.init()
    logger = TumLogger(args.topic, args.output, args.position_bound)
    deadline = time.time() + args.duration

    def _on_sigterm(_sig, _frame):
        logger.close()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    while time.time() < deadline:
        rclpy.spin_once(logger, timeout_sec=0.1)

    print(f"odom_to_tum: received={logger.received} written={logger.written}")
    logger.close()
    logger.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
