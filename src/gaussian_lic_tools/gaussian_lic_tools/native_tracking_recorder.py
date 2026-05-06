# SPDX-License-Identifier: GPL-3.0-or-later

import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from gaussian_lic_msgs.msg import TrackingStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


def stamp_to_nsec(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_to_tum(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def finite_float(value):
    numeric = float(value)
    return numeric if math.isfinite(numeric) else 0.0


def tracking_status_to_dict(msg):
    output = {}
    for name in msg.get_fields_and_field_types():
        value = getattr(msg, name)
        if isinstance(value, (bool, int, float, str)):
            output[name] = value
        else:
            try:
                output[name] = list(value)
            except TypeError:
                output[name] = str(value)
    return output


class NativeTrackingRecorder(Node):
    def __init__(self):
        super().__init__("native_tracking_recorder")
        self.output_dir = Path(self.declare_parameter("output_dir", "/tmp/gaussian_lic_native_tracking").value)
        self.odometry_topic = self.declare_parameter(
            "odometry_topic", "/gaussian_lic/frontend/odometry").value
        self.pointcloud_topic = self.declare_parameter("pointcloud_topic", "/points_for_gs").value
        self.status_topic = self.declare_parameter(
            "status_topic", "/gaussian_lic/frontend/status").value
        self.output_dir.mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(
            depth=50,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.poses = []
        self.topic_counts = {
            self.odometry_topic: 0,
            self.pointcloud_topic: 0,
            self.status_topic: 0,
        }
        self.first_stamp_ns = None
        self.last_stamp_ns = None
        self.last_status = {}

        self.create_subscription(Odometry, self.odometry_topic, self.on_odometry, qos)
        self.create_subscription(PointCloud2, self.pointcloud_topic, self.on_pointcloud, qos)
        self.create_subscription(TrackingStatus, self.status_topic, self.on_status, qos)
        self.create_timer(1.0, self.flush)
        self.get_logger().info(
            "Recording native tracking evidence to "
            f"{self.output_dir} from {self.odometry_topic}, {self.pointcloud_topic}, {self.status_topic}")

    def update_span(self, stamp):
        stamp_ns = stamp_to_nsec(stamp)
        if self.first_stamp_ns is None or stamp_ns < self.first_stamp_ns:
            self.first_stamp_ns = stamp_ns
        if self.last_stamp_ns is None or stamp_ns > self.last_stamp_ns:
            self.last_stamp_ns = stamp_ns

    def on_odometry(self, msg):
        self.topic_counts[self.odometry_topic] += 1
        self.update_span(msg.header.stamp)
        pose = msg.pose.pose
        self.poses.append((
            stamp_to_tum(msg.header.stamp),
            finite_float(pose.position.x),
            finite_float(pose.position.y),
            finite_float(pose.position.z),
            finite_float(pose.orientation.x),
            finite_float(pose.orientation.y),
            finite_float(pose.orientation.z),
            finite_float(pose.orientation.w),
        ))

    def on_pointcloud(self, msg):
        self.topic_counts[self.pointcloud_topic] += 1
        self.update_span(msg.header.stamp)

    def on_status(self, msg):
        self.topic_counts[self.status_topic] += 1
        self.last_status = tracking_status_to_dict(msg)

    def flush(self):
        trajectory_path = self.output_dir / "trajectory.tum"
        metrics_path = self.output_dir / "metrics.json"
        with trajectory_path.open("w", encoding="utf-8") as stream:
            for pose in sorted(self.poses, key=lambda item: item[0]):
                stream.write(
                    f"{pose[0]:.9f} {pose[1]:.9f} {pose[2]:.9f} {pose[3]:.9f} "
                    f"{pose[4]:.9f} {pose[5]:.9f} {pose[6]:.9f} {pose[7]:.9f}\n")

        duration_sec = 0.0
        if self.first_stamp_ns is not None and self.last_stamp_ns is not None:
            duration_sec = max(0.0, (self.last_stamp_ns - self.first_stamp_ns) * 1.0e-9)
        topic_hz = {
            topic: (count / duration_sec if duration_sec > 0.0 else 0.0)
            for topic, count in self.topic_counts.items()
        }
        metrics = {
            "duration_sec": duration_sec,
            "topic_counts": dict(self.topic_counts),
            "topic_hz": topic_hz,
            "trajectory_poses": len(self.poses),
            "tracking_status": {
                "samples": self.topic_counts[self.status_topic],
                "last": self.last_status,
            },
            "outputs": {
                "trajectory_tum": str(trajectory_path),
            },
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(args=None):
    rclpy.init(args=args)
    node = NativeTrackingRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
