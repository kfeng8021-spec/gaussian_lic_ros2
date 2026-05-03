# SPDX-License-Identifier: GPL-3.0-or-later

from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2


class TopicProbe(Node):
    def __init__(self):
        super().__init__("topic_probe")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("pointcloud_topic", "/livox/lidar")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("report_period_sec", 2.0)

        self.counts = defaultdict(int)
        self.last_counts = defaultdict(int)
        self.last_report_time = self.get_clock().now()

        self._subscribe("image", Image, self.get_parameter("image_topic").value)
        self._subscribe("camera_info", CameraInfo, self.get_parameter("camera_info_topic").value)
        self._subscribe("pointcloud", PointCloud2, self.get_parameter("pointcloud_topic").value)
        self._subscribe("imu", Imu, self.get_parameter("imu_topic").value)

        period = float(self.get_parameter("report_period_sec").value)
        self.timer = self.create_timer(period, self.report)
        self.get_logger().info("Topic probe started")

    def _subscribe(self, key, msg_type, topic):
        self.create_subscription(
            msg_type,
            topic,
            lambda msg, topic_key=key: self._on_msg(topic_key, msg),
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Subscribed to {topic} as {key}")

    def _on_msg(self, key, _msg):
        self.counts[key] += 1

    def report(self):
        now = self.get_clock().now()
        dt = (now - self.last_report_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        parts = []
        for key in ("image", "camera_info", "pointcloud", "imu"):
            delta = self.counts[key] - self.last_counts[key]
            hz = delta / dt
            parts.append(f"{key}={hz:.2f}Hz total={self.counts[key]}")
            self.last_counts[key] = self.counts[key]

        self.last_report_time = now
        self.get_logger().info(" | ".join(parts))


def main(args=None):
    rclpy.init(args=args)
    node = TopicProbe()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

