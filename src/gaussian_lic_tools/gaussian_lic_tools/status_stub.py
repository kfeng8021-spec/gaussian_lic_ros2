# SPDX-License-Identifier: GPL-3.0-or-later

import rclpy
from rclpy.node import Node

from gaussian_lic_msgs.msg import MappingStatus


class StatusStub(Node):
    def __init__(self):
        super().__init__("status_stub")
        self.declare_parameter("status_topic", "/gaussian_lic/status")
        self.declare_parameter("publish_period_sec", 1.0)

        status_topic = self.get_parameter("status_topic").value
        period = float(self.get_parameter("publish_period_sec").value)

        self.publisher = self.create_publisher(MappingStatus, status_topic, 10)
        self.timer = self.create_timer(period, self.publish_status)
        self.start_time = self.get_clock().now()
        self.get_logger().info(f"Publishing stub status on {status_topic}")

    def publish_status(self):
        msg = MappingStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.state = MappingStatus.STATE_INACTIVE
        msg.status_text = "ROS2 status stub running; launch with stub_mode:=false for native mapper"
        msg.tracking_state = MappingStatus.STATE_UNCONFIGURED
        msg.mapping_state = MappingStatus.STATE_INACTIVE
        msg.render_mode = MappingStatus.RENDER_MODE_OFF
        msg.num_gaussians = 0
        msg.num_keyframes = 0
        msg.num_tracking_frames = 0
        msg.num_mapping_frames = 0
        msg.num_dropped_messages = 0
        msg.num_errors = 0
        msg.tracking_hz = 0.0
        msg.mapping_hz = 0.0
        msg.tracking_latency_ms = 0.0
        msg.mapping_latency_ms = 0.0
        msg.mean_iteration_ms = 0.0
        msg.gpu_memory_mb = 0.0
        msg.active_profile = "stub"
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StatusStub()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
