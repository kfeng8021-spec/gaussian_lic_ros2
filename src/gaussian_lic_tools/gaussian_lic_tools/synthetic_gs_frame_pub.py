# SPDX-License-Identifier: GPL-3.0-or-later

import struct

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField


class SyntheticGsFramePublisher(Node):
    def __init__(self):
        super().__init__("synthetic_gs_frame_pub")
        self.declare_parameter("pointcloud_topic", "/points_for_gs")
        self.declare_parameter("pose_topic", "/pose_for_gs")
        self.declare_parameter("image_topic", "/image_for_gs")
        self.declare_parameter("camera_info_topic", "/camera_info_for_gs")
        self.declare_parameter("depth_topic", "/depth_for_gs")
        self.declare_parameter("imu_topic", "/imu_for_gs")
        self.declare_parameter("publish_rate_hz", 5.0)

        self.points_pub = self.create_publisher(
            PointCloud2, self.get_parameter("pointcloud_topic").value, 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, self.get_parameter("pose_topic").value, 10)
        self.image_pub = self.create_publisher(
            Image, self.get_parameter("image_topic").value, 10)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, self.get_parameter("camera_info_topic").value, 10)
        self.depth_pub = self.create_publisher(
            Image, self.get_parameter("depth_topic").value, 10)
        self.imu_pub = self.create_publisher(
            Imu, self.get_parameter("imu_topic").value, 10)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.publish_frame)
        self.frame_id = 0
        self.get_logger().info("Publishing synthetic synchronized GS input frames")

    def publish_frame(self):
        stamp = self.get_clock().now().to_msg()

        points = PointCloud2()
        points.header.stamp = stamp
        points.header.frame_id = "map"
        points.height = 1
        points.width = 1
        points.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        points.is_bigendian = False
        points.point_step = 16
        points.row_step = points.point_step
        points.is_dense = True
        rgb_bits = (255 << 16) | (32 << 8) | 16
        rgb_float = struct.unpack("f", struct.pack("I", rgb_bits))[0]
        points.data = struct.pack("ffff", 0.0, 0.0, 1.0, rgb_float)
        self.points_pub.publish(points)

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "map"
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = "camera"
        image.height = 1
        image.width = 1
        image.encoding = "bgr8"
        image.step = 3
        image.data = bytes([0, 0, 0])
        self.image_pub.publish(image)

        camera_info = CameraInfo()
        camera_info.header.stamp = stamp
        camera_info.header.frame_id = "camera"
        camera_info.height = 1
        camera_info.width = 1
        camera_info.distortion_model = "plumb_bob"
        camera_info.d = []
        camera_info.k = [1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0]
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        camera_info.p = [1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info_pub.publish(camera_info)

        depth = Image()
        depth.header.stamp = stamp
        depth.header.frame_id = "camera"
        depth.height = 1
        depth.width = 1
        depth.encoding = "32FC1"
        depth.step = 4
        depth.data = struct.pack("f", 1.0)
        self.depth_pub.publish(depth)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "imu"
        imu.orientation.w = 1.0
        imu.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu.linear_acceleration.z = 9.80665
        imu.linear_acceleration_covariance = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        self.imu_pub.publish(imu)

        self.frame_id += 1


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticGsFramePublisher()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt, RCLError):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RCLError):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (KeyboardInterrupt, RCLError):
                pass


if __name__ == "__main__":
    main()
