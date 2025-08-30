#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import numpy as np
from math import atan2, pi

class LidarFOVFilter(Node):
    def __init__(self):
        super().__init__('lidar_fov_filter')
        
        # Declare parameters
        self.declare_parameter('view_direction', 0.0)
        self.declare_parameter('view_width', 2 * pi)
        self.declare_parameter('input_topic', '/ouster/points')
        self.declare_parameter('output_topic', '/ouster/points_filtered')
        
        # Get parameters
        self.view_direction = self.get_parameter('view_direction').value
        self.view_width = self.get_parameter('view_width').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        
        # Validate parameters
        if not (-pi <= self.view_direction <= pi):
            self.get_logger().warn(f'view_direction ({self.view_direction}) should be between -π and π')
        
        if not (0.0 <= self.view_width <= 2 * pi):
            self.get_logger().warn(f'view_width ({self.view_width}) should be between 0 and 2π')
        
        # Calculate FOV bounds
        self.min_angle = self.view_direction - self.view_width / 2.0
        self.max_angle = self.view_direction + self.view_width / 2.0
        
        self.get_logger().info(
            f"Filtering LiDAR FOV: direction={self.view_direction:.3f} rad, "
            f"width={self.view_width:.3f} rad, "
            f"range=[{self.min_angle:.3f}, {self.max_angle:.3f}] rad"
        )
        
        # Create subscription and publisher
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.pointcloud_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            PointCloud2,
            output_topic,
            10
        )
        
        self.get_logger().info(f"Subscribing to {input_topic}, publishing to {output_topic}")
    
    def pointcloud_callback(self, msg):
        # Convert PointCloud2 to list of points
        points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        # Filter points based on horizontal angle
        filtered_points = []
        for point in points:
            x, y, z = point
            # Calculate horizontal angle (azimuth)
            angle = atan2(y, x)
            
            # Check if angle is within the specified FOV
            if self.is_angle_in_range(angle):
                filtered_points.append(point)
        
        # Create filtered PointCloud2 message
        if filtered_points:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
            ]
            
            filtered_cloud = point_cloud2.create_cloud(msg.header, fields, filtered_points)
            self.publisher.publish(filtered_cloud)
    
    def is_angle_in_range(self, angle):
        # Handle angle wrapping around ±π
        if self.min_angle <= self.max_angle:
            return self.min_angle <= angle <= self.max_angle
        else:
            # FOV wraps around the ±π boundary
            return angle >= self.min_angle or angle <= self.max_angle


def main(args=None):
    rclpy.init(args=args)
    lidar_fov_filter = LidarFOVFilter()
    
    try:
        rclpy.spin(lidar_fov_filter)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_fov_filter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()