#!/usr/bin/env python3
# bin_publisher_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
import struct
import time
from std_msgs.msg import Header

class BinPublisherNode(Node):
    def __init__(self):
        super().__init__('bin_publisher_node')
        
        # Declare parameters
        self.declare_parameter('bin_directory', '/lidar3d_detection_ws/data/innoviz')
        self.declare_parameter('output_topic', '/lidar_streamer_first_reflection')
        self.declare_parameter('publish_rate', 1.0)  # Hz
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('loop', True)
        
        # Get parameters
        self.bin_directory = self.get_parameter('bin_directory').value
        self.output_topic = self.get_parameter('output_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.loop = self.get_parameter('loop').value
        
        # Initialize publisher
        self.publisher = self.create_publisher(PointCloud2, self.output_topic, 10)
        
        # Get list of bin files
        self.bin_files = sorted([f for f in os.listdir(self.bin_directory) if f.endswith('.bin')])
        self.current_bin_index = 0
        
        if not self.bin_files:
            self.get_logger().error(f"No .bin files found in {self.bin_directory}")
            return
            
        self.get_logger().info(f"Found {len(self.bin_files)} .bin files in {self.bin_directory}")
        
        # Create timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_next_bin)
        
    def publish_next_bin(self):
        if self.current_bin_index >= len(self.bin_files):
            if self.loop:
                self.current_bin_index = 0
                self.get_logger().info("Looping back to first bin file")
            else:
                self.get_logger().info("Finished publishing all bin files")
                self.timer.cancel()
                return
        
        # Get current bin file
        bin_file = os.path.join(self.bin_directory, self.bin_files[self.current_bin_index])
        self.get_logger().info(f"Publishing bin file: {bin_file}")
        
        try:
            # Read bin file
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            
            # Create point cloud message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
            
            # Create a PointCloud2 message
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            
            cloud_msg = pc2.create_cloud(header, fields, points)
            
            # Publish the message
            self.publisher.publish(cloud_msg)
            self.get_logger().info(f"Published point cloud with {points.shape[0]} points")
            #sleep for 1 second
            time.sleep(1)
            
            # Increment index
            self.current_bin_index += 1
            
        except Exception as e:
            self.get_logger().error(f"Error publishing bin file: {e}")
            self.current_bin_index += 1  # Skip to next file

def main(args=None):
    rclpy.init(args=args)
    node = BinPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()