#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import os
from sensor_msgs_py import point_cloud2
import glob
from std_srvs.srv import SetBool
import math

def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert euler angles to quaternion.
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = [0] * 4
    q[0] = sr * cp * cy - cr * sp * sy  # x
    q[1] = cr * sp * cy + sr * cp * sy  # y
    q[2] = cr * cp * sy - sr * sp * cy  # z
    q[3] = cr * cp * cy + sr * sp * sy  # w

    return q

class KittiPlayerNode(Node):
    def __init__(self):
        super().__init__('kitti_player')
        
        # Declare parameters
        self.declare_parameter('publish_rate', 3.0)
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, '/lidar_streamer_first_reflection', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/detection_markers', 10)
        
        # Get list of KITTI bin files and labels
        self.velodyne_dir = '/lidar3d_detection_ws/data/kitti_velodyne_bins'
        self.label_dir = '/lidar3d_detection_ws/data/kitti_velodune_labels'
        
        # Add fallback paths if the primary paths don't exist
        if not os.path.exists(self.velodyne_dir):
            self.get_logger().warn(f'Primary velodyne directory not found: {self.velodyne_dir}')
            # Try testing directory as fallback
            self.velodyne_dir = '/lidar3d_detection_ws/data/kitti_velodyne_bins'
            self.get_logger().info(f'Trying fallback velodyne directory: {self.velodyne_dir}')
        
        # Check if directories exist
        if not os.path.exists(self.velodyne_dir):
            self.get_logger().error(f'Velodyne directory not found: {self.velodyne_dir}')
            raise FileNotFoundError(f'Directory not found: {self.velodyne_dir}')
        
        if not os.path.exists(self.label_dir):
            self.get_logger().error(f'Label directory not found: {self.label_dir}')
            raise FileNotFoundError(f'Directory not found: {self.label_dir}')
        
        # Get sorted lists of both bin and label files
        self.bin_files = sorted(glob.glob(os.path.join(self.velodyne_dir, '*.bin')))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, '*.txt')))
        
        if not self.bin_files:
            self.get_logger().error('No .bin files found!')
            raise FileNotFoundError('No .bin files found!')
        
        if len(self.bin_files) != len(self.label_files):
            self.get_logger().error(f'Mismatch in number of files: {len(self.bin_files)} bin files vs {len(self.label_files)} label files')
        
        self.get_logger().info(f'Found {len(self.bin_files)} .bin files and {len(self.label_files)} label files')
        self.current_frame = 0
        
        # Create timer for publishing frames
        self.create_timer(3.0, self.publish_frame)
        
        # Create service for pausing playback
        self.paused = False
        self.create_service(SetBool, 'pause_playback', self.pause_callback)
        
        # Colors for different classes
        self.class_colors = {
            0: (1.0, 0.0, 0.0, 1.0),    # Pedestrian: Red
            1: (0.0, 1.0, 0.0, 1.0),    # Car: Green
            2: (0.0, 0.0, 1.0, 1.0),    # Cyclist: Blue
        }
        
        # Class name to ID mapping
        self.class_name_to_id = {
            'Pedestrian': 0,
            'Car': 1,
            'Cyclist': 2,
            'Van': 1,
            'Truck': -3,
            'Person_sitting': 0,
            'Tram': -99,
            'Misc': -99,
            'DontCare': -1
        }
    
    def read_label_file(self, frame_id):
        label_path = os.path.join(self.label_dir, f'{frame_id:06d}.txt')
        objects = []
        
        if not os.path.exists(label_path):
            self.get_logger().warn(f'Label file not found: {label_path}')
            return objects
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split(' ')
            obj_type = parts[0]
            
            # Skip DontCare objects
            if obj_type == 'DontCare':
                continue
            
            # Get 3D location (in camera coordinates)
            x = float(parts[11])
            y = float(parts[12])
            z = float(parts[13])
            
            # Get dimensions
            h = float(parts[8])
            w = float(parts[9])
            l = float(parts[10])
            
            # Get rotation
            ry = float(parts[14])
            
            # Convert from camera to LiDAR coordinates
            x_lidar = z
            y_lidar = -x
            z_lidar = -y
            
            objects.append({
                'type': obj_type,
                'x': x_lidar,
                'y': y_lidar,
                'z': z_lidar,
                'h': h,
                'w': w,
                'l': l,
                'ry': ry - np.pi/2
            })
        
        return objects
    
    def create_cube_marker(self, obj, marker_id):
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.CUBE
        
        # Set position
        marker.pose.position.x = obj['x']
        marker.pose.position.y = obj['y']
        marker.pose.position.z = obj['z']
        
        # Set orientation (quaternion from yaw angle)
        q = quaternion_from_euler(0, 0, obj['ry'])
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        
        # Set scale
        marker.scale.x = obj['l']  # length
        marker.scale.y = obj['w']  # width
        marker.scale.z = obj['h']  # height
        
        # Set color based on object type
        if obj['type'] == 'Car' or obj['type'] == 'Van' or obj['type'] == 'Truck':
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif obj['type'] == 'Pedestrian' or obj['type'] == 'Person_sitting':
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        elif obj['type'] == 'Cyclist':
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        marker.color.a = 0.5
        
        return marker
    
    def yaw_to_quaternion(self, yaw):
        # Convert yaw angle to quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return [0.0, sy, 0.0, cy]
    
    def publish_frame(self):
        if self.paused:
            return
        
        if self.current_frame >= len(self.bin_files):
            self.current_frame = 0
        
        # Get current bin and label files
        current_bin = self.bin_files[self.current_frame]
        frame_id = int(os.path.basename(current_bin)[:-4])
        
        # Read point cloud
        points = np.fromfile(current_bin, dtype=np.float32).reshape(-1, 4)
        
        # Create and publish PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'innoviz'
        
        # Include intensity in the point cloud message
        pc2_msg = point_cloud2.create_cloud(msg.header, [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='intensity', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
        ], points)
        
        self.pc_pub.publish(pc2_msg)
        
        # First, send a delete all markers message
        delete_marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = "velodyne"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        delete_marker_array.markers.append(delete_marker)
        self.marker_pub.publish(delete_marker_array)
        
        # Read and publish new labels as markers
        objects = self.read_label_file(frame_id)
        
        # Create marker array for new objects
        marker_array = MarkerArray()
        
        # Create markers for each object
        for i, obj in enumerate(objects):
            marker = self.create_cube_marker(obj, i)
            marker_array.markers.append(marker)
        
        # Publish new marker array if there are any objects
        if marker_array.markers:
            self.marker_pub.publish(marker_array)
        
        self.get_logger().info(f'Published frame {self.current_frame}/{len(self.bin_files)} with {len(objects)} objects')
        self.current_frame += 1
    
    def pause_callback(self, request, response):
        self.paused = request.data
        response.success = True
        response.message = 'Playback paused' if self.paused else 'Playback resumed'
        return response

def main():
    rclpy.init()
    node = KittiPlayerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()