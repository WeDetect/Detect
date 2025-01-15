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
from std_msgs.msg import Header

def quaternion_from_euler(roll, pitch, yaw):
    """Convert euler angles to quaternion."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    q = [0] * 4
    q[0] = sr * cp * cy - cr * sp * sy  # x
    q[1] = cr * sp * cy + sr * cp * sy  # y
    q[2] = cr * cp * sy - sr * sp * cy  # z
    q[3] = cr * cp * cy + sr * sp * sy  # w
    return q

class KittiPlayerNode(Node):
    def __init__(self):
        super().__init__('kitti_player')
        
        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, '/ipcl_grabber', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/detection_markers', 10)
        
        # Paths for data
        self.velodyne_dir = '/home/ori/ai/od_ws/src/data/training/velodyne'
        self.label_dir = '/home/ori/ai/od_ws/src/data/label_2'
        
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
        
        # Create timer for publishing frames (every 3 seconds)
        self.create_timer(3.0, self.publish_frame)
        
        # Create service for pausing playback
        self.paused = False
        self.create_service(SetBool, 'pause_playback', self.pause_callback)
        
        # Class colors for visualization
        self.class_colors = {
            'Car': (1.0, 0.0, 0.0, 0.5),       # Red
            'Pedestrian': (0.0, 1.0, 0.0, 0.5), # Green
            'Cyclist': (0.0, 0.0, 1.0, 0.5),    # Blue
            'Van': (1.0, 0.0, 0.0, 0.5),        # Red (same as Car)
            'Person_sitting': (0.0, 1.0, 0.0, 0.5)  # Green (same as Pedestrian)
        }

    def read_label_file(self, frame_id):
        """Read and parse KITTI label file."""
        label_path = os.path.join(self.label_dir, f'{frame_id:06d}.txt')
        objects = []
        
        if not os.path.exists(label_path):
            return objects
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                obj_type = parts[0]
                
                if obj_type == 'DontCare':
                    continue
                    
                # Parse object data
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
                ry = float(parts[14])
                
                # Convert camera coordinates to LiDAR coordinates
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
        """Create a cube marker for visualization."""
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Set position and orientation
        marker.pose.position.x = obj['x']
        marker.pose.position.y = obj['y']
        marker.pose.position.z = obj['z']
        
        q = quaternion_from_euler(0, 0, obj['ry'])
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        
        # Set dimensions
        marker.scale.x = obj['l']
        marker.scale.y = obj['w']
        marker.scale.z = obj['h']
        
        # Set color
        color = self.class_colors.get(obj['type'], (1.0, 1.0, 1.0, 0.5))
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker

    def publish_frame(self):
        """Publish point cloud and detection markers for current frame."""
        if self.paused:
            return
            
        if self.current_frame >= len(self.bin_files):
            self.current_frame = 0
            
        # Get current frame files
        current_bin = self.bin_files[self.current_frame]
        frame_id = int(os.path.basename(current_bin)[:-4])
        
        # Read point cloud
        points = np.fromfile(current_bin, dtype=np.float32).reshape(-1, 4)
        
        # Create and publish PointCloud2 message with proper header
        header = PointCloud2().header
        header.frame_id = 'velodyne'
        header.stamp = self.get_clock().now().to_msg()
        
        pc2_msg = point_cloud2.create_cloud_xyz32(header, points[:, :3])
        self.pc_pub.publish(pc2_msg)
        
        # Clear previous markers
        delete_markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = "velodyne"
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        delete_markers.markers.append(delete_marker)
        self.marker_pub.publish(delete_markers)
        
        # Create and publish new markers
        objects = self.read_label_file(frame_id)
        marker_array = MarkerArray()
        
        for i, obj in enumerate(objects):
            marker = self.create_cube_marker(obj, i)
            marker_array.markers.append(marker)
            
        if marker_array.markers:
            self.marker_pub.publish(marker_array)
            
        self.get_logger().info(f'Published frame {self.current_frame}/{len(self.bin_files)} with {len(objects)} objects')
        self.current_frame += 1

    def pause_callback(self, request, response):
        """Handle pause/resume requests."""
        self.paused = request.data
        response.success = True
        response.message = 'Paused' if self.paused else 'Resumed'
        return response

def main(args=None):
    rclpy.init(args=args)
    node = KittiPlayerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()