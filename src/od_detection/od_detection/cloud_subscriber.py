#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import os
import time
import cv2
from sensor_msgs_py import point_cloud2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import datetime
from od_detection.functions_for_kitti import makeBEVMap, get_filtered_lidar, makeBEVMapUpgrade, makeHorizontalView
import od_detection.kitti_config as cnf

class PointCloudToBEVNode(Node):
    def __init__(self):
        super().__init__('cloud_subscriber')
        
        # Create subscribers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/ipcl_grabber',
            self.lidar_callback,
            10)
        
        # Add MarkerArray subscriber
        self.marker_sub = self.create_subscription(
            MarkerArray,
            '/detection_markers',
            self.marker_callback,
            10)
        
        # Initialize variables
        self.debug_dir = '/home/ori/ai/od_ws/data/images'
        self.current_markers = None
        self.current_frame = 0
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Add label directory path
        self.label_dir = '/home/ori/ai/od_ws/src/data/label_2'
        self.current_frame = 0
        
        # Colors for different classes
        self.class_colors = {
            'Car': (255, 0, 0),      # Red
            'Pedestrian': (0, 255, 0),  # Green
            'Cyclist': (0, 0, 255),   # Blue
            'Van': (255, 0, 0),      # Red (same as Car)
            'Person_sitting': (0, 255, 0)  # Green (same as Pedestrian)
        }

    def pointcloud2_to_array(self, cloud_msg):
        pc_list = []
        for p in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            pc_list.append([p[0], p[1], p[2], 0.0])  # Add intensity value of 0
        return np.array(pc_list, dtype=np.float32)

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

    def draw_detections(self, image, objects, view_type='bev'):
        """Draw detection boxes on the image"""
        img_h, img_w = image.shape[:2]
        image_draw = image.copy()
        
        for obj in objects:
            if view_type == 'bev':
                # Convert 3D coordinates to BEV image coordinates with proper flipping
                x = int(img_w - (obj['x'] / cnf.DISCRETIZATION + img_w/2))
                y = int(img_h - (obj['y'] / cnf.DISCRETIZATION + img_h/2))
                
                # Calculate box dimensions in pixels
                w = int(obj['w'] / cnf.DISCRETIZATION)
                l = int(obj['l'] / cnf.DISCRETIZATION)
                
                # Adjust angle for coordinate system and rotation
                angle = np.degrees(obj['ry']) + 180
                rect = ((x, y), (w, l), angle)
                box = np.int0(cv2.boxPoints(rect))
                
                # Draw the rectangle with thinner lines
                color = self.class_colors.get(obj['type'], (255, 255, 255))
                cv2.polylines(image_draw, [box], True, color, 1)
                
                # Add label text
                cv2.putText(image_draw, obj['type'], (x-w//2, y-l//2-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return image_draw

    def marker_callback(self, msg):
        """Store the latest markers"""
        self.current_markers = msg

    def draw_markers(self, image, markers):
        """Draw markers from MarkerArray on the image"""
        if markers is None:
            return image
            
        img_h, img_w = image.shape[:2]
        
        for marker in markers.markers:
            if marker.type != Marker.CUBE:
                continue
                
            # Direct conversion of marker position to image coordinates
            x = int((marker.pose.position.x / cnf.DISCRETIZATION) + img_w/2)
            y = int((marker.pose.position.y / cnf.DISCRETIZATION) + img_h/2)
            
            # Calculate dimensions
            w = int(marker.scale.y / cnf.DISCRETIZATION)  # width
            l = int(marker.scale.x / cnf.DISCRETIZATION)  # length
            
            # Get rotation directly from quaternion
            q = marker.pose.orientation
            yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            angle = np.degrees(yaw)
            
            # Draw rotated rectangle with thin lines
            rect = ((x, y), (w, l), angle)
            box = np.int0(cv2.boxPoints(rect))
            
            # Set color based on marker color
            color = (int(marker.color.b*255), int(marker.color.g*255), int(marker.color.r*255))
            cv2.polylines(image, [box], True, color, 1)
        
        return image

    def save_marker_label(self, markers, filename):
        """Save markers as KITTI format label file"""
        if markers is None:
            return
            
        label_path = os.path.join(self.debug_dir, filename)
        with open(label_path, 'w') as f:
            for marker in markers.markers:
                if marker.type != Marker.CUBE:
                    continue
                    
                # Determine object type from color
                obj_type = 'Unknown'
                if marker.color.r == 1.0: obj_type = 'Car'
                elif marker.color.g == 1.0: obj_type = 'Pedestrian'
                elif marker.color.b == 1.0: obj_type = 'Cyclist'
                
                # Write in KITTI format
                x = marker.pose.position.x
                y = marker.pose.position.y
                z = marker.pose.position.z
                h = marker.scale.z
                w = marker.scale.y
                l = marker.scale.x
                
                # Get rotation from quaternion
                q = marker.pose.orientation
                ry = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
                
                f.write(f'{obj_type} 0 0 0 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}\n')

    def find_points_in_marker(self, points, marker):
        """Find points that fall within a marker's boundaries"""
        # Get marker dimensions and position
        l, w, h = marker.scale.x, marker.scale.y, marker.scale.z
        x, y, z = marker.pose.position.x, marker.pose.position.y, marker.pose.position.z
        
        # Get rotation
        q = marker.pose.orientation
        yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
        
        # Create rotation matrix
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Transform points to marker's local coordinate system
        local_points = points - np.array([x, y, z])
        rotated_points = np.dot(local_points, R.T)
        
        # Find points within marker boundaries
        mask = (np.abs(rotated_points[:, 0]) <= l/2) & \
               (np.abs(rotated_points[:, 1]) <= w/2) & \
               (np.abs(rotated_points[:, 2]) <= h/2)
        
        return mask

    def process_point_cloud_with_markers(self, pc_array, markers):
        """Process point cloud and color points based on markers"""
        # Create color array for points (default white)
        colors = np.ones((pc_array.shape[0], 3), dtype=np.float32)
        
        if markers is not None:
            for marker in markers.markers:
                # Find points within this marker
                mask = self.find_points_in_marker(pc_array[:, :3], marker)
                
                # Color these points according to marker color
                colors[mask] = [marker.color.r, marker.color.g, marker.color.b]
        
        return colors

    def lidar_callback(self, msg):
        time = ""
      #  time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Add debug print at the start of callback
        self.get_logger().info(f'Received PointCloud2 message with {len(msg.data)} points')
        
        # 1. קבלת ענן הנקודות ופילטור ראשוני
        pc_array = self.pointcloud2_to_array(msg)
        filtered_pc = get_filtered_lidar(pc_array, cnf.boundary)
        
        # 2. יצירת מערך צבעים התחלתי - כל הנקודות שחורות
        point_colors = np.zeros((filtered_pc.shape[0], 3))
        detected_points_mask = np.zeros(filtered_pc.shape[0], dtype=bool)
        
        # 3. זיהוי הנקודות שנמצאות בתוך ה-markers
        if self.current_markers is not None:
            for marker in self.current_markers.markers:
                # מציאת הנקודות שנמצאות בתוך ה-marker הנוכחי
                points_in_marker = self.find_points_in_marker(filtered_pc[:, :3], marker)
                detected_points_mask |= points_in_marker  # ה�פת הנקודות למסיכה הכללית
        
        # 4. צביעת הנקודות שזוהו בלבן
        point_colors[detected_points_mask] = [1.0, 1.0, 1.0]
        
        # Debug print
        self.get_logger().info(f'Found {np.sum(detected_points_mask)} points inside markers')
        
        # 5. יצירת תמ�ות עם הנקודות המזוהות
        front_bevmap = makeBEVMapUpgrade(filtered_pc, cnf.boundary, point_colors)
        bev_img = (front_bevmap.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        horizontal_view = makeHorizontalView(filtered_pc, cnf.boundary, point_colors)
        horiz_img = (horizontal_view.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # Save images
        cv2.imwrite(os.path.join(self.debug_dir, f'bev_map_upgrade_{time}.png'), bev_img)
        cv2.imwrite(os.path.join(self.debug_dir, f'horizontal_view_{time}.png'), horiz_img)
        
        self.current_frame += 1

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToBEVNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
