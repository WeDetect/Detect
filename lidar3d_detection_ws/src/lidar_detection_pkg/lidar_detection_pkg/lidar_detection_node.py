#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import torch
import os
import yaml
from pathlib import Path
from ultralytics import YOLO
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import ColorRGBA, Header
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation
import tf2_geometry_msgs
from cv_bridge import CvBridge

# Add parent directory to path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# הוסף את תיקיית הפרויקט הראשית לנתיב החיפוש
project_dir = '/lidar3d_detection_ws'  # שנה לנתיב הנכון אם צריך
sys.path.append(project_dir)

# וודא שנתיב העבודה הנוכחי מוגדר נכון
os.chdir(project_dir)

try:
    from train.data_processing.preprocessing import create_bev_image, load_config
    from train.data_processing.preproccesing_0 import PointCloudProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

class LidarDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')
        
        # Load parameters
        self.declare_parameter('pointcloud_topic', '/lidar_streamer_first_reflection')
        self.declare_parameter('markers_topic', '/detected_objects')
        self.declare_parameter('model_path', '/lidar3d_detection_ws/train/epoch30.pt')
        self.declare_parameter('config_path', '/lidar3d_detection_ws/train/config/preprocessing_config.yaml')
        self.declare_parameter('confidence_threshold', 0.5)
        
        # Point cloud filtering parameters
        self.declare_parameter('x_min', 0.0)
        self.declare_parameter('x_max', 30.0)
        self.declare_parameter('y_min', -15.0)
        self.declare_parameter('y_max', 15.0)
        self.declare_parameter('z_min', -2.5)
        self.declare_parameter('z_max', 4.0)
        
        # Get parameters
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.config_path = self.get_parameter('config_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # Get filtering parameters
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.y_min = self.get_parameter('y_min').value
        self.y_max = self.get_parameter('y_max').value
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        
        # Load the preprocessing configuration directly like in evaluate.py
        self.config = load_config(self.config_path)
        
        # Class dimensions (height, width, length) in meters
        self.class_dimensions = {
            0: [0, 0, 0],  # DontCare
            1: [2.36, 2.36, 4.36],  # Car
            2: [2.0, 1.20, 1.20],  # Pedestrian
            3: [2.0, 1.5, 2.5],  # Cyclist
            4: [4.0, 3.5, 8.5],  # Bus
            5: [4.0, 3.5, 8.5]   # Truck
        }
        
        # Class names
        self.class_names = ['DontCare', 'Car', 'Pedestrian', 'Cyclist', 'Bus', 'Truck']
        
        # Class colors (r, g, b, a)
        self.class_colors = {
            0: ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.5),  # DontCare - gray
            1: ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.7),  # Car - red
            2: ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.7),  # Pedestrian - green
            3: ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7),  # Cyclist - blue
            4: ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7),  # Bus - yellow
            5: ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.7)   # Truck - magenta
        }
        
        # We still create processor for other functionality
        self.processor = PointCloudProcessor(config_path=self.config_path)
        
        # Update processor ranges with ROS parameters
        self.processor.fwd_range = (self.x_min, self.x_max)
        self.processor.side_range = (self.y_min, self.y_max)
        self.processor.height_range = (self.z_min, self.z_max)
        
        # Update processor dimensions based on the updated ranges
        self.processor.x_max = int((self.processor.fwd_range[1] - self.processor.fwd_range[0]) / self.processor.resolution)
        self.processor.y_max = int((self.processor.side_range[1] - self.processor.side_range[0]) / self.processor.resolution)
        self.processor.z_max = int((self.processor.height_range[1] - self.processor.height_range[0]) / self.processor.z_resolution)
        
        # Force BEV dimensions to match model input size (presumably 640x640)
        self.processor.bev_height = 640
        self.processor.bev_width = 640
        
        # Initialize CV Bridge for publishing images
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info(f"Loading model from {self.model_path}")
        self.model = YOLO(self.model_path)
        
        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Initialize publishers and subscribers
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            10
        )
        
        # Add BEV image publisher for visualization
        self.bev_image_pub = self.create_publisher(Image, '/bev_image', 10)
        
        self.get_logger().info("Lidar detection node initialized")
        
    def pointcloud_callback(self, msg):
        try:
            self.get_logger().info(f"Received point cloud with frame_id: {msg.header.frame_id}")
            
            # Convert ROS2 PointCloud2 to numpy array
            points_list = []
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                points_list.append([point[0], point[1], point[2], point[3]])
            
            if not points_list:
                self.get_logger().warn("Empty point cloud received")
                return
                
            points = np.array(points_list)
            
            # Transform points from innoviz frame to world frame using TF - FASTER VERSION
            if msg.header.frame_id != 'world':
                try:
                    self.get_logger().info(f"Transforming points from {msg.header.frame_id} to world frame")
                    
                    # Get the transform from TF
                    trans = self.tf_buffer.lookup_transform(
                        'world',
                        msg.header.frame_id,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    
                    # Extract rotation and translation components
                    q = trans.transform.rotation
                    t = trans.transform.translation
                    
                    # Convert quaternion to rotation matrix
                    quat = [q.x, q.y, q.z, q.w]
                    rot_matrix = Rotation.from_quat(quat).as_matrix()
                    
                    # Apply transformation in vectorized form
                    # First apply rotation
                    points_xyz = points[:, :3]
                    points_intensity = points[:, 3:]
                    
                    # Apply rotation in vectorized form
                    transformed_points_xyz = np.dot(points_xyz, rot_matrix.T)
                    
                    # Apply translation in vectorized form
                    transformed_points_xyz[:, 0] += t.x
                    transformed_points_xyz[:, 1] += t.y
                    transformed_points_xyz[:, 2] += t.z
                    
                    # Combine back with intensity
                    points = np.hstack((transformed_points_xyz, points_intensity))
                    
                    self.get_logger().info(f"Transformed {len(points)} points from {msg.header.frame_id} to world frame")
                    
                    # Create a fresh header
                    header = Header()
                    header.stamp = msg.header.stamp
                    header.frame_id = 'world'
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().error(f"Failed to transform points: {str(e)}")
                    # אם אין טרנספורמציה, נשתמש בנקודות המקוריות ונשנה את ה-frame_id
                    self.get_logger().warn("Using original points and changing frame_id to 'world'")
                    # Create a fresh header
                    header = Header()
                    header.stamp = msg.header.stamp
                    header.frame_id = 'world'
            else:
                # Create a fresh header
                header = Header()
                header.stamp = msg.header.stamp
                header.frame_id = 'world'
            
            # Filter points based on our defined range parameters
            mask = (
                (points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max) &
                (points[:, 1] >= self.y_min) & (points[:, 1] <= self.y_max) &
                (points[:, 2] >= self.z_min) & (points[:, 2] <= self.z_max)
            )
            filtered_points = points[mask]
            
            if filtered_points.shape[0] == 0:
                self.get_logger().warn("No points left after filtering - using original points")
                filtered_points = points  # Fall back to unfiltered points for debugging
            
            # Log point stats before passing to processor
            self.get_logger().info(f"Filtered points: {filtered_points.shape[0]}")
            self.get_logger().info(f"X range: {np.min(filtered_points[:,0]):.2f} to {np.max(filtered_points[:,0]):.2f}")
            self.get_logger().info(f"Y range: {np.min(filtered_points[:,1]):.2f} to {np.max(filtered_points[:,1]):.2f}")
            self.get_logger().info(f"Z range: {np.min(filtered_points[:,2]):.2f} to {np.max(filtered_points[:,2]):.2f}")
            
            # Create BEV image using PointCloudProcessor's create_bev_image
            bev_image = self.processor.create_bev_image(filtered_points)
            
            # Ensure image is uint8, not float32
            if bev_image.dtype != np.uint8:
                self.get_logger().info(f"Converting BEV image from {bev_image.dtype} to uint8")
                bev_image = (bev_image * 255).clip(0, 255).astype(np.uint8)
            
            # Save BEV image to file for visualization
            timestamp = msg.header.stamp.sec * 1000 + msg.header.stamp.nanosec // 1000000
            image_path = f"/lidar3d_detection_ws/bev_image_{timestamp}.png"
            cv2.imwrite(image_path, bev_image)
            self.get_logger().info(f"Saved BEV image to {image_path}")
            
            # Run inference with YOLO model
            results = self.model.predict(bev_image, conf=self.confidence_threshold)
            
            # Check if we have any detections
            if not results or len(results) == 0 or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
                self.get_logger().info("No detections found")
                num_detections = 0
                
                # Create empty marker array when no detections
                markers = MarkerArray()
                
                # Delete any previous markers
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                delete_marker.header = header
                markers.markers.append(delete_marker)
                
                # Publish empty markers
                self.marker_pub.publish(markers)
            else:
                num_detections = len(results[0].boxes)
                self.get_logger().info(f"Detected {num_detections} objects")
                
                # Save detected image
                det_image = results[0].plot()
                det_image_path = f"/lidar3d_detection_ws/bev_image_detections_{timestamp}.png"
                cv2.imwrite(det_image_path, det_image)
                self.get_logger().info(f"Saved detection image to {det_image_path}")
                
                # Log detection details
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates directly (no scaling needed)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    self.get_logger().info(f"Detection {i+1}: Class {self.class_names[cls]}, Confidence: {conf:.2f}")
                    self.get_logger().info(f"Box coords: ({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f})")
                
                # Process results and create 3D markers
                markers = self.create_3d_markers(results[0], filtered_points, header)
                
                # Publish markers
                self.marker_pub.publish(markers)
            
            # Also publish BEV image
            try:
                img_msg = self.bridge.cv2_to_imgmsg(bev_image, encoding="bgr8")
                img_msg.header = header
                self.bev_image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish BEV image: {str(e)}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")
            # Print stack trace for debugging
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def create_3d_markers(self, results, points, header):
        """
        Create 3D markers for visualization in RViz
        """
        # Create marker array message
        markers = MarkerArray()
        
        # Delete any previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header = header
        markers.markers.append(delete_marker)
        
        # If no results, return empty marker array
        if results is None or not hasattr(results, 'boxes') or results.boxes is None or len(results.boxes) == 0:
            return markers
        
        boxes = results.boxes
        for i, box in enumerate(boxes):
            # Get box coordinates in BEV image
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # Convert BEV coordinates to world coordinates
            # Calculate center and dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = abs(x2 - x1)
            length = abs(y2 - y1)
            
            # Convert from pixel to world coordinates
            # These are VERY IMPORTANT to get correct!
            # We're working with the original image dimensions from the processor
            img_height, img_width = self.processor.bev_height, self.processor.bev_width
            
            # Convert center from pixel to world coordinates
            world_y = self.processor.side_range[0] + (x_center / img_width) * (self.processor.side_range[1] - self.processor.side_range[0])
            world_x = self.processor.fwd_range[1] - (y_center / img_height) * (self.processor.fwd_range[1] - self.processor.fwd_range[0])
            
            # Convert width and length from pixel to world dimensions
            world_width = (width / img_width) * (self.processor.side_range[1] - self.processor.side_range[0])
            world_length = (length / img_height) * (self.processor.fwd_range[1] - self.processor.fwd_range[0])
            
            # Estimate the Z position and height
            # We'll use points that lie inside the bounding box
            points_in_bbox = []
            for point in points:
                point_x, point_y = point[0], point[1]
                # Check if the point is inside our 2D bounding box in world coordinates
                in_x = (world_x - world_length/2) <= point_x <= (world_x + world_length/2)
                in_y = (world_y - world_width/2) <= point_y <= (world_y + world_width/2)
                if in_x and in_y:
                    points_in_bbox.append(point)
            
            # If we don't have enough points, use default height
            if len(points_in_bbox) < 5:
                z_min = self.z_min
                z_max = z_min + self.class_dimensions[cls][0]  # Use class height
            else:
                # Calculate Z statistics from the points
                z_values = [p[2] for p in points_in_bbox]
                z_min = np.min(z_values)
                z_max = np.max(z_values)
                
                # Use class height for more reliable height
                height = self.class_dimensions[cls][0]
                z_max = z_min + height
            
            # Create marker message for cuboid
            marker = Marker()
            marker.header = header
            marker.ns = "detected_objects"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = (z_min + z_max) / 2  # Center of the cuboid
            
            # Set orientation (no rotation for now)
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Set scale
            marker.scale.x = world_length
            marker.scale.y = world_width
            marker.scale.z = z_max - z_min
            
            # Set color
            marker.color = self.class_colors[cls]
            
            # Set lifetime and other properties
            marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
            
            # Add to marker array
            markers.markers.append(marker)
            
            # Add text marker with class name
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "class_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            # Position slightly above the cuboid
            text_marker.pose.position.x = world_x
            text_marker.pose.position.y = world_y
            text_marker.pose.position.z = z_max + 0.5  # Above the cuboid
            
            # Set text
            text_marker.text = f"{self.class_names[cls]} ({conf:.2f})"
            
            # Set scale (text size)
            text_marker.scale.z = 0.8
            
            # Set color
            text_marker.color = self.class_colors[cls]
            
            # Set lifetime
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
            
            # Add to marker array
            markers.markers.append(text_marker)
        
        return markers

def main(args=None):
    rclpy.init(args=args)
    node = LidarDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()