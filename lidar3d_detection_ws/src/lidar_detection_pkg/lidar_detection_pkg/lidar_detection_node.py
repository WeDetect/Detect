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
import sys
import time
import numba
from numba import jit, prange
from rclpy.clock import Clock


# Add parent directory to path for imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


project_dir = '/lidar3d_detection_ws'  
sys.path.append(project_dir)


os.chdir(project_dir)

try:
    from train.data_processing.preprocessing import load_config
    from train.data_processing.preproccesing_0 import PointCloudProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# ANSI escape codes for colored output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class LidarDetectionNode(Node):
    def __init__(self):
        super().__init__('lidar_detection_node')
        
        # Load parameters
        self.declare_parameter('pointcloud_topic', '/lidar_streamer_first_reflection')
        self.declare_parameter('markers_topic', '/detected_objects')
        self.declare_parameter('model_path', '/lidar3d_detection_ws/best_inferences/best.pt')
        self.declare_parameter('config_path', '/lidar3d_detection_ws/train/config/preprocessing_config.yaml')
        self.declare_parameter('confidence_threshold', 0.3)
        
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
        
        # Define class colors (BGR format for OpenCV, but RGBA for RViz)
        self.class_colors = {
            0: ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0),  # DontCare - Black
            1: ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),  # Car - Green
            2: ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),  # Pedestrian - White
            3: ColorRGBA(r=0.6, g=0.0, b=0.8, a=1.0),  # Cyclist - Purple
            4: ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0),  # Bus - Yellow
            5: ColorRGBA(r=0.0, g=0.8, b=0.0, a=1.0)   # Truck - Dark Green
        }
        
        # יצירת הprocessor עם הקונפיג מהקובץ
        self.processor = PointCloudProcessor(config_path=self.config_path)
        
        # שים לב! לא לשנות את ה-bev_height ו-bev_width של ה-processor
        # יש להשתמש בערכים מהקובץ בדיוק כמו באימון
        # אין לשנות את הערכים הבאים:
        # self.processor.bev_height = 640
        # self.processor.bev_width = 640
        
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
        # Add detection image publisher
        self.detection_image_pub = self.create_publisher(Image, '/detection_image', 10)
        
        self.get_logger().info("Lidar detection node initialized")
        
        # Add a dictionary to store previous marker positions for smoothing
        self.previous_markers = {}
        
        # Smoothing factor (0-1): 0 = use only new position, 1 = use only old position
        self.smoothing_factor = 0.3
        
        # Marker lifetime in seconds
        self.marker_lifetime = 0.1  # Longer than the typical update rate
        
        # Add performance tracking
        self.processing_times = []
        self.max_times_to_track = 10
        
    # Optimized function for creating BEV image
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _create_bev_image_optimized_numba(
        x_points, y_points, z_points, intensity,
        fwd_range, side_range, height_range,
        resolution, z_resolution,
        x_max, y_max, z_max
    ):
        # Initialize BEV image array
        bev_image = np.zeros((y_max + 1, x_max + 1, 3), dtype=np.uint8)
        
        # Filter points that are within the specified ranges
        indices = []
        for i in range(len(x_points)):
            x = x_points[i]
            y = y_points[i]
            z = z_points[i]
            
            if (x > fwd_range[0] and x < fwd_range[1] and
                y > -side_range[1] and y < -side_range[0] and
                z > height_range[0] and z < height_range[1]):
                indices.append(i)
        
        # Define floor height range
        floor_min_height = -2.0
        floor_max_height = 0.3
        
        # Calculate which height slices correspond to the floor
        floor_min_idx = max(0, int((floor_min_height - height_range[0]) / z_resolution))
        floor_max_idx = min(z_max, int((floor_max_height - height_range[0]) / z_resolution))
        
        # Process each filtered point
        for idx in indices:
            x = x_points[idx]
            y = y_points[idx]
            z = z_points[idx]
            intens = intensity[idx]
            
            # Convert coordinates to pixel positions
            x_img = int(-y / resolution)
            y_img = int(-x / resolution)
            
            # Shift coordinates to image space
            x_img -= int(side_range[0] / resolution)
            y_img += int(fwd_range[1] / resolution)
            
            # Check if the point is within image bounds
            if (0 <= x_img < x_max + 1 and 0 <= y_img < y_max + 1):
                # Calculate height slice
                z_idx = int((z - height_range[0]) / z_resolution)
                if 0 <= z_idx < z_max:
                    # Skip floor points - they will remain black (0,0,0)
                    if z_idx >= floor_min_idx and z_idx <= floor_max_idx:
                        continue
                    
                    # Create a color gradient from blue (lower) to red (higher)
                    b = max(0, 255 - (z_idx * 255 // z_max))
                    r = min(255, z_idx * 255 // z_max)
                    g = min(100, z_idx * 100 // z_max)
                    
                    # Set the color in the image
                    bev_image[y_img, x_img, 0] = r
                    bev_image[y_img, x_img, 1] = g
                    bev_image[y_img, x_img, 2] = b
                else:
                    # Use intensity for points outside height range
                    intensity_val = min(255, max(0, int(intens)))
                    bev_image[y_img, x_img, 0] = intensity_val
                    bev_image[y_img, x_img, 1] = intensity_val
                    bev_image[y_img, x_img, 2] = intensity_val
        
        return bev_image
    
    def create_bev_image_fast(self, points):
        """
        Create a bird's eye view image from point cloud data - optimized version
        """
        start_time = time.time()
        
        # Extract point coordinates
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        intensity = points[:, 3]
        
        # Get parameters from processor
        fwd_range = np.array(self.processor.fwd_range)
        side_range = np.array(self.processor.side_range)
        height_range = np.array(self.processor.height_range)
        resolution = self.processor.resolution
        z_resolution = self.processor.z_resolution
        x_max = self.processor.x_max
        y_max = self.processor.y_max
        z_max = self.processor.z_max
        
        # Call the numba-optimized function
        bev_image = self._create_bev_image_optimized_numba(
            x_points, y_points, z_points, intensity,
            fwd_range, side_range, height_range,
            resolution, z_resolution,
            x_max, y_max, z_max
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Track processing time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_times_to_track:
            self.processing_times.pop(0)
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        self.get_logger().info(f"BEV image creation time: {processing_time:.4f}s, Avg: {avg_time:.4f}s")
        
        return bev_image
        
    def adjust_yaw_for_bounding_boxes(self, results):
        """
        Adjust the yaw of bounding boxes based on detection results.
        
        Parametersככככככ
        -----------
        results : list
            List of detection results from the YOLO model.
        
        Returns:
        --------
        list
            List of adjusted bounding boxes with corrected yaw.
        """
        adjusted_boxes = []
        for result in results:
            # Ensure result has the expected structure
            if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                continue
            
            for box in result.boxes:
                # Access the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                x = (x1 + x2) / 2
                y = (y1 + y2) / 2
                z = 0  # Assuming z is 0 or calculated elsewhere
                
                # Calculate or extract yaw
                if width > height:
                    yaw = 0  # Horizontal orientation
                else:
                    yaw = np.pi / 2  # Vertical orientation
                
                # Create adjusted bounding box
                adjusted_box = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'width': width,
                    'height': height,
                    'yaw': yaw
                }
                adjusted_boxes.append(adjusted_box)
        
        return adjusted_boxes

    # Optimized function for filtering points
    @staticmethod
    @jit(nopython=True, parallel=True)
    def filter_points(points, x_min, x_max, y_min, y_max, z_min, z_max):
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        return points[mask]

    @staticmethod
    def convert_pointcloud_to_numpy(msg):
        # Read all points into a structured array
        pc_data = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        return pc_data.astype(np.float32)

    def pointcloud_callback(self, msg):
        try:
            callback_start = time.time()
            self.get_logger().info(f"{bcolors.OKCYAN}Received point cloud with frame_id: {msg.header.frame_id}{bcolors.ENDC}")
            
            # Convert ROS2 PointCloud2 to numpy array
            conversion_start = time.time()
            points = self.convert_pointcloud_to_numpy(msg)
            conversion_end = time.time()
            self.get_logger().info(f"{bcolors.OKGREEN}Converted point cloud to numpy array in {conversion_end - conversion_start:.4f}s{bcolors.ENDC}")
            
            # Transform points from innoviz frame to world frame using TF
            if msg.header.frame_id != 'world':
                try:
                    transform_start = time.time()
                    self.get_logger().info(f"{bcolors.OKBLUE}Transforming points from {msg.header.frame_id} to world frame{bcolors.ENDC}")
                    
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
                    r = Rotation.from_quat([q.x, q.y, q.z, q.w])
                    rotation_matrix = r.as_matrix()
                    
                    # Apply rotation and translation to all points at once
                    points_xyz = points[:, :3]
                    points_intensity = points[:, 3:]
                    
                    # Rotate and translate
                    points_transformed = np.dot(points_xyz, rotation_matrix.T) + np.array([t.x, t.y, t.z])
                    
                    # Combine back with intensity
                    points = np.hstack((points_transformed, points_intensity))
                    
                    transform_end = time.time()
                    self.get_logger().info(f"{bcolors.OKGREEN}Transformed {len(points)} points to world frame in {transform_end - transform_start:.4f}s{bcolors.ENDC}")
                    
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    self.get_logger().error(f"TF Error: {e}")
                    return
            
            # Filter points based on spatial range
            filter_start = time.time()
            filtered_points = self.filter_points(points, self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)
            
            if filtered_points.shape[0] == 0:
                self.get_logger().warn("No points left after filtering - using original points")
                filtered_points = points  # Fall back to unfiltered points for debugging
            
            filter_end = time.time()
            self.get_logger().info(f"{bcolors.OKGREEN}Filtered points in {filter_end - filter_start:.4f}s, remaining: {filtered_points.shape[0]}{bcolors.ENDC}")
            
            # Use our optimized BEV image creation function
            bev_start = time.time()
            bev_img = self.create_bev_image_fast(filtered_points)
            bev_end = time.time()
            self.get_logger().info(f"{bcolors.OKGREEN}Created BEV image in {bev_end - bev_start:.4f}s{bcolors.ENDC}")
            
            # Publish BEV image
            try:
                img_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding="bgr8")
                img_msg.header = msg.header
                self.bev_image_pub.publish(img_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish BEV image: {str(e)}")
            
            # Run inference with the model
            inference_start = time.time()
            results = self.model.predict(
                source=bev_img,
                conf=self.confidence_threshold,
                iou=0.5,
                verbose=False
            )
            inference_end = time.time()
            self.get_logger().info(f"{bcolors.OKGREEN}Model inference took {inference_end - inference_start:.4f}s{bcolors.ENDC}")
            
            # Adjust yaw for bounding boxes
            adjusted_detections = self.adjust_yaw_for_bounding_boxes(results)
            
            # Process results and create markers
            markers_start = time.time()
            # Check if we have any detections
            if not results or len(results) == 0 or not hasattr(results[0], 'boxes') or len(results[0].boxes) == 0:
                self.get_logger().info("No detections found")
                num_detections = 0
                
                # Create empty marker array when no detections
                markers = MarkerArray()
                
                # Delete any previous markers
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                delete_marker.header = msg.header
                markers.markers.append(delete_marker)
                
                # Publish empty markers
                self.marker_pub.publish(markers)
            else:
                num_detections = len(results[0].boxes)
                self.get_logger().info(f"Detected {num_detections} objects")
                
                # Get detection image and publish it
                det_image = results[0].plot()
                try:
                    det_img_msg = self.bridge.cv2_to_imgmsg(det_image, encoding="bgr8")
                    det_img_msg.header = msg.header
                    self.detection_image_pub.publish(det_img_msg)
                except Exception as e:
                    self.get_logger().error(f"Failed to publish detection image: {str(e)}")
                
                # Extract class names from the detection image
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    # Get box coordinates directly (no scaling needed)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_idx = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    # Map the class index correctly - YOLO models often use different indexing
                    # 0 = Car, 1 = Pedestrian, 2 = Cyclist
                    mapped_cls = cls_idx + 1  # Add 1 to map 0->1 (Car), 1->2 (Pedestrian), 2->3 (Cyclist)
                    
                    # Print both the raw and mapped class
                    #self.get_logger().info(f"Detection {i+1}: Raw Class Index: {cls_idx}, Mapped to: {self.class_names[mapped_cls]}, Confidence: {conf:.2f}")
                    #self.get_logger().info(f"Box coords: ({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f})")
                
                # Process results and create 3D markers with the correct class mapping
                markers = self.create_3d_markers(results[0], filtered_points, msg.header, use_mapped_class=True)
                
                # Publish markers
                self.marker_pub.publish(markers)
            
            markers_end = time.time()
            self.get_logger().info(f"{bcolors.OKGREEN}Created and published markers in {markers_end - markers_start:.4f}s{bcolors.ENDC}")
            
            callback_end = time.time()
            total_time = callback_end - callback_start
            self.get_logger().info(f"{bcolors.OKGREEN}Total processing time: {total_time:.4f}s{bcolors.ENDC}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    

    def create_3d_markers(self, results, points, header, use_mapped_class=False):
        """
        Create 3D markers for visualization in RViz - optimized version
        """
        markers = MarkerArray()

        # זמן נוכחי (סנכרון מלא בין מרקרים לתמונה)
        now_stamp = header.stamp  # השתמש בזמן המקורי של ה-PointCloud


        # Header מעודכן
        world_header = Header()
        world_header.stamp = now_stamp
        world_header.frame_id = "world"

        # הכנה לרשימת מזהים פעילים
        current_ids = set()

        # במקרה ואין תוצאות
        if results is None or not hasattr(results, 'boxes') or results.boxes is None or len(results.boxes) == 0:
            return markers

        boxes = results.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()

            if use_mapped_class:
                cls += 1  # 0->1 (Car), 1->2 (Pedestrian), 2->3 (Cyclist)

            x_center_img = (x1 + x2) / 2
            y_center_img = (y1 + y2) / 2
            width_img = abs(x2 - x1)
            length_img = abs(y2 - y1)

            # המרה חזרה למערכת קואורדינטות עולמית
            x_img_adjusted = x_center_img - int(np.floor(-self.processor.side_range[0] / self.processor.resolution))
            y_img_adjusted = y_center_img - int(np.floor(self.processor.fwd_range[1] / self.processor.resolution))

            world_y = -x_img_adjusted * self.processor.resolution
            world_x = -y_img_adjusted * self.processor.resolution
            world_width = width_img * self.processor.resolution
            world_length = length_img * self.processor.resolution

            class_height = self.class_dimensions[cls][0]
            z_min = 0.0
            z_max = z_min + class_height

            # תיבת Wireframe
            marker = Marker()
            marker.header = world_header
            marker.ns = "detected_objects"
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.05
            marker.lifetime = rclpy.duration.Duration(seconds=0.3).to_msg()

            half_length = world_length / 2
            half_width = world_width / 2
            vertices = [
                [world_x - half_length, world_y - half_width, 0.0],
                [world_x + half_length, world_y - half_width, 0.0],
                [world_x + half_length, world_y + half_width, 0.0],
                [world_x - half_length, world_y + half_width, 0.0],
                [world_x - half_length, world_y - half_width, class_height],
                [world_x + half_length, world_y - half_width, class_height],
                [world_x + half_length, world_y + half_width, class_height],
                [world_x - half_length, world_y + half_width, class_height]
            ]
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            for line in lines:
                for idx in line:
                    p = Point(x=vertices[idx][0], y=vertices[idx][1], z=vertices[idx][2])
                    marker.points.append(p)
                    marker.colors.append(self.class_colors[cls])

            markers.markers.append(marker)

            # טקסט עם שם המחלקה
            text_marker = Marker()
            text_marker.header = world_header
            text_marker.ns = "class_labels"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = world_x
            text_marker.pose.position.y = world_y
            text_marker.pose.position.z = class_height + 0.5
            text_marker.text = f"{self.class_names[cls]} ({conf:.2f})"
            text_marker.scale.z = 0.8
            text_marker.color = self.class_colors[cls]
            text_marker.color.a = 1.0
            text_marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
            markers.markers.append(text_marker)

            current_ids.add(i)

        # מחיקת מרקרים שלא קיימים בפריים הזה
        if not hasattr(self, 'previous_marker_ids'):
            self.previous_marker_ids = set()

        deleted_ids = self.previous_marker_ids - current_ids
        for del_id in deleted_ids:
            m = Marker()
            m.header = world_header
            m.ns = "detected_objects"
            m.id = del_id
            m.action = Marker.DELETE
            markers.markers.append(m)

            m_text = Marker()
            m_text.header = world_header
            m_text.ns = "class_labels"
            m_text.id = del_id
            m_text.action = Marker.DELETE
            markers.markers.append(m_text)

        self.previous_marker_ids = current_ids
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