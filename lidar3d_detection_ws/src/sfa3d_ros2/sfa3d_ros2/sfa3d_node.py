#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import torch
import sys
import os
from sensor_msgs_py import point_cloud2
from transforms3d.euler import euler2quat
import cv2
from geometry_msgs.msg import Point
import time

# Add SFA3D to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../SFA3D'))

# Make sure the SFA3D path is correctly added
sfa3d_path = os.path.join(os.path.dirname(__file__), '../../SFA3D')
if os.path.exists(sfa3d_path):
    sys.path.append(sfa3d_path)
    print(f"Added SFA3D path: {sfa3d_path}")
else:
    print(f"WARNING: SFA3D path does not exist: {sfa3d_path}")
    # Try alternative paths
    alt_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SFA3D')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../SFA3D')),
        '/home/ori/projects/project_x/sfa3d_ws/src/SFA3D'
    ]
    
    for path in alt_paths:
        if os.path.exists(path):
            sys.path.append(path)
            print(f"Added alternative SFA3D path: {path}")
            break
    else:
        print("ERROR: Could not find SFA3D directory!")

# Print the Python path for debugging
print(f"Python path: {sys.path}")

from sfa.models.fpn_resnet import PoseResNet, BasicBlock
from sfa.utils.evaluation_utils import decode, post_processing, convert_det_to_real_values
from sfa.data_process.kitti_bev_utils import makeBEVMap
from sfa.data_process.kitti_data_utils import get_filtered_lidar
import sfa.config.kitti_config as cnf
from sfa.utils.torch_utils import _sigmoid
from sfa.models.model_utils import create_model
from sfa.utils.demo_utils import parse_demo_configs, do_detect

class SFA3DNode(Node):
    def __init__(self):
        super().__init__('sfa3d_node')
        
        # Add debug flag
        self.debug = True
        self.debug_dir = "/tmp/sfa3d_debug"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Create subscribers and publishers
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/ipcl_grabber',
            self.lidar_callback,
            10)
            
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            '/sfa3d/detections',
            10)

        # Load configs with debug info
        self.get_logger().info("Loading model configuration...")
        self.configs = parse_demo_configs()
        

        # Load configs and model with progress indication
        self.get_logger().info("Loading model configuration...")
        self.configs = parse_demo_configs()
        
        # Use absolute path for model weights
        model_dir = os.path.join('/home/ori/projects/project_x/sfa3d_ws/src/SFA3D/checkpoints/fpn_resnet_18')
        self.configs.pretrained_path = os.path.join(model_dir, 'fpn_resnet_18_epoch_300.pth')
        
        # Check if model file exists
        if not os.path.exists(self.configs.pretrained_path):
            self.get_logger().error(f"Model weights not found at: {self.configs.pretrained_path}")
            # Try alternative paths
            alt_model_paths = [
                os.path.join('/home/ori/projects/project_x/sfa3d_ws/src/SFA3D/checkpoints/fpn_resnet_18', 'fpn_resnet_18_epoch_300.pth'),
                os.path.join(os.path.dirname(__file__), '../../../SFA3D/checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth')
            ]
            
            for path in alt_model_paths:
                if os.path.exists(path):
                    self.configs.pretrained_path = path
                    self.get_logger().info(f"Found model weights at alternative path: {path}")
                    break
            else:
                self.get_logger().error("Could not find model weights file!")
                raise FileNotFoundError(f"Model weights file not found")
        
        # Force CPU mode
        self.configs.no_cuda = True
        self.configs.gpu_idx = -1
        self.configs.device = torch.device('cpu')

        # Adjust detection confidence threshold
        self.configs.conf_thresh = 0.15  # Lower from 0.2 to detect more objects

        # Create model with progress logging
        self.get_logger().info("Creating model architecture...")
        self.model = create_model(self.configs)
        
        self.get_logger().info("Loading model weights...")
        self.model.load_state_dict(torch.load(
            self.configs.pretrained_path, 
            map_location='cpu'
        ))
        
        self.get_logger().info("Moving model to device...")
        self.model = self.model.to(device=self.configs.device)
        self.model.eval()

        self.get_logger().info("Model initialization complete!")

        print(f"KITTI boundary settings: {cnf.boundary}")

    def pointcloud2_to_array(self, cloud_msg):
        pc_list = []
        for p in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            # Add intensity value of 0 since it's not provided in the message
            pc_list.append([p[0], p[1], p[2], 0.0])
        return np.array(pc_list, dtype=np.float32)

    def lidar_callback(self, msg):
        start = time.time()
        # Convert PointCloud2 to numpy array
        pc_array = self.pointcloud2_to_array(msg)
        
        # Add aggressive downsampling before filtering
        pc_array = self.downsample_pointcloud(
            pc_array, 
            method='voxel',
            params={'voxel_size': 0.2}  # Larger voxel size for more aggressive downsampling
        )
        
        # print("\n=== Processing New Point Cloud ===")
        # print(f"1. Raw point cloud shape: {pc_array.shape}")
        # print(f"Point cloud intensity range: [{pc_array[:,3].min():.2f}, {pc_array[:,3].max():.2f}]")
        
        # Create BEV map
        front_lidar = get_filtered_lidar(pc_array, cnf.boundary)
        # print(f"\n2. After filtering - points remaining: {front_lidar.shape[0]}/{pc_array.shape[0]}")
        # print(f"Filtered range - X: [{front_lidar[:,0].min():.2f}, {front_lidar[:,0].max():.2f}]")
        # print(f"Filtered range - Y: [{front_lidar[:,1].min():.2f}, {front_lidar[:,1].max():.2f}]")
        
        front_bevmap = makeBEVMap(front_lidar, cnf.boundary)
        #print(f"\n3. BEV map shape before processing: {front_bevmap.shape}")
        
        # Just convert to tensor without permuting - match demo_dataset.py lines 76-78
        front_bevmap = torch.from_numpy(front_bevmap).float()
        #print(f"4. Tensor shape before detection: {front_bevmap.shape}")
        
        # Run detection
        with torch.no_grad():
            detections, front_bevmap, fps = do_detect(self.configs, self.model, front_bevmap, is_front=True)
        
        print(f"\n5. Detection Results:")
        print(f"Raw detections structure: {type(detections)}")
        
        if isinstance(detections, dict):
            print("Detection dictionary contents:")
            for key, value in detections.items():
                if len(value) > 0:
                    print(f"Key: {key}, Value type: {type(value)}, Shape: {value.shape}")
                    print(f"Detection scores for class {key}: {value[:,0]}")  # Print confidence scores
            
            # Convert dictionary to list format
            det_list = [[] for _ in range(3)]
            for cls_id in range(3):
                if cls_id in detections:
                    det_list[cls_id] = detections[cls_id]
            
            kitti_dets = convert_det_to_real_values(det_list)
        else:
            kitti_dets = convert_det_to_real_values(detections)
        
        #print(f"###number of kitti_dets after conversion: {len(kitti_dets)}")
        
        if len(kitti_dets) > 0:
            print("Detected objects:")
            for i, det in enumerate(kitti_dets):
                obj_type = "Car" if det[0] == 0 else "Pedestrian" if det[0] == 1 else "Cyclist"
                print(f"  {i+1}. {obj_type} at position (x={det[1]:.2f}, y={det[2]:.2f}, z={det[3]:.2f})")
            end = time.time()
            length = end - start
            print(f"Detection time: {length:.2f} seconds")
            # Publish markers
            self.publish_detection_markers(kitti_dets)
        else:
            print("No objects detected")

        # After creating front_bevmap
        if self.debug:
            # Save raw BEV map
            bev_img = front_bevmap.numpy()
            bev_img = np.transpose(bev_img, (1, 2, 0))  # CHW to HWC
            bev_img = (bev_img * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.debug_dir, 'bev_map.png'), bev_img)
            
            # Print BEV map statistics
            #print(f"\nBEV Map Statistics:")
            #print(f"Min value: {front_bevmap.min()}")
            #print(f"Max value: {front_bevmap.max()}")
            #print(f"Mean value: {front_bevmap.mean()}")

    def publish_detection_markers(self, kitti_dets):
        marker_array = MarkerArray()
        
        # Colors for wireframe edges
        class_colors = {
            0: (1.0, 0.0, 0.0, 1.0),    # Car: Red
            1: (0.0, 1.0, 0.0, 1.0),    # Pedestrian: Green
            2: (0.0, 0.0, 1.0, 1.0),    # Cyclist: Blue
        }
        
        for i, det in enumerate(kitti_dets):
            # Create wireframe cube using LINE_LIST
            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            
            # Set position and orientation
            marker.pose.position.x = float(det[1])
            marker.pose.position.y = float(det[2])
            marker.pose.position.z = float(det[3])
            
            quat = self.yaw_to_quaternion(float(det[7]))
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            
            # Scale dimensions
            scale_factor = 1.5
            length = float(det[5]) * scale_factor
            width = float(det[6]) * scale_factor
            height = float(det[4]) * scale_factor
            
            # Set line width
            marker.scale.x = 0.05  # Line thickness
            
            # Define the 12 lines of the cube
            points = []
            # Bottom face
            points.extend([(-length/2, -width/2, -height/2), (length/2, -width/2, -height/2)])
            points.extend([(length/2, -width/2, -height/2), (length/2, width/2, -height/2)])
            points.extend([(length/2, width/2, -height/2), (-length/2, width/2, -height/2)])
            points.extend([(-length/2, width/2, -height/2), (-length/2, -width/2, -height/2)])
            # Top face
            points.extend([(-length/2, -width/2, height/2), (length/2, -width/2, height/2)])
            points.extend([(length/2, -width/2, height/2), (length/2, width/2, height/2)])
            points.extend([(length/2, width/2, height/2), (-length/2, width/2, height/2)])
            points.extend([(-length/2, width/2, height/2), (-length/2, -width/2, height/2)])
            # Vertical edges
            points.extend([(-length/2, -width/2, -height/2), (-length/2, -width/2, height/2)])
            points.extend([(length/2, -width/2, -height/2), (length/2, -width/2, height/2)])
            points.extend([(length/2, width/2, -height/2), (length/2, width/2, height/2)])
            points.extend([(-length/2, width/2, -height/2), (-length/2, width/2, height/2)])
            
            # Add points to marker
            for p1, p2 in zip(points[::2], points[1::2]):
                marker.points.extend([Point(x=p1[0], y=p1[1], z=p1[2]),
                                    Point(x=p2[0], y=p2[1], z=p2[2])])
            
            # Set color
            class_id = int(det[0])
            marker.color.r = class_colors[class_id][0]
            marker.color.g = class_colors[class_id][1]
            marker.color.b = class_colors[class_id][2]
            marker.color.a = class_colors[class_id][3]
            
            # Set lifetime
            marker.lifetime.sec = 2
            marker.lifetime.nanosec = 0
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)

    def yaw_to_quaternion(self, yaw):
        # Convert yaw angle to quaternion
        qw = np.cos(yaw / 2)
        qz = np.sin(yaw / 2)
        return [float(0), float(0), float(qz), float(qw)]  # Explicitly convert to float

    def downsample_pointcloud(self, points, method='uniform', params=None):
        """Aggressively downsample point cloud and reduce intensity to match Velodyne characteristics
        
        Args:
            points: Nx4 array of points (x,y,z,intensity)
            method: Downsampling method ('uniform', 'voxel', or 'random')
            params: Parameters for downsampling method
        
        Returns:
            Downsampled point cloud array with reduced intensity
        """
        if params is None:
            params = {}
        
        # First, reduce intensity values
        points = self.reduce_intensity(points)
        
        if method == 'uniform':
            # Take every Nth point (more aggressive uniform downsampling)
            skip_points = params.get('skip_factor', 4)  # Increased from 2 to 4
            return points[::skip_points]
            
        elif method == 'voxel':
            # Voxel grid downsampling with larger voxels
            voxel_size = params.get('voxel_size', 0.2)  # Increased from 0.1 to 0.2 meters
            
            # Compute voxel indices for each point
            voxel_indices = np.floor(points[:, :3] / voxel_size).astype(int)
            
            # Find unique voxels and keep center point
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            return points[unique_indices]
            
        elif method == 'random':
            # More aggressive random downsampling
            target_size = params.get('target_size', len(points) // 4)  # Reduced to 1/4 of points
            indices = np.random.choice(len(points), target_size, replace=False)
            return points[indices]
        
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

    def reduce_intensity(self, points):
        """Reduce intensity values to better match Velodyne characteristics
        
        Args:
            points: Nx4 array of points (x,y,z,intensity)
        
        Returns:
            Points with reduced intensity values
        """
        # Create a copy to avoid modifying the original array
        points_reduced = points.copy()
        
        # Scale down intensity values (column 3)
        intensity_scale_factor = 0.5  # Reduce intensity by half
        points_reduced[:, 3] *= intensity_scale_factor
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, points_reduced[:, 3].shape)
        points_reduced[:, 3] = np.clip(points_reduced[:, 3] + noise, 0, 1)
        
        # Optional: Quantize intensity values to simulate lower bit depth
        num_levels = 64  # Velodyne-like quantization
        points_reduced[:, 3] = np.floor(points_reduced[:, 3] * num_levels) / num_levels
        
        return points_reduced

def main(args=None):
    rclpy.init(args=args)
    node = SFA3DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()