#!/usr/bin/env python3
#pointcloud_saver_node.py

import os
import time
import numpy as np
import struct
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, IntegerRange
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
import tf2_ros
from geometry_msgs.msg import TransformStamped, Point, PointStamped
import message_filters
from pathlib import Path
import yaml
from scipy.spatial.transform import Rotation as R
from rcl_interfaces.msg import SetParametersResult


class KittiPointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver_node')
        
        # Declare parameters
        self.declare_node_parameters()
        
        # Get parameters
        self.global_frame = self.get_parameter('global_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.input_topic = self.get_parameter('input_topic').value
        self.filtered_topic = self.get_parameter('filtered_topic').value
        self.trigger_topic = self.get_parameter('trigger_topic').value
        self.save_directory = os.path.expanduser(self.get_parameter('save_directory').value)
        self.label_directory = os.path.expanduser(self.get_parameter('label_directory').value)
        self.world_min_z_filter = self.get_parameter('world_min_z_filter').value
        self.world_max_z_filter = self.get_parameter('world_max_z_filter').value
        self.world_min_y_filter = self.get_parameter('world_min_y_filter').value
        self.world_max_y_filter = self.get_parameter('world_max_y_filter').value
        self.world_min_x_filter = self.get_parameter('world_min_x_filter').value
        self.world_max_x_filter = self.get_parameter('world_max_x_filter').value
        self.save_intensity = self.get_parameter('save_intensity').value
        self.file_prefix = self.get_parameter('file_prefix').value
        self.file_index = self.get_parameter('file_index').value
        
        # Create directories if they don't exist
        os.makedirs(self.save_directory, exist_ok=True)
        os.makedirs(self.label_directory, exist_ok=True)
        
        # Set up parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameters_callback)
        
        # Set up TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publishers and subscribers
        self.filtered_pub = self.create_publisher(PointCloud2, self.filtered_topic, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.pointcloud_callback,
            10)
        self.trigger_sub = self.create_subscription(
            Bool,
            self.trigger_topic,
            self.trigger_callback,
            10)
        
        # Store the latest filtered point cloud
        self.latest_filtered_cloud = None
        self.latest_raw_cloud = None
        
        # Register shutdown callback to save parameters
        rclpy.get_default_context().on_shutdown(self.shutdown_callback)
        
        # Log initialization
        self.get_logger().info('=' * 30)
        self.get_logger().info('\033[1;33mKPointCloud Saver Node initialized\033[0m')
        self.get_logger().info(f'\033[1;33mListening to: {self.input_topic}\033[0m')
        self.get_logger().info(f'\033[1;33mPublishing filtered cloud to: {self.filtered_topic}\033[0m')
        self.get_logger().info(f'\033[1;33mTrigger topic: {self.trigger_topic}\033[0m')
        self.get_logger().info(f'\033[1;33mSaving to: {self.save_directory}\033[0m')
        self.get_logger().info('=' * 30)
    
    def declare_node_parameters(self):
        # Frame parameters
        frame_desc = ParameterDescriptor(description='Frame ID')
        self.declare_parameter('global_frame', 'world', frame_desc)
        self.declare_parameter('lidar_frame', 'innoviz', frame_desc)
        
        # Topic parameters
        topic_desc = ParameterDescriptor(description='ROS topic name')
        self.declare_parameter('input_topic', '/lidar_streamer_first_reflection', topic_desc)
        self.declare_parameter('filtered_topic', '/pointcloud_saver/filtered_points', topic_desc)
        self.declare_parameter('trigger_topic', '/pointcloud_saver/save_trigger', topic_desc)
        
        # Directory parameters
        dir_desc = ParameterDescriptor(description='Directory path')
        self.declare_parameter('save_directory', '/lidar3d_detection_ws/data/innoviz/', dir_desc)
        self.declare_parameter('label_directory', '/lidar3d_detection_ws/data/labels/', dir_desc)
        
        # Filter parameters with ranges
        for axis in ['x', 'y', 'z']:
            min_desc = ParameterDescriptor(
                description=f'Minimum {axis.upper()} value for filtering points (in world frame)',
                floating_point_range=[FloatingPointRange(
                    from_value=-50.0,
                    to_value=50.0,
                    step=0.1
                )]
            )
            max_desc = ParameterDescriptor(
                description=f'Maximum {axis.upper()} value for filtering points (in world frame)',
                floating_point_range=[FloatingPointRange(
                    from_value=0.0,
                    to_value=100.0,
                    step=0.1
                )]
            )
            self.declare_parameter(f'world_min_{axis}_filter', -2.0, min_desc)
            self.declare_parameter(f'world_max_{axis}_filter', 4.5, max_desc)
        
        # Other parameters
        bool_desc = ParameterDescriptor(description='Whether to save intensity values')
        self.declare_parameter('save_intensity', True, bool_desc)
        
        prefix_desc = ParameterDescriptor(description='Prefix for saved files')
        self.declare_parameter('file_prefix', '', prefix_desc)
        
        index_desc = ParameterDescriptor(
            description='File index counter',
            integer_range=[IntegerRange(from_value=0, to_value=1000000, step=1)]
        )
        self.declare_parameter('file_index', 0, index_desc)
    
    def parameters_callback(self, params):
        result = SetParametersResult()
        result.successful = True
        
        params_changed = False
        changed_param_names = []
        
        for param in params:
            changed_param_names.append(param.name)
            
            if param.name == 'global_frame':
                self.global_frame = param.value
                params_changed = True
            elif param.name == 'lidar_frame':
                self.lidar_frame = param.value
                params_changed = True
            elif param.name == 'input_topic':
                # Need to recreate subscription if topic changes
                self.input_topic = param.value
                self.pointcloud_sub = self.create_subscription(
                    PointCloud2,
                    self.input_topic,
                    self.pointcloud_callback,
                    10)
                params_changed = True
            elif param.name == 'filtered_topic':
                # Need to recreate publisher if topic changes
                self.filtered_topic = param.value
                self.filtered_pub = self.create_publisher(PointCloud2, self.filtered_topic, 10)
                params_changed = True
            elif param.name == 'trigger_topic':
                # Need to recreate subscription if topic changes
                self.trigger_topic = param.value
                self.trigger_sub = self.create_subscription(
                    Bool,
                    self.trigger_topic,
                    self.trigger_callback,
                    10)
                params_changed = True
            elif param.name == 'save_directory':
                self.save_directory = os.path.expanduser(param.value)
                os.makedirs(self.save_directory, exist_ok=True)
                params_changed = True
            elif param.name == 'label_directory':
                self.label_directory = os.path.expanduser(param.value)
                os.makedirs(self.label_directory, exist_ok=True)
                params_changed = True
            elif param.name == 'world_min_z_filter':
                self.world_min_z_filter = param.value
                params_changed = True
            elif param.name == 'world_max_z_filter':
                self.world_max_z_filter = param.value
                params_changed = True
            elif param.name == 'world_min_y_filter':
                self.world_min_y_filter = param.value
                params_changed = True
            elif param.name == 'world_max_y_filter':
                self.world_max_y_filter = param.value
                params_changed = True
            elif param.name == 'world_min_x_filter':
                self.world_min_x_filter = param.value
                params_changed = True
            elif param.name == 'world_max_x_filter':
                self.world_max_x_filter = param.value
                params_changed = True
            elif param.name == 'save_intensity':
                self.save_intensity = param.value
                params_changed = True
            elif param.name == 'file_prefix':
                self.file_prefix = param.value
                params_changed = True
            elif param.name == 'file_index':
                self.file_index = param.value
                params_changed = True
        
        if params_changed:
            self.get_logger().info('Parameters updated')
            self.save_parameters_to_yaml(changed_param_names)
        
        return result
    
    def pointcloud_callback(self, msg):
        try:
            # Store the raw cloud
            self.latest_raw_cloud = msg
            
            # Wait for transform to be available
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame,
                    msg.header.frame_id,
                    rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warning(f'TF Error: {e}')
                return
            
            # Extract points directly using sensor_msgs_py.point_cloud2
            points_list = []
            intensities_list = []
            
            for point in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                # Each point is a tuple (x, y, z, intensity)
                points_list.append([point[0], point[1], point[2]])
                intensities_list.append(point[3])
            
            if not points_list:
                self.get_logger().warning('Received empty point cloud')
                return
            
            # Convert to numpy arrays
            points = np.array(points_list, dtype=np.float32)
            intensities = np.array(intensities_list, dtype=np.float32)
            
            # Get transformation matrix
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            # Convert quaternion to rotation matrix
            r = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
            rotation_matrix = r.as_matrix()
            
            # Apply rotation and translation to all points at once
            points_transformed = np.dot(points, rotation_matrix.T) + np.array([translation.x, translation.y, translation.z])
            
            # Apply filters in world frame
            mask = (
                (points_transformed[:, 0] >= self.world_min_x_filter) &
                (points_transformed[:, 0] <= self.world_max_x_filter) &
                (points_transformed[:, 1] >= self.world_min_y_filter) &
                (points_transformed[:, 1] <= self.world_max_y_filter) &
                (points_transformed[:, 2] >= self.world_min_z_filter) &
                (points_transformed[:, 2] <= self.world_max_z_filter)
            )
            
            filtered_points = points_transformed[mask]
            filtered_intensities = intensities[mask]
            
            # Combine points and intensities
            filtered_pc = np.column_stack((filtered_points, filtered_intensities))
            
            # Create a new PointCloud2 message for the filtered cloud
            filtered_msg = PointCloud2()
            filtered_msg.header = msg.header
            filtered_msg.header.frame_id = self.global_frame
            
            # Define the fields
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]
            
            # Create the PointCloud2 message
            filtered_msg = pc2.create_cloud(filtered_msg.header, fields, filtered_pc)
            
            # Store the filtered cloud
            self.latest_filtered_cloud = filtered_msg
            
            # Publish the filtered cloud
            self.filtered_pub.publish(filtered_msg)
            
            # Log stats occasionally
            if self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Roughly every second
                self.get_logger().info(f'Filtered {len(filtered_pc)} points from {len(points)} original points')
        
        except Exception as e:
            self.get_logger().error(f'Error in pointcloud_callback: {e}')
    
    def trigger_callback(self, msg):
        if not msg.data:
            return
        
        if self.latest_filtered_cloud is None:
            self.get_logger().warning('No point cloud data available to save')
            return
        
        try:
            # Generate filename with 5 digits (00000, 00001, etc.)
            filename = f'{self.file_prefix}{self.file_index:05d}.bin'
            filepath = os.path.join(self.save_directory, filename)
            
            # Convert PointCloud2 to numpy array - use the filtered cloud
            pc_array = []
            for p in pc2.read_points(self.latest_filtered_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                pc_array.append(p)
            
            if not pc_array:
                self.get_logger().warning('Filtered point cloud is empty, nothing to save')
                return
            
            # Convert to numpy array
            pc_array = np.array(pc_array)
            
            # Save in KITTI format (x, y, z, intensity as float32)
            with open(filepath, 'wb') as f:
                for point in pc_array:
                    f.write(struct.pack('ffff', point[0], point[1], point[2], point[3]))
            
            # Create an empty label file
            # label_filename = f'{self.file_prefix}{self.file_index:05d}.txt'
            # label_filepath = os.path.join(self.label_directory, label_filename)
            # with open(label_filepath, 'w') as f:
            #     pass  # Create empty file
            
            self.get_logger().info(f'\033[1;32mSaved point cloud to {filepath}\033[0m')
            # self.get_logger().info(f'\033[1;32mCreated empty label file at {label_filepath}\033[0m')
            self.get_logger().info(f'\033[1;32mSaved {len(pc_array)} points\033[0m')
            
            # Increment file index and save parameter
            self.file_index += 1
            self.set_parameters([Parameter('file_index', Parameter.Type.INTEGER, self.file_index)])
            
        except Exception as e:
            self.get_logger().error(f'Error saving point cloud: {e}')
    
    def save_parameters_to_yaml(self, changed_params=None):
        try:
            # Helper function to format doubles with decimal point
            def format_double(value):
                return f"{value:.1f}"
            
            # If no parameters changed, nothing to do
            if not changed_params:
                return
            
            # 1. Save to the install directory of lidar_solution_bringup package
            try:
                # Use ament_index to find the installed package path
                bringup_package_name = "lidar_solution_bringup"
                
                try:
                    from ament_index_python.packages import get_package_share_directory
                    bringup_package_path = get_package_share_directory(bringup_package_name)
                except Exception as e:
                    self.get_logger().error(f'Failed to get bringup package directory: {e}')
                    return
                
                config_dir = os.path.join(bringup_package_path, 'config')
                params_file_path = os.path.join(config_dir, 'parameters.yaml')
                
                # Create directory if it doesn't exist
                os.makedirs(config_dir, exist_ok=True)
                
                # Check if file exists
                if not os.path.exists(params_file_path):
                    # Create a basic YAML file with our node's parameters
                    with open(params_file_path, 'w') as file:
                        file.write("/**:\n")
                        file.write("  ros__parameters:\n")
                        file.write("    global_frame: world\n")
                        file.write("    lidar_frame: innoviz\n\n")
                        file.write("pointcloud_saver_node:\n")
                        file.write("  ros__parameters:\n")
                        for param_name in changed_params:
                            if param_name == 'global_frame':
                                file.write(f"    global_frame: {self.global_frame}\n")
                            elif param_name == 'lidar_frame':
                                file.write(f"    lidar_frame: {self.lidar_frame}\n")
                            elif param_name == 'input_topic':
                                file.write(f"    input_topic: {self.input_topic}\n")
                            elif param_name == 'filtered_topic':
                                file.write(f"    filtered_topic: {self.filtered_topic}\n")
                            elif param_name == 'trigger_topic':
                                file.write(f"    trigger_topic: {self.trigger_topic}\n")
                            elif param_name == 'save_directory':
                                file.write(f"    save_directory: {self.save_directory}\n")
                            elif param_name == 'label_directory':
                                file.write(f"    label_directory: {self.label_directory}\n")
                            elif param_name == 'world_min_z_filter':
                                file.write(f"    world_min_z_filter: {format_double(self.world_min_z_filter)}\n")
                            elif param_name == 'world_max_z_filter':
                                file.write(f"    world_max_z_filter: {format_double(self.world_max_z_filter)}\n")
                            elif param_name == 'world_min_y_filter':
                                file.write(f"    world_min_y_filter: {format_double(self.world_min_y_filter)}\n")
                            elif param_name == 'world_max_y_filter':
                                file.write(f"    world_max_y_filter: {format_double(self.world_max_y_filter)}\n")
                            elif param_name == 'world_min_x_filter':
                                file.write(f"    world_min_x_filter: {format_double(self.world_min_x_filter)}\n")
                            elif param_name == 'world_max_x_filter':
                                file.write(f"    world_max_x_filter: {format_double(self.world_max_x_filter)}\n")
                            elif param_name == 'save_intensity':
                                file.write(f"    save_intensity: {str(self.save_intensity).lower()}\n")
                            elif param_name == 'file_prefix':
                                # Handle empty string properly with quotes
                                prefix = '""' if self.file_prefix == "" else self.file_prefix
                                file.write(f"    file_prefix: {prefix}\n")
                            elif param_name == 'file_index':
                                file.write(f"    file_index: {self.file_index}\n")
                
                    self.get_logger().info(f'\033[1;33mCreated new parameters file: {params_file_path}\033[0m')
                    return
                
                # Read the file line by line
                with open(params_file_path, 'r') as file:
                    lines = file.readlines()
                
                # Process the file line by line
                new_lines = []
                in_pointcloud_saver_section = False
                in_ros_parameters_section = False
                pointcloud_saver_section_found = False
                
                for i, line in enumerate(lines):
                    # Check if we're entering the pointcloud_saver_node section
                    if line.strip() == "pointcloud_saver_node:" or line.strip().startswith("pointcloud_saver_node:"):
                        in_pointcloud_saver_section = True
                        pointcloud_saver_section_found = True
                        new_lines.append(line)
                        continue
                    
                    # Check if we're entering the ros__parameters section within pointcloud_saver
                    if in_pointcloud_saver_section and (line.strip() == "ros__parameters:" or 
                                                       line.strip().startswith("ros__parameters:")):
                        in_ros_parameters_section = True
                        new_lines.append(line)
                        continue
                    
                    # If we're in the ros__parameters section, check if this line is a parameter we need to update
                    if in_pointcloud_saver_section and in_ros_parameters_section:
                        param_line = line.lstrip()
                        if ':' in param_line:
                            param_name = param_line.split(':')[0].strip()
                            
                            # If this parameter needs to be updated
                            if param_name in changed_params:
                                # Replace with updated value
                                indent = len(line) - len(line.lstrip())
                                spaces = ' ' * indent
                                
                                if param_name == 'global_frame':
                                    new_lines.append(f"{spaces}global_frame: {self.global_frame}\n")
                                elif param_name == 'lidar_frame':
                                    new_lines.append(f"{spaces}lidar_frame: {self.lidar_frame}\n")
                                elif param_name == 'input_topic':
                                    new_lines.append(f"{spaces}input_topic: {self.input_topic}\n")
                                elif param_name == 'filtered_topic':
                                    new_lines.append(f"{spaces}filtered_topic: {self.filtered_topic}\n")
                                elif param_name == 'trigger_topic':
                                    new_lines.append(f"{spaces}trigger_topic: {self.trigger_topic}\n")
                                elif param_name == 'save_directory':
                                    new_lines.append(f"{spaces}save_directory: {self.save_directory}\n")
                                elif param_name == 'label_directory':
                                    new_lines.append(f"{spaces}label_directory: {self.label_directory}\n")
                                elif param_name == 'world_min_z_filter':
                                    new_lines.append(f"{spaces}world_min_z_filter: {format_double(self.world_min_z_filter)}\n")
                                elif param_name == 'world_max_z_filter':
                                    new_lines.append(f"{spaces}world_max_z_filter: {format_double(self.world_max_z_filter)}\n")
                                elif param_name == 'world_min_y_filter':
                                    new_lines.append(f"{spaces}world_min_y_filter: {format_double(self.world_min_y_filter)}\n")
                                elif param_name == 'world_max_y_filter':
                                    new_lines.append(f"{spaces}world_max_y_filter: {format_double(self.world_max_y_filter)}\n")
                                elif param_name == 'world_min_x_filter':
                                    new_lines.append(f"{spaces}world_min_x_filter: {format_double(self.world_min_x_filter)}\n")
                                elif param_name == 'world_max_x_filter':
                                    new_lines.append(f"{spaces}world_max_x_filter: {format_double(self.world_max_x_filter)}\n")
                                elif param_name == 'save_intensity':
                                    new_lines.append(f"{spaces}save_intensity: {str(self.save_intensity).lower()}\n")
                                elif param_name == 'file_prefix':
                                    # Handle empty string properly with quotes
                                    prefix = '""' if self.file_prefix == "" else self.file_prefix
                                    new_lines.append(f"{spaces}file_prefix: {prefix}\n")
                                elif param_name == 'file_index':
                                    new_lines.append(f"{spaces}file_index: {self.file_index}\n")
                                
                                # Mark this parameter as processed
                                changed_params.remove(param_name)
                                continue
                        
                    # Check if we're exiting the pointcloud_saver section
                    if in_pointcloud_saver_section and line.strip() and not line.startswith(" "):
                        in_pointcloud_saver_section = False
                        in_ros_parameters_section = False
                        
                        # If there are still parameters to add, add them before exiting the section
                        if changed_params and pointcloud_saver_section_found:
                            spaces = "    "  # Standard indentation
                            for param_name in list(changed_params):  # Use list to avoid modifying during iteration
                                if param_name == 'global_frame':
                                    new_lines.append(f"{spaces}global_frame: {self.global_frame}\n")
                                elif param_name == 'lidar_frame':
                                    new_lines.append(f"{spaces}lidar_frame: {self.lidar_frame}\n")
                                # ... (repeat for all parameters)
                                elif param_name == 'file_prefix':
                                    # Handle empty string properly with quotes
                                    prefix = '""' if self.file_prefix == "" else self.file_prefix
                                    new_lines.append(f"{spaces}file_prefix: {prefix}\n")
                                elif param_name == 'file_index':
                                    new_lines.append(f"{spaces}file_index: {self.file_index}\n")
                                
                                changed_params.remove(param_name)
                    
                    # Add the line to the output
                    new_lines.append(line)
                
                # If we didn't find the pointcloud_saver section, add it at the end
                if not pointcloud_saver_section_found:
                    new_lines.append("\npointcloud_saver_node:\n")
                    new_lines.append("  ros__parameters:\n")
                    for param_name in changed_params:
                        if param_name == 'global_frame':
                            new_lines.append(f"    global_frame: {self.global_frame}\n")
                        elif param_name == 'lidar_frame':
                            new_lines.append(f"    lidar_frame: {self.lidar_frame}\n")
                        # ... (repeat for all parameters)
                        elif param_name == 'file_prefix':
                            # Handle empty string properly with quotes
                            prefix = '""' if self.file_prefix == "" else self.file_prefix
                            new_lines.append(f"    file_prefix: {prefix}\n")
                        elif param_name == 'file_index':
                            new_lines.append(f"    file_index: {self.file_index}\n")
                
                # Write the updated content back to the file
                with open(params_file_path, 'w') as file:
                    file.writelines(new_lines)
                
                self.get_logger().info(f'\033[1;33mParameters saved to bringup package: {params_file_path}\033[0m')
                
            except Exception as e:
                self.get_logger().error(f'Error saving parameters to bringup package: {e}')
        
        except Exception as e:
            self.get_logger().error(f'Error saving parameters: {e}')
    
    def shutdown_callback(self):
        self.get_logger().info('\033[1;33mShutting down, saving parameters...\033[0m')
        self.save_parameters_to_yaml()


def main(args=None):
    rclpy.init(args=args)
    node = KittiPointCloudSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

