#!check
import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, LogInfo
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package directory
    package_dir = get_package_share_directory('lidar_solution_bringup')
    
    # Load parameters from YAML file
    params_file_path = os.path.join(package_dir, 'config', 'parameters.yaml')
    
    # Print the full path of the parameters file
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    print("\033[1;32mLoading parameters from: " + params_file_path + "\033[0m")
    print("\033[1;32m" + "=" * 80 + "\033[0m")
    
    # Create the launch description with the nodes
    return LaunchDescription([
        
        # First node starts immediately
        Node(
            package='lidar_tf_position_publisher',
            executable='lidar_tf_position_publisher_node',
            name='lidar_tf_position_publisher_node',
            output='screen',
            parameters=[params_file_path]
        ),
        # Node(
        #     package='pointcloud_saver',
        #     executable='pointcloud_saver_node',
        #     name='pointcloud_saver_node',
        #     output='screen',
        #     parameters=[params_file_path]
        # ),
      
    ])
