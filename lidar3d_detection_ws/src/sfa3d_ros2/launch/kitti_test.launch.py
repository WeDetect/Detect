from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_prefix

def generate_launch_description():
    # Get the package path
    package_path = get_package_prefix('sfa3d_ros2')
    
    # Get paths
    kitti_path = os.path.join(
        os.path.expanduser('~'),
        'project_x/sfa3d_ws/src/SFA3D/dataset/kitti/training/velodyne'
    )
    rviz_config = os.path.join(
        os.path.dirname(package_path),
        'share/sfa3d_ros2/rviz/kitti_view.rviz'
    )
    
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='innoviz_transform',
            arguments=['0', '0', '0', '0', '0', '0', 'velodyne', 'innoviz'],
            parameters=[{
                'x_offset': -5.5,
                'z_offset': 8.0,
                'pitch': 0.3,
            }]
        ),
        Node(
            package='sfa3d_ros2',
            executable='kitti_player_node',
            name='kitti_player',
            parameters=[{
                'velodyne_path': kitti_path,
                'publish_rate': 0.286
            }]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config]
        )
    ]) 