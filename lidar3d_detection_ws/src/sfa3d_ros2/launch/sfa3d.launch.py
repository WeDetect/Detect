from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sfa3d_ros2',
            executable='sfa3d_node',
            name='sfa3d_node',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        )
    ])
