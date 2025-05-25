from setuptools import setup
import os
from glob import glob

package_name = 'sfa3d_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ori',
    maintainer_email='ori@todo.todo',
    description='ROS2 wrapper for SFA3D',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sfa3d_node = sfa3d_ros2.sfa3d_node:main',
            'kitti_player_node = sfa3d_ros2.kitti_player_node:main',
            'counter_pub = sfa3d_ros2.counter_pub:main'
        ],
    },
)
