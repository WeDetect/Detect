from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lidar_detection_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['package.xml']),
        ('share/' + package_name, ['package.xml']),
        # הוסף כאן קבצי launch אם יש לך
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='ori@cogniteam.com',
    description='KITTI format pointcloud saver for LiDAR data',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_detection_node = lidar_detection_pkg.lidar_detection_node:main',
            'bin_publisher_node = lidar_detection_pkg.bin_publisher_node:main',
        ],
    },
)
