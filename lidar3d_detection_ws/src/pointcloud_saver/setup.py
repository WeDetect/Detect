from setuptools import find_packages, setup

package_name = 'pointcloud_saver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'pointcloud_saver_node = pointcloud_saver.pointcloud_saver_node:main',
        ],
    },
)
