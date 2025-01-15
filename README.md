# 3D Object Detection ROS2 Package

## Overview

This ROS2 package implements 3D object detection using LiDAR point clouds and provides Bird's Eye View (BEV) visualization. The system processes point cloud data, detects objects, and generates KITTI format labels.

## 🚀 Features

- Real-time LiDAR point cloud processing
- Bird's Eye View (BEV) visualization
- KITTI format label generation
- Object detection visualization using markers
- Support for multiple object classes:
  - Cars
  - Pedestrians
  - Cyclists
  - Vans
  - Trucks

## 📋 Prerequisites

- ROS2 (Humble/Foxy)
- Python 3.8+
- NumPy
- OpenCV
- Point Cloud Library (PCL)

## 🛠️ Installation

1. Clone the repository:
   bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   git clone <repository-url>

## 🎮 Usage

1. Start the KITTI player node:

```bash
ros2 run od_detection kitti_player_node
```

2. Start the point cloud processing node:

```bash
ros2 run od_detection cloud_subscriber_node
```

## 📊 Data Format

### Input

- Point Cloud: `/ipcl_grabber` topic (sensor_msgs/PointCloud2)
- Detection Markers: `/detection_markers` topic (visualization_msgs/MarkerArray)

### Output

- BEV Images: Saved in `data/images/`
- Labels: KITTI format, saved in `data/label_2/`

## 📝 KITTI Label Format

Each line in the label files represents one object with the following format:
