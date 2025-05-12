

# LiDAR 3D Detection Project

This project provides a complete solution for labeling LiDAR point clouds, training detection models, and deploying object detection systems for 3D perception.

## Table of Contents
- [Setup](#setup)
- [Point Cloud Annotation](#point-cloud-annotation)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

## Setup

### Prerequisites
- ROS2 Humble Desktop
- Docker
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/WeDetect/Detect
```

2. Create required data directories:
```bash
cd Detect
mkdir -p data/innoviz data/labels
```

3. Run the Docker container:
```bash
lidar3d_detection='docker run -it --name lidar3d_detection_container \
--privileged --ipc host --pid host \
--device /dev:/dev \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /path/to/your/cloned/repo/lidar3d_detection_ws:/lidar3d_detection_ws \
-w /lidar3d_detection_ws \
oridaniel100/lidar3d_detection_img:latest'
```

4. For subsequent terminal sessions, use:
```bash
docker start lidar3d_detection_container
docker exec -it lidar3d_detection_container bash
```

## Point Cloud Annotation

### Terminal Setup (5 terminals required)

**Terminal 1 (ROS Launch):**
```bash
cd lidar3d_detection_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch lidar_solution_bringup lidar_solution_bringup.launch.py
```

**Terminal 2 (ROS Bag Playback):**
```bash
cd Bags
ros2 bag play [bag_name] -l
```

**Terminal 3 (Parameter Configuration):**
```bash
ros2 run rqt_reconfigure rqt_reconfigure
```
- Select `lidar_tf_publisher_node` in the left panel to configure LiDAR position parameters

**Terminal 4 (Point Cloud Visualization):**
```bash
rviz2
```
- Set Fixed Frame to `world`
- Add a PointCloud2 display and select the `/pointcloud_saver/filtered_points` topic
- Adjust parameters in rqt_reconfigure to align point cloud with ground plane (height 0)

**Terminal 5 (Point Cloud Saving):**
```bash
ros2 topic pub --once /pointcloud_saver/save_trigger std_msgs/msg/Bool data:\ true
```
- This saves the current filtered point cloud as a `.bin` file to the configured output directory

### KITTI Annotation Format

When labeling point clouds, use the `kitti_untransformed` format with the following parameters:

**Class dimensions (height, width, length):**
- Car: 2.36 m × 2.36 m × 4.36 m
- Person: 2.0 m × 1.20 m × 1.20 m
- Motorcycle/Bicycle: 2.0 m × 1.5 m × 2.5 m

**KITTI format:**
```
object_type truncation occlusion alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z yaw
```

## Model Training

The system supports different training approaches. Choose the one that best fits your requirements:

### Training Commands

**Regular model without augmentations:**
```bash
python3 train/models/train.py --augmentations False --transfer_learning False
```

**Regular model with augmentations:**
```bash
python3 train/models/train.py --augmentations True --transfer_learning False
```

**Transfer learning model without augmentations:**
```bash
python3 train/models/train.py --augmentations False --transfer_learning True --unfreeze_layers 20
```

**Transfer learning model with augmentations:**
```bash
python3 train/models/train.py --augmentations True --transfer_learning True --unfreeze_layers 20
```

**Continue training from existing model without augmentations:**
```bash
python3 train/models/train.py --augmentations False --continue_from /lidar3d_detection_ws/train/dataset/output/bev_transfer_final.pt
```

**Continue training from existing model with augmentations:**
```bash
python3 train/models/train.py --augmentations True --continue_from /lidar3d_detection_ws/train/dataset/output/bev_transfer_final.pt
```

### Training Configuration

Training parameters and results can be found in:
- `/lidar3d_detection_ws/train/dataset/output/[model_name]/args.yaml`
- `/lidar3d_detection_ws/train/dataset/output/[model_name]/results.csv`

## Project Structure

### train/data_processing
Handles dataset creation, preprocessing and augmentation for BEV images:
- `dataset.py`: Implements the BEV dataset class for training
- `preprocessing.py`: Functions for point cloud processing
- `augmentation.py`: Implements data augmentation techniques

### train/models
Contains model definitions and loss functions:
- `train.py`: Main training script
- `loss.py`: YOLOLoss implementation with CIoU, DIoU and GIoU options

### src/lidar_solution_bringup
Main package that coordinates the LiDAR modules:
- `config/parameters.yaml`: Central configuration for all LiDAR modules
- `launch/lidar_solution_bringup.launch.py`: Main launch file

### src/lidar_tf_position_publisher
Publishes the LiDAR TF transform:
- Allows dynamic reconfiguration of LiDAR position/orientation
- Saves parameters to YAML files

### src/pointcloud_saver
Saves and filters point clouds:
- Converts ROS PointCloud2 messages to KITTI format
- Applies configurable spatial filters
- Triggers point cloud saving via ROS topic

## Configuration

### LiDAR Positioning and Filtering

All parameters are in:
```
/lidar3d_detection_ws/src/lidar_solution_bringup/config/parameters.yaml
```

These parameters can be modified live using `rqt_reconfigure` and are automatically saved.

**Note:** There is a known issue with the `lidar_pitch` parameter which resets to 0.1 but should be 0.15 - verify this setting before saving point clouds.

### Requirements

The project requires the following Python packages:
```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
opencv-python>=4.4.0
pyyaml>=5.3.1
scikit-learn>=0.23.2
matplotlib>=3.3.0
ultralytics
open3d
tqdm
```
