# LiDAR 3D Detection Project

![Main Image](images/WhatsApp%20Image%202025-05-25%20at%2012.20.22.jpeg)

This project provides a complete solution for labeling LiDAR point clouds, training detection models, and deploying object detection systems for 3D perception.

## Table of Contents
- [Setup](#setup)
- [Point Cloud Annotation](#point-cloud-annotation)
- [Model Training](#model-training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Model Process](#model-process)

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

3. Download the Docker image:
```bash
docker pull oridaniel100/lidar3d_detection_img:latest
```

4. Run the Docker container:
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

5. For subsequent terminal sessions, use:
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

## Model Process

### WeDetect – 3D Object Detection System Based on LiDAR

1. **Introduction**
   As part of an advanced project in artificial intelligence, a unique real-time object detection system was developed using 3D LiDAR technology combined with an advanced visual detection model of the YOLO type.

   The system converts scan data into a Bird's Eye View (BEV), processes the image, detects objects within it, and then calculates their full position in space (X, Y, Z) with high accuracy.

   The system is designed for various uses, including:
   - Autonomous vehicle systems
   - Urban infrastructure monitoring
   - Intersection safety systems
   - Advanced robotics

2. **Detection Process – From LiDAR to Final Result**
   Our system is built from several sequential stages:
   - 3D LiDAR scanning generates a point cloud file.
   - Preprocessing converts the data into a 2D Bird's Eye View (BEV) image.
   - This image is fed into a YOLO model specially adapted for this input format.
   - Postprocessing converts the detection results into precise spatial positions (including depth, height, and direction calculations).

   The system is trained to detect over 95% of objects in the scene, maintaining precise positioning and tracking capability for each object.

3. **Advantages of LiDAR Compared to a Regular Camera**
   - Operates in varying conditions: The system functions properly at night, in rain, fog, and is unaffected by shadows, sun glare, or high contrast.
   - High geometric accuracy: A full 3D scene allows for a true understanding of the environment, including height, volume, and distance between objects.
   - Real-time 3D mapping: Can calculate distances, identify the height of bridges or signs, and alert to relevant dangers – for example, a pedestrian crossing in a dangerous location.
   - Wide integration: The system can be integrated into any technology – Web, ROS, edge applications, cloud systems, and more.

4. **Disadvantages and Challenges**
   - Heavy computational load: A LiDAR file contains a vast amount of data, sometimes several megabytes per single frame, requiring stronger computational resources compared to regular image processing.
   - Slower speed: Compared to camera-based systems operating at high FPS, our system operates at a relatively slow pace but yields higher quality data.

5. **System Detection Configuration**
   - Horizontal range (sideways): Up to 15 meters in each direction
   - Depth range (forward distance): Up to 30 meters forward
   - Height range: From 2 meters below the zero point, up to a height of 4.5 meters

   The model was designed and trained on a LiDAR sensor installed at a height of 2.2 meters – a common height for placement on a car roof, street pole, traffic light, smart camera, and more.

6. **Sensor and System Alignment Before Use**
   To ensure the system's success in real-time, it is necessary to perform prior alignment of the LiDAR:
   - Correct Roll, Pitch, Yaw angles according to TF data.
   - Maintain a leveled floor at height Z=0.
   - Operate based on a fixed world frame.

   During training, we ensured that all scenes were reset so that the floor appears in a uniform color (black), while objects (people, vehicles, bicycles) stand out above it – ideally suited for machine learning.

   Additionally, there is a Node operating in ROS that performs the necessary TF corrections – it must be activated before running the model.

7. **Information on Innoviz Sensor (Example Model)**
   The LiDAR sensor we used comes from Innoviz and includes:
   - Horizontal field of view: Approximately 120 degrees
   - Vertical field of view: Approximately 26 degrees (depending on the model)
   - Scanning accuracy: 0.05 cm
   - Scanning frequency: 5-25 hz

   (It is recommended to complete the specific details from the product page on the Innoviz website: [Innoviz](https://www.innoviz.tech/))

8. **Source Code and System Operation**
   The system code is located in the following repository: [GitHub Repository](https://github.com/WeDetect/Detect)

   The repo includes:
   - Preprocessing code for converting LiDAR to BEV
   - Adapted YOLO model
   - Postprocessing code for X, Y, Z positioning
   - Image examples
   - Operating instructions on ROS2 and Docker
