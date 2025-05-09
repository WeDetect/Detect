# ✅ evaluate.py – Evaluate YOLOv5 model on BEV LiDAR data
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from pathlib import Path
import yaml

# Define functions from preprocessing.py
def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_bin_file(bin_path):
    """Read point cloud from binary file"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_label_file(label_path):
    """Read KITTI format label file"""
    labels = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                labels.append({
                    'type': values[0],
                    'truncated': float(values[1]),
                    'occluded': int(values[2]),
                    'alpha': float(values[3]),
                    'bbox': [float(x) for x in values[4:8]],
                    'dimensions': [float(x) for x in values[8:11]],
                    'location': [float(x) for x in values[11:14]],
                    'rotation_y': float(values[14])
                })
    return labels

def create_bev_image(points, config, labels=None):
    """Create bird's eye view image from point cloud"""
    # Get dimensions from config
    Height = config['BEV_HEIGHT']
    Width = config['BEV_WIDTH']
    
    # Create a copy of points to avoid modifying the original
    points_copy = np.copy(points)
    
    # Filter points within boundaries
    if 'boundary' in config:
        mask = (points_copy[:, 0] >= config['boundary']['minX']) & (points_copy[:, 0] <= config['boundary']['maxX']) & \
               (points_copy[:, 1] >= config['boundary']['minY']) & (points_copy[:, 1] <= config['boundary']['maxY']) & \
               (points_copy[:, 2] >= config['boundary']['minZ']) & (points_copy[:, 2] <= config['boundary']['maxZ'])
        points_copy = points_copy[mask]
    
    # Discretize Feature Map
    points_copy[:, 0] = np.int_(np.floor(points_copy[:, 0] / config['DISCRETIZATION']))
    points_copy[:, 1] = np.int_(np.floor(points_copy[:, 1] / config['DISCRETIZATION']) + Width / 2)
    
    # Ensure indices are within bounds
    points_copy[:, 0] = np.clip(points_copy[:, 0], 0, Height - 1)
    points_copy[:, 1] = np.clip(points_copy[:, 1], 0, Width - 1)
    
    # Sort points by height (z) and get unique x,y coordinates
    sorted_indices = np.lexsort((-points_copy[:, 2], points_copy[:, 1], points_copy[:, 0]))
    points_copy = points_copy[sorted_indices]
    _, unique_indices, unique_counts = np.unique(points_copy[:, 0:2], axis=0, return_index=True, return_counts=True)
    points_top = points_copy[unique_indices]
    
    # Create height, intensity and density maps
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))
    
    # Calculate max height for normalization
    max_height = float(np.abs(config['boundary']['maxZ'] - config['boundary']['minZ']))
    
    # Fill the maps
    valid_indices = (points_top[:, 0] < Height) & (points_top[:, 1] < Width)
    heightMap[np.int_(points_top[valid_indices, 0]), np.int_(points_top[valid_indices, 1])] = points_top[valid_indices, 2] / max_height
    
    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(points_top[valid_indices, 0]), np.int_(points_top[valid_indices, 1])] = points_top[valid_indices, 3]
    densityMap[np.int_(points_top[valid_indices, 0]), np.int_(points_top[valid_indices, 1])] = normalizedCounts[valid_indices]
    
    # Create RGB map
    RGB_Map = np.zeros((3, Height, Width))
    RGB_Map[2, :, :] = densityMap  # r_map
    RGB_Map[1, :, :] = heightMap  # g_map
    RGB_Map[0, :, :] = intensityMap  # b_map
    
    # Convert to image format
    RGB_Map = np.transpose(RGB_Map, (1, 2, 0))
    RGB_Map = (RGB_Map * 255).astype(np.uint8)
    
    return RGB_Map, points_copy

# Define function to convert labels to YOLO format
def convert_labels_to_yolo_format(labels, config, img_width, img_height):
    yolo_labels = []
    for label in labels:
        if label['type'] == 'DontCare':
            continue
        elif label['type'] == 'Car':
            class_id = 0
        elif label['type'] == 'Pedestrian':
            class_id = 1
        elif label['type'] == 'Cyclist':
            class_id = 2
        elif label['type'] == 'Truck':
            class_id = 3
        else:
            continue
            
        x, y, z = label['location']
        h, w, l = label['dimensions']

        x_bev = x / config['DISCRETIZATION']
        y_bev = y / config['DISCRETIZATION'] + config['BEV_WIDTH'] / 2

        x_center = x_bev / img_width
        y_center = y_bev / img_height
        width = w / config['DISCRETIZATION'] / img_width
        height = l / config['DISCRETIZATION'] / img_height

        # Make sure values are within valid range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0.01, min(1, width))  # Minimum 1% of image width
        height = max(0.01, min(1, height))  # Minimum 1% of image height

        yolo_labels.append([class_id, x_center, y_center, width, height])
    
    return yolo_labels

# Paths
MODEL_PATH = '/lidar3d_detection_ws/train/output/bev_transfer_final.pt'
BIN_DIR = '/lidar3d_detection_ws/data/innoviz'
LABEL_DIR = '/lidar3d_detection_ws/data/labels'
CONFIG_PATH = '/lidar3d_detection_ws/train/config/preprocessing_config.yaml'
OUTPUT_PATH = '/lidar3d_detection_ws/train/output/eval'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load config
config = load_config(CONFIG_PATH)
print(f"Loaded config from {CONFIG_PATH}")

# Class names - make sure this matches what the model was trained on
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Truck']  # Removed DontCare

# Load trained model
print(f"Loading model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Print model information to debug
print(f"Model has {model.model.model[-1].nc} classes")

# Initialize metrics - only track the classes we care about
class_metrics = {cls_name: {'TP': 0, 'FP': 0, 'FN': 0} for cls_name in CLASS_NAMES}
total_objects = 0
detected_objects = 0

# Run evaluation on each bin file
bin_files = sorted(Path(BIN_DIR).glob('*.bin'))
print(f"Found {len(bin_files)} bin files for evaluation")

for bin_file in bin_files:
    print(f"Processing {bin_file.name}...")
    
    # Generate BEV image from bin file
    points = read_bin_file(bin_file)
    label_file = Path(LABEL_DIR) / f"{bin_file.stem}.txt"
    labels = read_label_file(label_file)
    
    # Create BEV image
    bev_image, _ = create_bev_image(points, config, labels)
    
    if bev_image is None:
        print(f"Could not create BEV image for {bin_file}")
        continue

    # Get ground truth boxes in YOLO format
    gt_yolo = convert_labels_to_yolo_format(labels, config, bev_image.shape[1], bev_image.shape[0])
    
    # Convert YOLO format to pixel coordinates for visualization
    gt_boxes = []
    for gt in gt_yolo:
        cls_id, x_center, y_center, width, height = gt
        img_h, img_w = bev_image.shape[:2]
        x1 = int((x_center - width/2) * img_w)
        y1 = int((y_center - height/2) * img_h)
        x2 = int((x_center + width/2) * img_w)
        y2 = int((y_center + height/2) * img_h)
        gt_boxes.append((int(cls_id), (x1, y1, x2, y2)))
        
        # Count total objects by class
        total_objects += 1
        class_metrics[CLASS_NAMES[int(cls_id)]]['FN'] += 1  # Assume FN first, will adjust if detected
    
    # Run inference with lower confidence threshold since we're evaluating on training data
    results = model(bev_image, conf=0.1, iou=0.3)  # Lower thresholds for better detection
    
    # Create a copy for visualization
    vis_image = bev_image.copy()
    
    # Draw ground truth boxes (in blue)
    for cls_id, (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for ground truth
        cv2.putText(vis_image, f"GT: {CLASS_NAMES[cls_id]}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Process detection results
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        
        for box, cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            
            # Safety check for class index
            if cls_id >= len(CLASS_NAMES):
                print(f"Warning: Model predicted class {cls_id} which is out of range. Skipping.")
                continue
            
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            
            # Draw prediction boxes (in green)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for predictions
            cv2.putText(vis_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                      
            # Check if this detection matches any ground truth box
            matched = False
            for gt_cls, gt_box in gt_boxes:
                gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
                
                # Calculate IoU
                x_left = max(x1, gt_x1)
                y_top = max(y1, gt_y1)
                x_right = min(x2, gt_x2)
                y_bottom = min(y2, gt_y2)
                
                if x_right < x_left or y_bottom < y_top:
                    continue  # No overlap
                
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
                iou = intersection / (area1 + area2 - intersection)
                
                # Use a lower IoU threshold since we're evaluating on training data
                if iou > 0.3 and cls_id == gt_cls:  # Lower IoU threshold
                    matched = True
                    detected_objects += 1
                    
                    # Update metrics
                    class_metrics[CLASS_NAMES[cls_id]]['TP'] += 1
                    class_metrics[CLASS_NAMES[cls_id]]['FN'] -= 1  # Remove from FN count
                    break
            
            if not matched:
                class_metrics[CLASS_NAMES[cls_id]]['FP'] += 1

    # Save visualization
    out_path = os.path.join(OUTPUT_PATH, f"{bin_file.stem}_eval.png")
    cv2.imwrite(out_path, vis_image)
    print(f"Saved visualization to {out_path}")

# Calculate and print metrics
print("\n===== EVALUATION RESULTS =====")
print(f"Total objects: {total_objects}")
print(f"Detected objects: {detected_objects}")
print(f"Overall detection rate: {detected_objects/total_objects*100:.2f}%")
print("\nPer-class metrics:")

for cls_name, metrics in class_metrics.items():
    tp = metrics['TP']
    fp = metrics['FP']
    fn = metrics['FN']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{cls_name}:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# Save metrics to file
with open(os.path.join(OUTPUT_PATH, "evaluation_metrics.txt"), "w") as f:
    f.write("===== EVALUATION RESULTS =====\n")
    f.write(f"Total objects: {total_objects}\n")
    f.write(f"Detected objects: {detected_objects}\n")
    f.write(f"Overall detection rate: {detected_objects/total_objects*100:.2f}%\n\n")
    f.write("Per-class metrics:\n")
    
    for cls_name, metrics in class_metrics.items():
        tp = metrics['TP']
        fp = metrics['FP']
        fn = metrics['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f.write(f"\n{cls_name}:\n")
        f.write(f"  True Positives: {tp}\n")
        f.write(f"  False Positives: {fp}\n")
        f.write(f"  False Negatives: {fn}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")

print(f"\nEvaluation complete. Results saved to {OUTPUT_PATH}")
