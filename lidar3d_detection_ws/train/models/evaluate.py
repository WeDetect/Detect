# Add import for OS environment variables at the top
import os

# Set headless mode for matplotlib and Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""

# ✅ evaluate.py – Evaluate YOLOv5 model on BEV LiDAR data
from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from pathlib import Path
import yaml
import argparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# Constants
DEFAULT_CONFIG = {
    'model_path': '/lidar3d_detection_ws/train/output/bev_transfer_final.pt',
    'bin_dir': '/lidar3d_detection_ws/data/innoviz',
    'label_dir': '/lidar3d_detection_ws/data/labels',
    'config_path': '/lidar3d_detection_ws/train/config/preprocessing_config.yaml',
    'output_dir': '/lidar3d_detection_ws/train/output/eval',
    'num_samples': 5,
    'conf_threshold': 0.25,
    'iou_threshold': 0.45
}

# Class names
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Truck']

def convert_labels_to_boxes(labels, config, img_width, img_height):
    """
    Convert KITTI format labels to bounding boxes
    
    Args:
        labels: Labels in KITTI format
        config: Preprocessing configuration
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of (class_id, box) tuples where box is (x1, y1, x2, y2)
    """
    boxes = []
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

        # Convert to BEV coordinates
        x_bev = x / config['DISCRETIZATION']
        y_bev = y / config['DISCRETIZATION'] + img_width / 2

        # Calculate width and height in pixels
        width_px = w / config['DISCRETIZATION']
        height_px = l / config['DISCRETIZATION']
        
        # Calculate box coordinates
        x1 = int(x_bev - width_px/2)
        y1 = int(y_bev - height_px/2)
        x2 = int(x_bev + width_px/2)
        y2 = int(y_bev + height_px/2)
        
        # Make sure box is within image bounds
        x1 = max(0, min(img_width-1, x1))
        y1 = max(0, min(img_height-1, y1))
        x2 = max(0, min(img_width-1, x2))
        y2 = max(0, min(img_height-1, y2))
        
        # Only add box if it has positive area
        if x2 > x1 and y2 > y1:
            boxes.append((class_id, (x1, y1, x2, y2)))
    
    return boxes

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    
    Args:
        box1: First box (x1, y1, x2, y2)
        box2: Second box (x1, y1, x2, y2)
        
    Returns:
        IoU score
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0
    
    return intersection_area / union_area

def evaluate_model(model_path, bin_dir, label_dir, config_path, output_dir, 
                  num_samples=5, conf_threshold=0.25, iou_threshold=0.45, save_images=True):
    """
    Evaluate YOLO model on BEV LiDAR data
    
    Args:
        model_path: Path to trained model file
        bin_dir: Directory with .bin files
        label_dir: Directory with label .txt files
        config_path: Path to preprocessing config YAML
        output_dir: Directory to save evaluation results
        num_samples: Number of samples to evaluate (randomly selected)
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for matching boxes
        save_images: Whether to save visualization of detection results
    
    Returns:
        Evaluation metrics dictionary
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Load config
    config = load_config(config_path)
    
    # Get all bin files
    bin_files = list(Path(bin_dir).glob("*.bin"))
    
    # Randomly select files if too many
    if len(bin_files) > num_samples:
        bin_files = random.sample(bin_files, num_samples)
    
    # Initialize metrics
    class_metrics = {cls_name: {'TP': 0, 'FP': 0, 'FN': 0} for cls_name in CLASS_NAMES}
    total_objects = 0
    detected_objects = 0
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate each file
    for i, bin_file in enumerate(tqdm(bin_files, desc="Evaluating model")):
        print(f"\nProcessing {bin_file.name}...")
        
        # Get corresponding label file
        label_file = Path(label_dir) / f"{bin_file.stem}.txt"
        if not label_file.exists():
            print(f"Label file {label_file} not found, skipping")
            continue
        
        # Load point cloud and labels
        points = read_bin_file(bin_file)
        labels = read_label_file(label_file)
        
        # Create BEV image without annotations for model input
        bev_image, _ = create_bev_image(points, config)
        
        # Convert labels to boxes for evaluation
        gt_boxes = convert_labels_to_boxes(labels, config, bev_image.shape[1], bev_image.shape[0])
        
        # Count total objects
        total_objects += len(gt_boxes)
        for cls_id, _ in gt_boxes:
            class_metrics[CLASS_NAMES[cls_id]]['FN'] += 1  # Assume FN first
        
        # Run inference
        results = model(bev_image, conf=conf_threshold, iou=iou_threshold)
        
        # Extract predictions
        pred_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Convert box to integer coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                pred_boxes.append((cls_id, (x1, y1, x2, y2), conf))
        
        # Match predictions with ground truth
        gt_matched = [False] * len(gt_boxes)
        
        for p_idx, (p_cls_id, p_box, p_conf) in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth box
            for gt_idx, (gt_cls_id, gt_box) in enumerate(gt_boxes):
                # Only match with same class
                if gt_cls_id != p_cls_id:
                    continue
                
                # Check if already matched
                if gt_matched[gt_idx]:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(gt_box, p_box)
                
                # Update best match if better
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If found a match, it's a true positive
            if best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                detected_objects += 1
                class_metrics[CLASS_NAMES[p_cls_id]]['TP'] += 1
                class_metrics[CLASS_NAMES[p_cls_id]]['FN'] -= 1  # Remove FN assumption
            else:
                # No match, it's a false positive
                class_metrics[CLASS_NAMES[p_cls_id]]['FP'] += 1
        
        # Save visualization if requested
        if save_images:
            # Create visualization of detection results
            vis_image = visualize_detection_results(bev_image.copy(), gt_boxes, pred_boxes)
            
            # Save visualization
            output_path = os.path.join(output_dir, f"{bin_file.stem}_detection.png")
            cv2.imwrite(output_path, vis_image)
            
            print(f"Saved visualization to {output_path}")
    
    # Calculate metrics
    metrics = {
        'total_objects': total_objects,
        'detected_objects': detected_objects,
        'detection_rate': detected_objects / total_objects if total_objects > 0 else 0,
        'class_metrics': {}
    }
    
    # Calculate per-class metrics
    for cls_name, counts in class_metrics.items():
        tp = counts['TP']
        fp = counts['FP']
        fn = counts['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['class_metrics'][cls_name] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Print metrics
    print("\n===== EVALUATION RESULTS =====")
    print(f"Total objects: {total_objects}")
    print(f"Detected objects: {detected_objects}")
    print(f"Overall detection rate: {metrics['detection_rate']*100:.2f}%")
    print("\nPer-class metrics:")
    
    for cls_name, cls_metrics in metrics['class_metrics'].items():
        print(f"\n{cls_name}:")
        print(f"  True Positives: {cls_metrics['TP']}")
        print(f"  False Positives: {cls_metrics['FP']}")
        print(f"  False Negatives: {cls_metrics['FN']}")
        print(f"  Precision: {cls_metrics['precision']:.4f}")
        print(f"  Recall: {cls_metrics['recall']:.4f}")
        print(f"  F1 Score: {cls_metrics['f1']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write("===== EVALUATION RESULTS =====\n")
        f.write(f"Total objects: {total_objects}\n")
        f.write(f"Detected objects: {detected_objects}\n")
        f.write(f"Overall detection rate: {metrics['detection_rate']*100:.2f}%\n")
        f.write("\nPer-class metrics:\n")
        
        for cls_name, cls_metrics in metrics['class_metrics'].items():
            f.write(f"\n{cls_name}:\n")
            f.write(f"  True Positives: {cls_metrics['TP']}\n")
            f.write(f"  False Positives: {cls_metrics['FP']}\n")
            f.write(f"  False Negatives: {cls_metrics['FN']}\n")
            f.write(f"  Precision: {cls_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {cls_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {cls_metrics['f1']:.4f}\n")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on BEV LiDAR data")
    parser.add_argument("--model", default=DEFAULT_CONFIG['model_path'], help="Path to trained model file")
    parser.add_argument("--bin_dir", default=DEFAULT_CONFIG['bin_dir'], help="Directory with .bin files")
    parser.add_argument("--label_dir", default=DEFAULT_CONFIG['label_dir'], help="Directory with label .txt files")
    parser.add_argument("--config", default=DEFAULT_CONFIG['config_path'], help="Path to preprocessing config YAML")
    parser.add_argument("--output", default=DEFAULT_CONFIG['output_dir'], help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_CONFIG['num_samples'], help="Number of samples to evaluate")
    parser.add_argument("--conf_threshold", type=float, default=DEFAULT_CONFIG['conf_threshold'], help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=DEFAULT_CONFIG['iou_threshold'], help="IoU threshold")
    parser.add_argument("--save_images", action="store_true", help="Whether to save visualization of detection results")
    
    args = parser.parse_args()
    
    print(f"Evaluating model {args.model} on {args.num_samples} samples")
    
    metrics = evaluate_model(
        model_path=args.model,
        bin_dir=args.bin_dir,
        label_dir=args.label_dir,
        config_path=args.config,
        output_dir=args.output,
        num_samples=args.num_samples,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        save_images=args.save_images
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
