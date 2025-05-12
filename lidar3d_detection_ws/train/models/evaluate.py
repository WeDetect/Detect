# Add import for OS environment variables at the top
import os
import sys

# Fix import path with absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
parent_dir = os.path.dirname(current_dir)  # Get parent directory (train/)
project_root = os.path.dirname(parent_dir)  # Get project root

print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Project root: {project_root}")

# Add both directories to the Python path
sys.path.insert(0, parent_dir)  # Add train/ directory
sys.path.insert(0, project_root)  # Add project root directory

# Now print the path to verify
print("Python path:")
for p in sys.path:
    print(f"  - {p}")

# Import the exact same functions from preprocessing_0 as in train.py
try:
    from data_processing.preproccesing_0 import PointCloudProcessor, convert_labels_to_yolo_format
    print("✅ Successfully imported PointCloudProcessor")
except Exception as e:
    print(f"❌ Import error: {e}")
    # List files in the data_processing directory if it exists
    data_processing_dir = os.path.join(parent_dir, "data_processing")
    if os.path.exists(data_processing_dir):
        print(f"Files in {data_processing_dir}:")
        for file in os.listdir(data_processing_dir):
            print(f"  - {file}")
    else:
        print(f"Directory not found: {data_processing_dir}")

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
from data_processing.preprocessing import load_config, read_bin_file, read_label_file

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

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: (x1, y1, x2, y2) format
        box2: (x1, y1, x2, y2) format
    
    Returns:
        IoU value
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes overlap
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate area of intersection
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate IoU
    iou = area_i / (area1 + area2 - area_i)
    
    return iou

def yolo_to_bbox(yolo_box, img_width, img_height):
    """
    Convert YOLO format box to (x1,y1,x2,y2) format
    
    Args:
        yolo_box: [class_id, x_center, y_center, width, height]
        img_width: Image width
        img_height: Image height
    
    Returns:
        (x1, y1, x2, y2) tuple
    """
    class_id, x_center, y_center, width, height = yolo_box
    
    # Denormalize
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Calculate corners
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return (x1, y1, x2, y2)

def visualize_detection_results(bev_image, gt_boxes, pred_boxes):
    """
    Visualize detection results with ground truth and predictions
    using PointCloudProcessor to draw boxes exactly like in preprocessing_0.py
    
    Args:
        bev_image: BEV image
        gt_boxes: List of ground truth boxes [(class_id, bbox, conf), ...]
        pred_boxes: List of predicted boxes [(class_id, bbox, conf), ...]
    
    Returns:
        Visualization image
    """
    # Create a processor object for visualization
    processor = PointCloudProcessor()
    
    # Copy images for drawing
    gt_vis = bev_image.copy()
    pred_vis = bev_image.copy()
    
    # Draw ground truth boxes
    for cls_id, bbox, _ in gt_boxes:
        # Convert bbox format back to corners for visualization
        x1, y1, x2, y2 = bbox
        
        # Calculate center and corners in BEV format
        center_bev = ((x1 + x2) // 2, (y1 + y2) // 2)
        corners_bev = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # Use same draw method as in training
        obj_type = CLASS_NAMES[cls_id]
        gt_vis = processor.draw_filled_box_on_bev(gt_vis, corners_bev, center_bev, obj_type)
    
    # Draw predicted boxes
    for cls_id, bbox, conf in pred_boxes:
        # Convert bbox format back to corners for visualization
        x1, y1, x2, y2 = bbox
        
        # Calculate center and corners in BEV format
        center_bev = ((x1 + x2) // 2, (y1 + y2) // 2)
        corners_bev = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        # Use same draw method as in training
        obj_type = CLASS_NAMES[cls_id]
        pred_vis = processor.draw_filled_box_on_bev(pred_vis, corners_bev, center_bev, obj_type)
        
        # Add confidence score
        cv2.putText(pred_vis, f"{conf:.2f}", 
                  (center_bev[0], center_bev[1] + 15), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add titles
    title_height = 40
    gt_title = "Ground Truth"
    pred_title = "Predictions"
    
    # Create title bars
    gt_title_bar = np.ones((title_height, gt_vis.shape[1], 3), dtype=np.uint8) * 255
    pred_title_bar = np.ones((title_height, pred_vis.shape[1], 3), dtype=np.uint8) * 255
    
    # Add text to title bars
    cv2.putText(gt_title_bar, gt_title, 
              (gt_title_bar.shape[1]//2 - 80, title_height//2 + 5), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(pred_title_bar, pred_title, 
              (pred_title_bar.shape[1]//2 - 60, title_height//2 + 5), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Combine with title bars
    gt_vis_with_title = np.vstack([gt_title_bar, gt_vis])
    pred_vis_with_title = np.vstack([pred_title_bar, pred_vis])
    
    # Create separator
    separator_width = 20
    separator = np.ones((max(gt_vis_with_title.shape[0], pred_vis_with_title.shape[0]), 
                        separator_width, 3), dtype=np.uint8) * 255
    
    # Combine horizontally
    combined_image = np.hstack([gt_vis_with_title, separator, pred_vis_with_title])
    
    # Add overall title
    overall_title = f"BEV LiDAR Detection Results"
    cv2.putText(combined_image, overall_title, 
              (combined_image.shape[1]//2 - 200, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return combined_image

def create_processor():
    """Create and configure a PointCloudProcessor instance"""
    config_path = "/lidar3d_detection_ws/train/config/preprocessing_config.yaml"
    data_config_path = "/lidar3d_detection_ws/train/config/data.yaml"
    processor = PointCloudProcessor(config_path, data_config_path)
    return processor

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
        num_samples: Number of samples to evaluate
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        save_images: Whether to save visualization of detection results
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    
    # Create PointCloudProcessor for consistent preprocessing - exactly the same as in train.py
    processor = create_processor()
    
    # Update parameters from config is not needed since it's loaded in the constructor
    
    # Set image dimensions based on config
    processor.x_max = int((processor.fwd_range[1] - processor.fwd_range[0]) / processor.resolution)
    processor.y_max = int((processor.side_range[1] - processor.side_range[0]) / processor.resolution)
    processor.z_max = int((processor.height_range[1] - processor.height_range[0]) / processor.z_resolution)
    
    # List bin files
    bin_dir = Path(bin_dir)
    bin_files = sorted(list(bin_dir.glob('*.bin')))
    
    # Select random samples if more files than requested
    if len(bin_files) > num_samples:
        bin_files = random.sample(bin_files, num_samples)
    else:
        num_samples = len(bin_files)
    
    print(f"Evaluating on {num_samples} samples")
    
    # Initialize metrics
    total_objects = 0
    detected_objects = 0
    class_metrics = {name: {'TP': 0, 'FP': 0, 'FN': 0} for name in CLASS_NAMES}
    
    # Process each sample
    for bin_file in tqdm(bin_files, desc="Evaluating samples"):
        # Get corresponding label file
        label_file = Path(label_dir) / f"{bin_file.stem}.txt"
        
        if not label_file.exists():
            print(f"Warning: Label file {label_file} not found, skipping...")
            continue
        
        # Process using the same method as in train.py
        bev_image, yolo_labels = processor.process_point_cloud(str(bin_file), str(label_file))
        
        # Run inference (bev_image is already correctly oriented)
        results = model.predict(bev_image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        # Convert ground truth labels to YOLO format
        # First transform 3D boxes to BEV coordinates
        gt_boxes = []
        for obj in yolo_labels:
            if obj['type'] not in CLASS_NAMES:
                continue
                
            # Process only care about valid classes
            class_id = CLASS_NAMES.index(obj['type'])
            
            # Transform box to BEV
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            
            # Get YOLO format label
            yolo_label_str = processor.create_yolo_label(corners_bev, obj['type'], bev_image.shape[:2])
            
            # Parse YOLO label
            parts = yolo_label_str.split()
            yolo_box = [int(parts[0])] + [float(p) for p in parts[1:]]
            
            # Convert YOLO box to (x1,y1,x2,y2) format
            bbox = yolo_to_bbox(yolo_box, bev_image.shape[1], bev_image.shape[0])
            
            gt_boxes.append((class_id, bbox, 1.0))  # Add confidence of 1.0 for GT
            
            # Count total objects by class
            total_objects += 1
            class_metrics[obj['type']]['FN'] += 1  # Assume false negative initially
        
        # Parse detection results
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
            for gt_idx, (gt_cls_id, gt_box, _) in enumerate(gt_boxes):
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
