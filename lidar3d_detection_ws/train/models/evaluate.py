import os
import sys
import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path
import torch
from ultralytics import YOLO

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our processor and functions
from data_processing.preproccesing_0 import PointCloudProcessor
# Import convert_labels_to_yolo_format directly
from data_processing.preproccesing_0 import convert_labels_to_yolo_format

# Define class names
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Truck']
CLASS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR format

# Default paths
DEFAULT_MODEL_PATH = '/lidar3d_detection_ws/train/output/best.pt'
DEFAULT_CONFIG_PATH = '/lidar3d_detection_ws/train/config/preprocessing_config.yaml'
DEFAULT_OUTPUT_DIR = '/lidar3d_detection_ws/output/eval'
DEFAULT_BIN_FILE = '/lidar3d_detection_ws/data/innoviz/innoviz_00010.bin'
DEFAULT_LABEL_FILE = '/lidar3d_detection_ws/data/labels/innoviz_00010.txt'

def evaluate_single_file(model_path, bin_file, label_file, config_path, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Evaluate model on a single point cloud file
    
    Args:
        model_path: Path to trained YOLO model
        bin_file: Path to point cloud bin file
        label_file: Path to corresponding label file
        config_path: Path to preprocessing config file
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file name without extension
    base_name = os.path.basename(bin_file).split('.')[0]
    
    # Load trained YOLO model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Create processor with the same settings as training
    print(f"Loading config from {config_path}")
    processor = PointCloudProcessor(config_path=config_path)
    
    # Load the configuration manually for YOLO format conversion
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Processing file: {base_name}")
    
    # Load point cloud data
    points = processor.load_point_cloud(bin_file)
    
    # Load labels
    if os.path.exists(label_file):
        labels = processor.load_labels(label_file)
        print(f"Raw labels from file:")
        for label in labels:
            print(f"  {label}")
    else:
        print(f"Warning: Label file not found: {label_file}")
        labels = []
    
    # Create BEV image - this is the standard orientation
    bev_image = processor.create_bev_image(points)
    
    # Create a copy for drawing results
    result_image = bev_image.copy()
    
    # Get image dimensions for converting YOLO format
    height, width = bev_image.shape[:2]
    
    # Extract ground truth boxes from labels - TWO APPROACHES
    gt_boxes = []
    
    # 1. First approach using transform_3d_box_to_bev (for visualization only)
    print("\nGround Truth Boxes (3D Transform Method):")
    for label in labels:
        if label['type'] not in CLASS_NAMES:
            continue
            
        cls_idx = CLASS_NAMES.index(label['type'])
        
        # Get box coordinates for visualization
        corners_bev, center_bev = processor.transform_3d_box_to_bev(
            label['dimensions'], label['location'], label['rotation_y']
        )
        
        # Draw ground truth box (polygon shape from 3D transform)
        cv2.polylines(result_image, [np.array(corners_bev).astype(np.int32)], True, (0, 255, 0), 2)
        
        # Get min/max to create box coordinates
        if isinstance(corners_bev, list):
            x_coords = [corner[0] for corner in corners_bev]
            y_coords = [corner[1] for corner in corners_bev]
        else:
            x_coords = corners_bev[:, 0]
            y_coords = corners_bev[:, 1]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Ensure non-zero width/height
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1
        
        # Draw rectangular box from 3D transform (yellow)
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)

    # 2. Second approach: Use convert_labels_to_yolo_format (same as training)
    print("\nGround Truth Boxes (YOLO Format - Same as Training):")
    
    # Use the YOLO format conversion (same method used in training) - calling directly
    yolo_format_labels = convert_labels_to_yolo_format(
        labels, config, width, height
    )
    
    # Debug the structure of yolo_format_labels
    print(f"YOLO format labels structure: {type(yolo_format_labels)}")
    if len(yolo_format_labels) > 0:
        print(f"First element type: {type(yolo_format_labels[0])}")
        print(f"First element: {yolo_format_labels[0]}")
    
    # Create a rotated version of the BEV image for training compatibility
    # This is what the model expects during inference
    rotated_bev = cv2.rotate(bev_image.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_height, rotated_width = rotated_bev.shape[:2]
    
    # Create a copy for drawing rotated ground truth
    rotated_gt_image = rotated_bev.copy()
    
    # Convert YOLO format back to pixel coordinates for evaluation
    for label in yolo_format_labels:
        # Check if label is a list/array with at least 5 elements
        if not isinstance(label, (list, np.ndarray)) or len(label) < 5:
            print(f"Skipping invalid label: {label}")
            continue
            
        # YOLO format: [class_id, x_center, y_center, width, height] (normalized)
        cls_idx = int(label[0])
        if cls_idx >= len(CLASS_NAMES):
            print(f"Skipping label with invalid class index: {cls_idx}")
            continue
            
        # Original coordinates (non-rotated)
        x_center, y_center = label[1] * width, label[2] * height
        w, h = label[3] * width, label[4] * height
        
        # Convert to pixel coordinates
        x_min = int(x_center - w/2)
        y_min = int(y_center - h/2)
        x_max = int(x_center + w/2)
        y_max = int(y_center + h/2)
        
        # Ensure valid coordinates
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width-1, x_max)
        y_max = min(height-1, y_max)
        
        # Ensure non-zero width/height
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        
        # Store the box for evaluation (using YOLO format method)
        gt_box = (cls_idx, x_min, y_min, x_max, y_max)
        gt_boxes.append(gt_box)
        
        # Draw rectangular box from YOLO format (blue) on original orientation
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        # Add class label
        label_text = f"{CLASS_NAMES[cls_idx]} (YOLO)"
        cv2.putText(result_image, label_text, (x_min, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Print box info
        print(f"GT Box (YOLO) for {CLASS_NAMES[cls_idx]}: {gt_box}")
        
        # Now transform the coordinates for the rotated image
        # In 90° counterclockwise rotation:
        # new_x = y, new_y = height - x
        rot_x_min = y_min
        rot_y_min = width - x_max
        rot_x_max = y_max
        rot_y_max = width - x_min
        
        # Draw on rotated image
        cv2.rectangle(rotated_gt_image, (rot_x_min, rot_y_min), (rot_x_max, rot_y_max), (255, 0, 0), 2)
        cv2.putText(rotated_gt_image, label_text, (rot_x_min, rot_y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save the ground truth visualizations
    gt_vis_path = os.path.join(output_dir, f"{base_name}_gt.png")
    cv2.imwrite(gt_vis_path, result_image)
    
    rotated_gt_vis_path = os.path.join(output_dir, f"{base_name}_gt_rotated.png")
    cv2.imwrite(rotated_gt_vis_path, rotated_gt_image)
    
    print(f"Ground truth visualization saved to: {gt_vis_path}")
    print(f"Rotated ground truth visualization saved to: {rotated_gt_vis_path}")
    
    # Run inference with the model on the rotated image
    print("\nRunning inference with YOLO model...")
    results = model.predict(rotated_bev, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
    
    # Create a copy of the rotated image for predictions
    rotated_pred_image = rotated_bev.copy()
    
    # Extract predictions
    pred_boxes = []
    
    print("\nPredicted Boxes:")
    for i, det in enumerate(results.boxes.data.tolist()):
        x_min, y_min, x_max, y_max, conf, cls_idx = det
        
        # Convert to integers
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cls_idx = int(cls_idx)
        
        if cls_idx >= len(CLASS_NAMES):
            print(f"Skipping prediction with invalid class index: {cls_idx}")
            continue
        
        # Store prediction in rotated coordinates
        rot_pred_box = (cls_idx, x_min, y_min, x_max, y_max, conf)
        
        # Draw prediction box on rotated image
        cv2.rectangle(rotated_pred_image, (x_min, y_min), (x_max, y_max), CLASS_COLORS[cls_idx], 2)
        
        # Add class label and confidence
        label_text = f"{CLASS_NAMES[cls_idx]} {conf:.2f}"
        cv2.putText(rotated_pred_image, label_text, (x_min, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASS_COLORS[cls_idx], 2)
        
        # Print prediction info in rotated coordinates
        print(f"Pred Box (rotated) for {CLASS_NAMES[cls_idx]}: {rot_pred_box}")
        
        # Transform back to original orientation for evaluation
        # In 90° clockwise rotation (inverse of counterclockwise):
        # new_x = width - y, new_y = x
        orig_x_min = rotated_height - y_max
        orig_y_min = x_min
        orig_x_max = rotated_height - y_min
        orig_y_max = x_max
        
        # Store in original orientation for evaluation
        pred_box = (cls_idx, orig_x_min, orig_y_min, orig_x_max, orig_y_max, conf)
        pred_boxes.append(pred_box)
        
        # Print prediction info in original coordinates
        print(f"Pred Box (original) for {CLASS_NAMES[cls_idx]}: {pred_box}")
    
    # Save the prediction visualization
    rotated_pred_vis_path = os.path.join(output_dir, f"{base_name}_pred_rotated.png")
    cv2.imwrite(rotated_pred_vis_path, rotated_pred_image)
    print(f"Rotated prediction visualization saved to: {rotated_pred_vis_path}")
    
    # Create a combined visualization of rotated images
    combined_rotated = np.hstack((rotated_gt_image, rotated_pred_image))
    combined_rotated_path = os.path.join(output_dir, f"{base_name}_combined_rotated.png")
    cv2.imwrite(combined_rotated_path, combined_rotated)
    print(f"Combined rotated visualization saved to: {combined_rotated_path}")
    
    # Calculate metrics (simple IoU-based)
    print("\nCalculating metrics...")
    
    # Count true positives, false positives, false negatives
    tp = 0
    fp = 0
    fn = 0
    
    # For each ground truth box, find the best matching prediction
    for gt_box in gt_boxes:
        gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
        best_iou = 0
        best_pred_idx = -1
        
        for i, pred_box in enumerate(pred_boxes):
            if pred_box is None:
                continue
                
            pred_cls, pred_x1, pred_y1, pred_x2, pred_y2, _ = pred_box
            
            # Only compare boxes of the same class
            if gt_cls != pred_cls:
                continue
            
            # Calculate IoU
            x_left = max(gt_x1, pred_x1)
            y_top = max(gt_y1, pred_y1)
            x_right = min(gt_x2, pred_x2)
            y_bottom = min(gt_y2, pred_y2)
            
            if x_right < x_left or y_bottom < y_top:
                # No overlap
                continue
            
            intersection = (x_right - x_left) * (y_bottom - y_top)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            union = gt_area + pred_area - intersection
            
            iou = intersection / union
            
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = i
        
        if best_iou >= iou_threshold:
            # True positive
            tp += 1
            # Mark this prediction as used
            if best_pred_idx >= 0:
                pred_boxes[best_pred_idx] = None
        else:
            # False negative
            fn += 1
    
    # Count remaining predictions as false positives
    for pred_box in pred_boxes:
        if pred_box is not None:
            fp += 1
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'gt_vis_path': gt_vis_path,
        'rotated_gt_vis_path': rotated_gt_vis_path,
        'rotated_pred_vis_path': rotated_pred_vis_path,
        'combined_rotated_path': combined_rotated_path
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on point cloud data")
    
    # Add arguments with default values
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, 
                      help="Path to trained YOLO model")
    parser.add_argument("--bin_file", default=DEFAULT_BIN_FILE,
                      help="Path to point cloud bin file")
    parser.add_argument("--label_file", default=DEFAULT_LABEL_FILE,
                      help="Path to label file")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG_PATH,
                      help="Path to preprocessing config file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                      help="Directory to save results")
    parser.add_argument("--conf", type=float, default=0.25,
                      help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.45,
                      help="IoU threshold for NMS")
    
    args = parser.parse_args()
    
    # Call evaluate function with parsed arguments
    evaluate_single_file(
        args.model, 
        args.bin_file, 
        args.label_file, 
        args.config_path, 
        args.output_dir,
        args.conf,
        args.iou
    )

if __name__ == "__main__":
    main()