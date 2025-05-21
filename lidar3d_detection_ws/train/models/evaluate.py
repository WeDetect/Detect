import os
import sys
import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from ultralytics import YOLO
import matplotlib
# Use non-interactive backend to avoid GUI issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from our other modules
from data_processing.preprocessing import load_config, read_bin_file, read_label_file, create_bev_image
from data_processing.preproccesing_0 import convert_labels_to_yolo_format, PointCloudProcessor
from data_processing.augmentation import create_range_adapted_bev_image

def evaluate_model(model_path, bin_dir, label_dir, config_path, output_dir, conf_threshold=0.25, iou_threshold=0.5, max_samples=10):
    """
    Evaluate a trained YOLO model on BEV images
    
    Args:
        model_path: Path to trained model weights
        bin_dir: Directory containing bin files
        label_dir: Directory containing label files
        config_path: Path to preprocessing config file
        output_dir: Output directory for evaluation results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS and TP calculation
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n===== EVALUATING MODEL =====")
    print(f"Model: {model_path}")
    print(f"Bin directory: {bin_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Max samples: {max_samples}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Initialize processor
    processor = PointCloudProcessor(config_path)
    
    # Get list of bin files
    bin_files = sorted([os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith('.bin')])
    
    # Limit number of samples if needed
    if max_samples > 0:
        bin_files = bin_files[:max_samples]
    
    # Initialize statistics
    class_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    class_ious = defaultdict(list)
    
    # Process each bin file
    for i, bin_file in enumerate(tqdm(bin_files, desc="Evaluating")):
        try:
            # Get corresponding label file
            base_name = os.path.splitext(os.path.basename(bin_file))[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for {bin_file}, skipping")
                continue
            
            # Read point cloud and labels
            points = read_bin_file(bin_file)
            labels = read_label_file(label_file)
            
            # Create clean BEV image (without boxes) for inference
            bev_image = processor.create_bev_image(points)
            
            # Create YOLO format labels for ground truth
            yolo_labels = []
            yolo_labels_str = []
            
            for obj in labels:
                if obj['type'] == 'DontCare':
                    continue
                
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label_str = processor.create_yolo_label(
                    corners_bev, obj['type'], bev_image.shape[:2]
                )
                
                if yolo_label_str:
                    yolo_labels_str.append(yolo_label_str)
            
            # Skip if no labels
            if not yolo_labels_str:
                print(f"Warning: No labels found for {bin_file}, skipping")
                continue
            
            # Convert string labels to numeric format for ground truth
            gt_boxes = []
            gt_classes = []
            
            for label_str in yolo_labels_str:
                parts = label_str.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # Convert normalized coordinates to pixel coordinates
                    img_height, img_width = bev_image.shape[:2]
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height
                    
                    # Calculate box corners
                    x1 = x_center_px - width_px / 2
                    y1 = y_center_px - height_px / 2
                    x2 = x_center_px + width_px / 2
                    y2 = y_center_px + height_px / 2
                    
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(int(class_id))
            
            # Run inference on the clean BEV image
            results = model.predict(bev_image, conf=conf_threshold, iou=iou_threshold)[0]
            
            # Extract predictions
            pred_boxes = []
            pred_classes = []
            pred_scores = []
            
            if len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    pred_boxes.append([x1, y1, x2, y2])
                    pred_classes.append(cls)
                    pred_scores.append(conf)
            
            # Calculate IoU for each ground truth box with best matching prediction
            for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                best_iou = 0
                best_pred_idx = -1
                
                for pred_idx, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
                    if pred_cls == gt_cls:  # Only compare boxes of same class
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = pred_idx
                
                # Get class name
                class_name = processor.class_names[gt_cls]
                
                if best_iou >= iou_threshold:
                    # True positive
                    class_stats[class_name]['TP'] += 1
                    class_ious[class_name].append(best_iou)
                else:
                    # False negative (ground truth not detected)
                    class_stats[class_name]['FN'] += 1
            
            # Check for false positives (predictions without matching ground truth)
            for pred_idx, (pred_box, pred_cls) in enumerate(zip(pred_boxes, pred_classes)):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                    if pred_cls == gt_cls:
                        iou = calculate_iou(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                # Get class name
                class_name = processor.class_names[pred_cls]
                
                if best_iou < iou_threshold:
                    # False positive
                    class_stats[class_name]['FP'] += 1
            
            # Create visualization for the first 3 samples
            if i < 3:
                # Create a copy of the clean BEV image for ground truth visualization
                gt_img = bev_image.copy()
                
                # Draw ground truth boxes
                for gt_box, gt_cls in zip(gt_boxes, gt_classes):
                    x1, y1, x2, y2 = map(int, gt_box)
                    color = processor.colors[processor.class_names[gt_cls]]
                    cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(gt_img, processor.class_names[gt_cls], (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Get prediction image from results
                pred_img = results.plot()
                
                # Create side-by-side comparison
                comparison_img = np.hstack((gt_img, pred_img))
                comparison_path = os.path.join(visualization_dir, f"sample_{i+1}_comparison.png")
                cv2.imwrite(comparison_path, comparison_img)
                print(f"Saved comparison image to: {comparison_path}")
        
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue
    
    # Calculate metrics for each class
    class_metrics = {}
    for class_name, stats in class_stats.items():
        tp = stats['TP']
        fp = stats['FP']
        fn = stats['FN']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }
    
    # Calculate average IoU for each class
    avg_ious = {}
    for class_name, ious in class_ious.items():
        if ious:
            avg_ious[class_name] = sum(ious) / len(ious)
        else:
            avg_ious[class_name] = 0
    
    # Calculate overall metrics
    avg_precision = sum(m['precision'] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0
    avg_recall = sum(m['recall'] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0
    macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0
    
    # Create confusion matrix visualization
    # Instead of using sklearn's confusion_matrix, we'll create our own
    class_names = list(class_stats.keys())
    num_classes = len(class_names)
    
    # Create a mapping from class name to index
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Initialize confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    # Fill confusion matrix using our statistics
    for class_name, stats in class_stats.items():
        i = class_to_idx[class_name]
        # True positives go on the diagonal
        cm[i, i] = stats['TP']
        
        # False negatives are spread across the row (ground truth is class i but predicted as something else)
        # We'll just put them in a special "missed" column for simplicity
        if num_classes > 1:  # Only if we have more than one class
            missed_col = (i + 1) % num_classes  # Just pick another column
            cm[i, missed_col] = stats['FN']
        
        # False positives are spread across the column (predicted as class i but actually something else)
        # We'll just put them in a special "false" row for simplicity
        if num_classes > 1:  # Only if we have more than one class
            false_row = (i + 1) % num_classes  # Just pick another row
            cm[false_row, i] = stats['FP']
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to: {cm_path}")
    
    # Save report to file
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n\n")
        
        f.write(f"Model: {model_path}\n")
        f.write(f"Confidence threshold: {conf_threshold}\n")
        f.write(f"IoU threshold: {iou_threshold}\n\n")
        
        f.write("Class-wise Performance:\n")
        f.write("----------------------\n")
        for class_name, metrics in class_metrics.items():
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
            f.write(f"  Support: {metrics['support']}\n")
            if class_name in avg_ious:
                f.write(f"  Average IoU: {avg_ious[class_name]:.4f}\n")
            f.write("\n")
        
        f.write("Overall Performance:\n")
        f.write("-------------------\n")
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Macro Avg F1-Score: {macro_f1:.4f}\n")
    
    print(f"Saved classification report to: {report_path}")
    
    # Print summary to console
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Macro Avg F1-Score: {macro_f1:.4f}")
    print("\nClass-wise Performance:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        if class_name in avg_ious:
            print(f"    Average IoU: {avg_ious[class_name]:.4f}")
    
    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics,
        'avg_ious': avg_ious
    }

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on BEV images')
    
    # Model and dataset options
    parser.add_argument('--model_path', type=str, default='/lidar3d_detection_ws/train/output/bev-from-scratch/train/weights/best.pt', 
                        help='Path to trained model weights')
    parser.add_argument('--bin_dir', type=str, default='/lidar3d_detection_ws/data/innoviz/', 
                        help='Directory containing bin files')
    parser.add_argument('--label_dir', type=str, default='/lidar3d_detection_ws/data/labels/', 
                        help='Directory containing label files')
    parser.add_argument('--config_path', type=str, default='/lidar3d_detection_ws/train/config/preprocessing_config.yaml', 
                        help='Path to preprocessing config file')
    parser.add_argument('--output_dir', type=str, default='/lidar3d_detection_ws/train/evaluation', 
                        help='Output directory for evaluation results')
    
    # Evaluation options
    parser.add_argument('--conf_threshold', type=float, default=0.25, 
                        help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.5, 
                        help='IoU threshold for NMS and TP calculation')
    parser.add_argument('--max_samples', type=int, default=10, 
                        help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Evaluate model
    metrics = evaluate_model(
        model_path=args.model_path,
        bin_dir=args.bin_dir,
        label_dir=args.label_dir,
        config_path=args.config_path,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_samples=args.max_samples
    )
    
    return metrics

if __name__ == "__main__":
    main()