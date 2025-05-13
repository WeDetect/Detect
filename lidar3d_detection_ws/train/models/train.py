import os
import sys
import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
from ultralytics import YOLO
import torch
import glob

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from our other modules
from data_processing.preprocessing import load_config, read_bin_file, read_label_file, create_bev_image
from data_processing.preproccesing_0 import convert_labels_to_yolo_format
from data_processing.augmentation import (rotate_points_and_labels, 
                                         shift_lateral_points_and_labels,
                                         shift_vertical_points_and_labels,
                                         filter_points_by_range,
                                         create_range_adapted_bev_image)
from models.yolo_bev import create_full_trainable_model, create_transfer_learning_model
from data_processing.preproccesing_0 import PointCloudProcessor

# Global configuration
DEFAULT_CONFIG = {
    'bin_dir': '/lidar3d_detection_ws/data/innoviz',
    'label_dir': '/lidar3d_detection_ws/data/labels',
    'config_path': '/lidar3d_detection_ws/train/config/preprocessing_config.yaml',
    'output_base': '/lidar3d_detection_ws/train',
    'train_val_split': 0.8,  # 80% training, 20% validation
    'epochs': 70,
    'batch_size': 8,
    'img_size': 608,
    'augmentations': False,
    'augmentation_factor': 3,  # Number of augmented samples per original
    'device': 'cpu',  # 'cpu' or 'cuda:0'
}



def train_on_single_image_from_scratch(bin_file, label_file, config_path, output_dir, epochs=100, img_size=640, batch_size=1, device='cpu'):
    """
    Train a model from scratch on a single image for testing purposes
    
    Args:
        bin_file: Path to bin file
        label_file: Path to label file
        config_path: Path to config file
        output_dir: Output directory
        epochs: Number of epochs
        img_size: Image size
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Path to best weights
    """
    print("\n===== TRAINING FROM SCRATCH ON SINGLE IMAGE =====")
    print(f"Using bin file: {bin_file}")
    print(f"Using label file: {label_file}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "single_image_visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Initialize processor
    processor = PointCloudProcessor(config_path=config_path)
    
    # Process the point cloud to get BEV image and YOLO labels
    bev_image, yolo_labels_str = processor.process_point_cloud(
        bin_file, label_file, 
        os.path.join(visualization_dir, "original_with_boxes.png"),
        None  # Don't save labels to file
    )
    
    # Convert string labels to numeric format
    yolo_labels = []
    for label_str in yolo_labels_str:
        parts = label_str.split()
        if len(parts) == 5:
            yolo_labels.append([float(p) for p in parts])
    
    # הוסף בדיקה של התיבות המקוריות
    print("\n===== CHECKING ORIGINAL 3D BOXES =====")
    objects = processor.load_labels(label_file)
    for i, obj in enumerate(objects):
        print(f"Object {i+1}: Type={obj['type']}, Location=({obj['location'][0]:.2f}, {obj['location'][1]:.2f}, {obj['location'][2]:.2f}), Dimensions=({obj['dimensions'][0]:.2f}, {obj['dimensions'][1]:.2f}, {obj['dimensions'][2]:.2f}), Rotation={obj['rotation_y']:.2f}")
    
    # הוסף בדיקה של התיבות ב-BEV
    print("\n===== CHECKING BEV BOXES =====")
    for i, obj in enumerate(objects):
        corners_bev, center_bev = processor.transform_3d_box_to_bev(
            obj['dimensions'], obj['location'], obj['rotation_y']
        )
        x_coords = [x for x, y in corners_bev]
        y_coords = [y for x, y in corners_bev]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        print(f"Object {i+1}: Type={obj['type']}, BEV Width={width}, BEV Height={height}")
    
    # הוסף בדיקה של התיבות
    print("\n===== CHECKING YOLO LABELS =====")
    print("Ground Truth Labels (YOLO format):")
    for i, label in enumerate(yolo_labels):
        class_id, x_center, y_center, width, height = label
        class_name = processor.class_names[int(class_id)]
        print(f"Object {i+1}: Class={class_name}, Center=({x_center:.4f}, {y_center:.4f}), Size=({width:.4f}, {height:.4f})")
    
    # יצירת תמונה עם תיבות מסומנות לפי התוויות
    verification_dir = os.path.join(output_dir, "label_verification")
    os.makedirs(verification_dir, exist_ok=True)
    
    # העתק את התמונה המקורית
    verify_img = bev_image.copy()
    
    # צייר את התיבות לפי התוויות
    for label in yolo_labels:
        class_id, x_center, y_center, width, height = label
        # המר מקואורדינטות מנורמלות לפיקסלים
        x_center_px = int(x_center * bev_image.shape[1])
        y_center_px = int(y_center * bev_image.shape[0])
        width_px = int(width * bev_image.shape[1])
        height_px = int(height * bev_image.shape[0])
        
        # חשב את הפינות של התיבה
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        # צייר את התיבה
        color = processor.colors[processor.class_names[int(class_id)]]
        cv2.rectangle(verify_img, (x1, y1), (x2, y2), color, 2)
        
    # שמור את התמונה עם התיבות
    verify_path = os.path.join(verification_dir, "labels_verification.png")
    cv2.imwrite(verify_path, verify_img)
    print(f"Saved label verification image to: {verify_path}")
    
    # יצירת תמונה עם תיבות מסומנות לפי התוויות המקוריות
    original_img = bev_image.copy()
    for obj in objects:
        corners_bev, center_bev = processor.transform_3d_box_to_bev(
            obj['dimensions'], obj['location'], obj['rotation_y']
        )
        # צייר את התיבה המקורית
        pts = np.array(corners_bev, np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = processor.colors[obj['type']]
        cv2.polylines(original_img, [pts], True, color, 2)
    
    # שמור את התמונה עם התיבות המקוריות
    original_path = os.path.join(verification_dir, "original_boxes.png")
    cv2.imwrite(original_path, original_img)
    print(f"Saved original boxes image to: {original_path}")
    
    # יצירת תמונה משולבת להשוואה
    combined_img = np.hstack((original_img, verify_img))
    combined_path = os.path.join(verification_dir, "comparison.png")
    cv2.imwrite(combined_path, combined_img)
    print(f"Saved comparison image to: {combined_path}")
    
    # Create dataset for training
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    train_img_dir = os.path.join(train_dir, "images")
    train_label_dir = os.path.join(train_dir, "labels")
    val_img_dir = os.path.join(val_dir, "images")
    val_label_dir = os.path.join(val_dir, "labels")
    
    # Create directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Save image and labels for training
    train_img_path = os.path.join(train_img_dir, "train_0.png")
    train_label_path = os.path.join(train_label_dir, "train_0.txt")
    cv2.imwrite(train_img_path, bev_image)
    with open(train_label_path, 'w') as f:
        for label_str in yolo_labels_str:
            f.write(label_str + '\n')
    
    # Save the same image for validation
    val_img_path = os.path.join(val_img_dir, "val_0.png")
    val_label_path = os.path.join(val_label_dir, "val_0.txt")
    cv2.imwrite(val_img_path, bev_image)
    with open(val_label_path, 'w') as f:
        for label_str in yolo_labels_str:
            f.write(label_str + '\n')
    
    # Create custom YAML dataset configuration
    dataset_config = {
        'path': output_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': 4,  # Number of classes
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Truck']
    }
    
    # Create a temporary YAML file with dataset configuration
    dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Check CUDA availability and use appropriate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    # Create a new model from scratch
    print("\n===== CREATING NEW MODEL FROM SCRATCH =====")
    model = create_full_trainable_model()
    
    # Prepare training arguments
    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': output_dir,
        'name': 'bev-from-scratch',
        'exist_ok': True,
        'cache': True,
        'data': dataset_yaml,
        'patience': 50,  # Early stopping patience
        'save_period': 10  # Save checkpoint every 10 epochs
    }
    
    print(f"Using device: {device}")
    
    # Train the model
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    model.train(**train_args)
    
    # Return the path to best weights
    weights_dir = os.path.join(output_dir, 'bev-from-scratch', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    
    # Also save a copy with a more descriptive name
    final_weights = os.path.join(output_dir, 'bev_from_scratch_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    
    # Run inference with the trained model
    print("\n===== RUNNING INFERENCE WITH TRAINED MODEL =====")
    results = model.predict(bev_image, conf=0.25)
    
    # Visualize results
    result_img = results[0].plot()
    result_path = os.path.join(verification_dir, "model_prediction.png")
    cv2.imwrite(result_path, result_img)
    print(f"Saved model prediction to: {result_path}")
    
    # Create comparison with ground truth
    comparison_img = np.hstack((verify_img, result_img))
    comparison_path = os.path.join(verification_dir, "ground_truth_vs_prediction.png")
    cv2.imwrite(comparison_path, comparison_img)
    print(f"Saved ground truth vs prediction comparison to: {comparison_path}")
    
    # Print detected objects
    print("\n===== DETECTED OBJECTS =====")
    for i, det in enumerate(results[0].boxes.data):
        x1, y1, x2, y2, conf, cls = det
        class_name = processor.class_names[int(cls)]
        print(f"Object {i+1}: Class={class_name}, Confidence={conf:.4f}, Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    # Clean up temporary directories
    print("Cleaning up temporary dataset directories...")
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    
    return best_weights

def train_on_all_data_from_scratch(bin_dir, label_dir, config_path, output_dir, epochs=100, img_size=640, batch_size=16, device='cpu'):
    """
    Train a model from scratch on all data, similar to single image training but for all files
    
    Args:
        bin_dir: Directory containing bin files
        label_dir: Directory containing label files
        config_path: Path to config file
        output_dir: Output directory
        epochs: Number of epochs
        img_size: Image size
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Path to best weights
    """
    print("\n===== TRAINING FROM SCRATCH ON ALL DATA =====")
    print(f"Using bin directory: {bin_dir}")
    print(f"Using label directory: {label_dir}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    visualization_dir = os.path.join(output_dir, "all_data_visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    verification_dir = os.path.join(output_dir, "label_verification")
    os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize processor
    processor = PointCloudProcessor(config_path=config_path)
    
    # Get all bin files
    bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
    if not bin_files:
        raise ValueError(f"No bin files found in {bin_dir}")
    
    print(f"Found {len(bin_files)} bin files")
    
    # Create dataset for training
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    train_img_dir = os.path.join(train_dir, "images")
    train_label_dir = os.path.join(train_dir, "labels")
    val_img_dir = os.path.join(val_dir, "images")
    val_label_dir = os.path.join(val_dir, "labels")
    
    # Create directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Process all files and split into train/val
    all_data = []
    
    for i, bin_file in enumerate(tqdm(bin_files, desc="Processing point clouds")):
        # Get corresponding label file
        base_name = os.path.basename(bin_file)
        label_name = os.path.splitext(base_name)[0] + ".txt"
        label_file = os.path.join(label_dir, label_name)
        
        if not os.path.exists(label_file):
            print(f"Warning: Label file {label_file} not found, skipping {bin_file}")
            continue
        
        # Process the point cloud
        try:
            # Create visualization path for this file
            vis_path = os.path.join(visualization_dir, f"{os.path.splitext(base_name)[0]}_with_boxes.png")
            
            # Process point cloud to get BEV image and YOLO labels
            bev_image, yolo_labels_str = processor.process_point_cloud(
                bin_file, label_file, vis_path, None
            )
            
            # Skip if no labels
            if not yolo_labels_str:
                print(f"Warning: No valid labels found for {bin_file}, skipping")
                continue
            
            # Add to dataset
            all_data.append({
                'bin_file': bin_file,
                'label_file': label_file,
                'bev_image': bev_image,
                'yolo_labels': yolo_labels_str
            })
            
            # Create verification image for this file
            verify_img = bev_image.copy()
            
            # Draw boxes based on YOLO labels
            for label_str in yolo_labels_str:
                parts = label_str.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_center_px = int(x_center * bev_image.shape[1])
                    y_center_px = int(y_center * bev_image.shape[0])
                    width_px = int(width * bev_image.shape[1])
                    height_px = int(height * bev_image.shape[0])
                    
                    # Calculate box corners
                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)
                    
                    # Draw box
                    color = processor.colors[processor.class_names[int(class_id)]]
                    cv2.rectangle(verify_img, (x1, y1), (x2, y2), color, 2)
            
            # Save verification image
            verify_path = os.path.join(verification_dir, f"{os.path.splitext(base_name)[0]}_labels_verification.png")
            cv2.imwrite(verify_path, verify_img)
            
            # Also create original boxes visualization
            original_img = bev_image.copy()
            objects = processor.load_labels(label_file)
            for obj in objects:
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                # Draw original box
                pts = np.array(corners_bev, np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = processor.colors[obj['type']]
                cv2.polylines(original_img, [pts], True, color, 2)
            
            # Save original boxes image
            original_path = os.path.join(verification_dir, f"{os.path.splitext(base_name)[0]}_original_boxes.png")
            cv2.imwrite(original_path, original_img)
            
            # Create comparison image
            combined_img = np.hstack((original_img, verify_img))
            combined_path = os.path.join(verification_dir, f"{os.path.splitext(base_name)[0]}_comparison.png")
            cv2.imwrite(combined_path, combined_img)
            
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue
    
    print(f"Successfully processed {len(all_data)} files")
    
    # Split into train/val
    train_val_split = 0.8  # 80% training, 20% validation
    train_size = int(len(all_data) * train_val_split)
    
    # Shuffle data
    random.shuffle(all_data)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]
    
    print(f"Training set: {len(train_data)} files")
    print(f"Validation set: {len(val_data)} files")
    
    # Save images and labels for training
    for i, item in enumerate(tqdm(train_data, desc="Saving training data")):
        train_img_path = os.path.join(train_img_dir, f"train_{i}.png")
        train_label_path = os.path.join(train_label_dir, f"train_{i}.txt")
        
        cv2.imwrite(train_img_path, item['bev_image'])
        with open(train_label_path, 'w') as f:
            for label_str in item['yolo_labels']:
                f.write(label_str + '\n')
    
    # Save images and labels for validation
    for i, item in enumerate(tqdm(val_data, desc="Saving validation data")):
        val_img_path = os.path.join(val_img_dir, f"val_{i}.png")
        val_label_path = os.path.join(val_label_dir, f"val_{i}.txt")
        
        cv2.imwrite(val_img_path, item['bev_image'])
        with open(val_label_path, 'w') as f:
            for label_str in item['yolo_labels']:
                f.write(label_str + '\n')
    
    # Create custom YAML dataset configuration
    dataset_config = {
        'path': output_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': 4,  # Number of classes
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Truck']
    }
    
    # Create a temporary YAML file with dataset configuration
    dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Check CUDA availability and use appropriate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    # Create a new model from scratch
    print("\n===== CREATING NEW MODEL FROM SCRATCH =====")
    model = create_full_trainable_model()
    
    # Prepare training arguments
    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'workers': 8,
        'project': os.path.join(output_dir, 'bev-from-scratch'),
        'name': 'train',
        'exist_ok': True,
        'pretrained': False,
        'optimizer': 'SGD',  # or 'Adam'
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'cache': True,
        'data': dataset_yaml,
        'patience': 50,  # Early stopping patience
        'save_period': 10  # Save checkpoint every 10 epochs
    }
    
    print(f"Using device: {device}")
    
    # Train the model
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    model.train(**train_args)
    
    # Return the path to best weights
    weights_dir = os.path.join(output_dir, 'bev-from-scratch', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    
    # Also save a copy with a more descriptive name
    final_weights = os.path.join(output_dir, 'bev_all_data_from_scratch_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    
    # Run inference with the trained model on a validation image
    print("\n===== RUNNING INFERENCE WITH TRAINED MODEL =====")
    
    # Choose a random validation image
    if val_data:
        val_item = random.choice(val_data)
        val_img = val_item['bev_image']
        val_labels = val_item['yolo_labels']
        val_bin_file = val_item['bin_file']
        
        # Run inference
        results = model.predict(val_img, conf=0.25)
        
        # Visualize results
        result_img = results[0].plot()
        result_path = os.path.join(verification_dir, "model_prediction.png")
        cv2.imwrite(result_path, result_img)
        print(f"Saved model prediction to: {result_path}")
        
        # Create verification image with ground truth labels
        verify_img = val_img.copy()
        
        # Draw boxes based on YOLO labels
        for label_str in val_labels:
            parts = label_str.split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                
                # Convert normalized coordinates to pixel coordinates
                x_center_px = int(x_center * verify_img.shape[1])
                y_center_px = int(y_center * verify_img.shape[0])
                width_px = int(width * verify_img.shape[1])
                height_px = int(height * verify_img.shape[0])
                
                # Calculate box corners
                x1 = int(x_center_px - width_px / 2)
                y1 = int(y_center_px - height_px / 2)
                x2 = int(x_center_px + width_px / 2)
                y2 = int(y_center_px + height_px / 2)
                
                # Draw box
                color = processor.colors[processor.class_names[int(class_id)]]
                cv2.rectangle(verify_img, (x1, y1), (x2, y2), color, 2)
        
        # Create comparison with ground truth
        comparison_img = np.hstack((verify_img, result_img))
        comparison_path = os.path.join(verification_dir, "ground_truth_vs_prediction.png")
        cv2.imwrite(comparison_path, comparison_img)
        print(f"Saved ground truth vs prediction comparison to: {comparison_path}")
        
        # Print detected objects
        print("\n===== DETECTED OBJECTS =====")
        for i, det in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = det
            class_name = processor.class_names[int(cls)]
            print(f"Object {i+1}: Class={class_name}, Confidence={conf:.4f}, Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
    
    # Clean up temporary directories
    print("Cleaning up temporary dataset directories...")
    # Uncomment if you want to clean up after training
    # shutil.rmtree(train_dir, ignore_errors=True)
    # shutil.rmtree(val_dir, ignore_errors=True)
    
    return best_weights

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLO model on BEV images')
    
    # Dataset generation options
    parser.add_argument('--bin_dir', type=str, default='/lidar3d_detection_ws/data/innoviz/', 
                        help='Directory containing bin files')
    parser.add_argument('--label_dir', type=str, default='/lidar3d_detection_ws/data/labels/', 
                        help='Directory containing label files')
    parser.add_argument('--config_path', type=str, default='/lidar3d_detection_ws/train/config/preprocessing_config.yaml', 
                        help='Path to preprocessing config file')
    parser.add_argument('--output_base', type=str, default='/lidar3d_detection_ws/train', 
                        help='Base directory for output')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='Image size')
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device to use (cuda:0, cpu, or auto)')
    
    # Model options
    parser.add_argument('--transfer_learning', action='store_true', 
                        help='Use transfer learning')
    parser.add_argument('--unfreeze_layers', type=int, default=10, 
                        help='Number of layers to unfreeze for transfer learning')
    parser.add_argument('--continue_from', type=str, default='', 
                        help='Path to model weights to continue training from')
    
    # Augmentation options
    parser.add_argument('--augmentations', action='store_true', 
                        help='Enable data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=2, 
                        help='Augmentation factor (multiplier for dataset size)')
    
    # Single image test options
    parser.add_argument('--single_image_test', action='store_true', 
                        help='Train on a single image for testing')
    parser.add_argument('--bin_file', type=str, default='', 
                        help='Path to specific bin file for single image test')
    parser.add_argument('--label_file', type=str, default='', 
                        help='Path to specific label file for single image test')
    parser.add_argument('--train_from_scratch', action='store_true',
                        help='Train from scratch on a single image')
    
    # All data training option
    parser.add_argument('--all_data_from_scratch', action='store_true',
                        help='Train from scratch on all data with detailed visualization')
    
    args = parser.parse_args()
    
    # Handle device selection - auto-detect if set to "auto"
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Auto-selected device: {args.device}")
    
    # אם נבחרה האופציה לאימון על תמונה בודדת
    if args.single_image_test:
        # קבע קבצי ברירת מחדל אם לא צוינו
        bin_file = args.bin_file or os.path.join(args.bin_dir, "innoviz_00010.bin")
        label_file = args.label_file or os.path.join(args.label_dir, "innoviz_00010.txt")
        
        print(f"Using bin file: {bin_file}")
        
        if args.train_from_scratch:
            # אימון מאפס על תמונה בודדת
            model_path = train_on_single_image_from_scratch(
                bin_file=bin_file,
                label_file=label_file,
                config_path=args.config_path,
                output_dir=os.path.join(args.output_base, "output"),
                epochs=args.epochs,
                img_size=args.img_size,
                batch_size=args.batch,
                device=args.device
            )
            print(f"Single image training from scratch completed. Model saved to: {model_path}")
        else:
            # המשך אימון על תמונה בודדת (אם צריך)
            print("Please use --train_from_scratch for single image training")
        
        return model_path
    
    # אם נבחרה האופציה לאימון מאפס על כל הדאטה עם ויזואליזציה מפורטת
    elif args.all_data_from_scratch:
        model_path = train_on_all_data_from_scratch(
            bin_dir=args.bin_dir,
            label_dir=args.label_dir,
            config_path=args.config_path,
            output_dir=os.path.join(args.output_base, "output"),
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device
        )
        print(f"All data training from scratch completed. Model saved to: {model_path}")
        return model_path
    
    # אימון על כל הדאטה בשיטה הרגילה
    else:
        model_path = train_on_full_dataset(
            bin_dir=args.bin_dir,
            label_dir=args.label_dir,
            config_path=args.config_path,
            output_base=args.output_base,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch,
            device=args.device,
            augmentations=args.augmentations,
            augmentation_factor=args.augmentation_factor,
            transfer_learning=args.transfer_learning,
            unfreeze_layers=args.unfreeze_layers,
            continue_from=args.continue_from
        )
        return model_path

if __name__ == "__main__":
    main()