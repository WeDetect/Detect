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
import copy
import pandas as pd

# Set matplotlib to headless mode BEFORE any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import from our other modules
from data_processing.preprocessing import load_config, read_bin_file, read_label_file
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
    'batch_size': 16,
    'img_size': 608,
    'augmentations': False,
    'augmentation_factor': 3,  # Number of augmented samples per original
    'device': 'cpu',  # 'cpu' or 'cuda:0'
}

def plot_metrics(train_losses, val_losses, val_maps_50, val_maps_50_95, output_dir='training_logs'):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory

        # mAP@0.5 plot
        plt.figure(figsize=(10, 6))
        plt.plot(val_maps_50, label='mAP@0.5', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5')
        plt.title('Validation mAP@0.5')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'map_50_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory

        # mAP@0.5:0.95 plot
        plt.figure(figsize=(10, 6))
        plt.plot(val_maps_50_95, label='mAP@0.5:0.95', linewidth=2, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5:0.95')
        plt.title('Validation mAP@0.5:0.95')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'map_50_95_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory

        print(f"\n✅ Training plots saved successfully in: {output_dir}/")
        print(f"   - loss_plot.png")
        print(f"   - map_50_plot.png") 
        print(f"   - map_50_95_plot.png")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

def train_on_all_data_from_scratch(bin_dir, label_dir, config_path, output_dir, epochs=100, img_size=640, batch_size=16, device='cpu', augmentations=False, augmentation_factor=3):
    """
    Train a model from scratch on all data - Memory optimized version that saves directly to disk
    
    Args:
        bin_dir: Directory containing bin files
        label_dir: Directory containing label files
        config_path: Path to config file
        output_dir: Output directory
        epochs: Number of training epochs
        img_size: Image size
        batch_size: Batch size
        device: Device to use (cuda:0 or cpu)
        augmentations: Whether to use augmentations
        augmentation_factor: Number of augmented samples per original
        
    Returns:
        Path to best weights
    """
    print("\n===== TRAINING FROM SCRATCH ON ALL DATA (MEMORY OPTIMIZED) =====")
    print(f"Using bin directory: {bin_dir}")
    print(f"Using label directory: {label_dir}")
    print(f"Using augmentations: {augmentations}")
    
    # Create output directories with proper permissions
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create train and val directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create images and labels directories
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Create verification directory
    verification_dir = os.path.join(output_dir, 'verification')
    os.makedirs(verification_dir, exist_ok=True)
    
    # Create model output directory with explicit permissions
    model_output_dir = os.path.join(output_dir, 'bev-from-scratch')
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, 'train'), exist_ok=True)
    
    # Get list of ALL bin files first
    all_bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))
    print(f"Found {len(all_bin_files)} total bin files")
    
    # Filter bin files that have corresponding label files
    bin_files = []
    for bin_file in all_bin_files:
        label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.txt')
        if os.path.exists(label_file):
            bin_files.append(bin_file)
        else:
            print(f"Skipping {os.path.basename(bin_file)} - no corresponding label file")
    
    print(f"Found {len(bin_files)} bin files with corresponding labels")
    print(f"Skipped {len(all_bin_files) - len(bin_files)} bin files without labels")
    
    # Split bin files into train and val FIRST (before processing)
    random.shuffle(bin_files)
    train_val_split = 0.9
    split_idx = int(train_val_split * len(bin_files))
    train_bin_files = bin_files[:split_idx]
    val_bin_files = bin_files[split_idx:]
    
    print(f"Training bin files: {len(train_bin_files)}")
    print(f"Validation bin files: {len(val_bin_files)}")
    
    # Create processor
    processor = PointCloudProcessor(config_path=config_path)
    
    # Counters for saved files
    train_counter = 0
    val_counter = 0
    
    def save_sample_to_disk(bev_image, yolo_labels, sample_type, counter, suffix=""):
        """Save a single sample directly to disk"""
        if sample_type == 'train':
            img_path = os.path.join(train_img_dir, f"train_{counter}{suffix}.png")
            label_path = os.path.join(train_label_dir, f"train_{counter}{suffix}.txt")
        else:  # val
            img_path = os.path.join(val_img_dir, f"val_{counter}{suffix}.png")
            label_path = os.path.join(val_label_dir, f"val_{counter}{suffix}.txt")
        
        # Save image
        cv2.imwrite(img_path, bev_image)
        
        # Save labels
        with open(label_path, 'w') as f:
            for label_str in yolo_labels:
                f.write(label_str + '\n')
        
        return counter + 1
    
    # Import augmentation functions
    if augmentations:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_processing.augmentation import (
            rotate_points_and_labels,
            scale_distance_points_and_labels,
            shift_lateral_points_and_labels,
            shift_vertical_points_and_labels,
            create_range_adapted_bev_image
        )
        
        # Load config for create_range_adapted_bev_image
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    def process_augmentations(points, labels, bin_idx, bin_file, sample_type, counter, is_verification=False):
        """Process all augmentations for a given sample"""
        current_counter = counter
        
        # Augmentation 1: Rotate point cloud by different angles
        for angle in [-45, -30, -15, 15, 30, 45]:
            # Rotate points and labels
            rotated_points, rotated_labels = rotate_points_and_labels(points, labels, angle)
            
            # Create clean BEV image for training
            rotated_bev_image = processor.create_bev_image(rotated_points)
            
            # Process rotated objects and create YOLO labels
            rotated_yolo_labels = []
            
            for obj in rotated_labels:
                if obj['type'] == 'DontCare':
                    continue
                    
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], rotated_bev_image.shape[:2]
                )
                rotated_yolo_labels.append(yolo_label)
            
            # Only save if we have valid labels
            if rotated_yolo_labels:
                current_counter = save_sample_to_disk(rotated_bev_image, rotated_yolo_labels, sample_type, current_counter, f"_rotation_{angle}")
            
            # Create verification image with boxes (only for first few files)
            if is_verification and bin_idx < 5:
                rotated_bev_with_boxes = rotated_bev_image.copy()
                for obj in rotated_labels:
                    if obj['type'] == 'DontCare':
                        continue
                        
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    rotated_bev_with_boxes = processor.draw_box_on_bev(
                        rotated_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_{sample_type}_rotation_{angle}.png")
                cv2.imwrite(verify_path, rotated_bev_with_boxes)
            
            # Clear variables to free memory
            del rotated_points, rotated_labels, rotated_bev_image, rotated_yolo_labels
        
        # Augmentation 2: Scale distance (move objects closer/further)
        for scale in [-2.5, 2.5]:
            # Scale points and labels
            scaled_points, scaled_labels = scale_distance_points_and_labels(points, labels, scale)
            
            # Create clean BEV image for training
            scaled_bev_image = processor.create_bev_image(scaled_points)
            
            # Process scaled objects and create YOLO labels
            scaled_yolo_labels = []
            
            for obj in scaled_labels:
                if obj['type'] == 'DontCare':
                    continue
                    
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], scaled_bev_image.shape[:2]
                )
                scaled_yolo_labels.append(yolo_label)
            
            # Only save if we have valid labels
            if scaled_yolo_labels:
                current_counter = save_sample_to_disk(scaled_bev_image, scaled_yolo_labels, sample_type, current_counter, f"_scale_{scale}")
        
            # Create verification image with boxes (only for first few files)
            if is_verification and bin_idx < 5:
                scaled_bev_with_boxes = scaled_bev_image.copy()
                for obj in scaled_labels:
                    if obj['type'] == 'DontCare':
                        continue
                    
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    scaled_bev_with_boxes = processor.draw_box_on_bev(
                        scaled_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_{sample_type}_scale_{scale}.png")
                cv2.imwrite(verify_path, scaled_bev_with_boxes)
            
            # Clear variables to free memory
            del scaled_points, scaled_labels, scaled_bev_image, scaled_yolo_labels
        
        # Augmentation 3: Shift laterally (left/right)
        for x_shift in [-2.5, 2.5]:
            # Shift points and labels
            shifted_points, shifted_labels = shift_lateral_points_and_labels(points, labels, x_shift)
            
            # Create clean BEV image for training
            shifted_bev_image = processor.create_bev_image(shifted_points)
            
            # Process shifted objects and create YOLO labels
            shifted_yolo_labels = []
            
            for obj in shifted_labels:
                if obj['type'] == 'DontCare':
                    continue
                    
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], shifted_bev_image.shape[:2]
                )
                shifted_yolo_labels.append(yolo_label)
            
            # Only save if we have valid labels
            if shifted_yolo_labels:
                current_counter = save_sample_to_disk(shifted_bev_image, shifted_yolo_labels, sample_type, current_counter, f"_x_shift_{x_shift}")
            
            # Create verification image with boxes (only for first few files)
            if is_verification and bin_idx < 5:
                shifted_bev_with_boxes = shifted_bev_image.copy()
                for obj in shifted_labels:
                    if obj['type'] == 'DontCare':
                        continue
                    
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    shifted_bev_with_boxes = processor.draw_box_on_bev(
                        shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_{sample_type}_x_shift_{x_shift}.png")
                cv2.imwrite(verify_path, shifted_bev_with_boxes)
            
            # Clear variables to free memory
            del shifted_points, shifted_labels, shifted_bev_image, shifted_yolo_labels
        
        # Augmentation 4: Fixed range zoom-in views
        # Define 2 fixed regions to zoom in (reduced for memory)
        fixed_regions = [
            {"x_min": 0, "x_max": 10, "y_min": -5, "y_max": 5, "name": "front_center"},
            {"x_min": 10, "x_max": 20, "y_min": -5, "y_max": 5, "name": "mid_center"}
        ]
        
        for region in fixed_regions:
            # Create clean zoomed BEV image, BEV with boxes, and YOLO labels for this fixed region
            zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels = create_range_adapted_bev_image(
                points, labels, 
                region["x_min"], region["x_max"], 
                region["y_min"], region["y_max"], 
                config
            )
            
            # Only save if we have valid labels
            if zoom_yolo_labels:
                current_counter = save_sample_to_disk(zoom_bev_clean, zoom_yolo_labels, sample_type, current_counter, f"_zoom_{region['name']}")
                
                # For verification, use the image with boxes (only for first few files)
                if is_verification and bin_idx < 5:
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_{sample_type}_zoom_{region['name']}.png")
                    cv2.imwrite(verify_path, zoom_bev_with_boxes)
            
            # Clear variables to free memory
            del zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels
        
        # Augmentation 5: Height shift
        for height_shift in [-30, -20, -10, -5, 5, 10, 20, 30]:
            # Convert cm to meters
            height_shift_m = height_shift / 100.0
            
            # Apply height shift to points using the existing function
            shifted_points, shifted_labels = shift_vertical_points_and_labels(points, labels, height_shift_m)
            
            # Create clean BEV image for training
            shifted_bev_image = processor.create_bev_image(shifted_points)
            
            # Process objects and create YOLO labels
            shifted_yolo_labels = []
            
            for obj in shifted_labels:
                if obj['type'] == 'DontCare':
                    continue
                    
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], shifted_bev_image.shape[:2]
                )
                shifted_yolo_labels.append(yolo_label)
            
            # Only save if we have valid labels
            if shifted_yolo_labels:
                current_counter = save_sample_to_disk(shifted_bev_image, shifted_yolo_labels, sample_type, current_counter, f"_height_shift_{height_shift}")
            
            # Create verification image with boxes (only for first few files)
            if is_verification and bin_idx < 5:
                shifted_bev_with_boxes = shifted_bev_image.copy()
                for obj in shifted_labels:
                    if obj['type'] == 'DontCare':
                        continue
                    
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    shifted_bev_with_boxes = processor.draw_box_on_bev(
                        shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_{sample_type}_height_shift_{height_shift}.png")
                cv2.imwrite(verify_path, shifted_bev_with_boxes)
            
            # Clear variables to free memory
            del shifted_points, shifted_labels, shifted_bev_image, shifted_yolo_labels
        
        return current_counter
    
    # Process training bin files
    print("\nProcessing training bin files...")
    for bin_idx, bin_file in enumerate(tqdm(train_bin_files, desc="Processing training bin files")):
        try:
            # Get label file path
            label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.txt')
            
            # Process point cloud to get original BEV image (without boxes)
            points = processor.load_point_cloud(bin_file)
            labels = processor.load_labels(label_file)
            
            # Create clean BEV image for training
            bev_image = processor.create_bev_image(points)
            
            # Create YOLO labels
            yolo_labels = []
            
            for obj in labels:
                if obj['type'] == 'DontCare':
                    continue
                
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], bev_image.shape[:2]
                )
                yolo_labels.append(yolo_label)
            
            # Save original sample directly to disk
            train_counter = save_sample_to_disk(bev_image, yolo_labels, 'train', train_counter, "_original")
            
            # Create verification image with boxes (only for first few files)
            if bin_idx < 20:
                bev_with_boxes = bev_image.copy()
                for obj in labels:
                    if obj['type'] == 'DontCare':
                        continue
                    
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    bev_with_boxes = processor.draw_box_on_bev(
                        bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_train_original.png")
                cv2.imwrite(verify_path, bev_with_boxes)
            
            # Create augmentations if enabled
            if augmentations:
                train_counter = process_augmentations(points, labels, bin_idx, bin_file, 'train', train_counter, is_verification=True)
            
            # Clear main variables after each bin file to free memory
            del points, labels, bev_image, yolo_labels
            
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue
    
    # Process validation bin files
    print("\nProcessing validation bin files...")
    for bin_idx, bin_file in enumerate(tqdm(val_bin_files, desc="Processing validation bin files")):
        try:
            # Get label file path
            label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.txt')
            
            # Process point cloud to get original BEV image (without boxes)
            points = processor.load_point_cloud(bin_file)
            labels = processor.load_labels(label_file)
            
            # Create clean BEV image for training
            bev_image = processor.create_bev_image(points)
            
            # Create YOLO labels
            yolo_labels = []
            
            for obj in labels:
                if obj['type'] == 'DontCare':
                    continue
                
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], bev_image.shape[:2]
                )
                yolo_labels.append(yolo_label)
            
            # Save original sample directly to disk
            val_counter = save_sample_to_disk(bev_image, yolo_labels, 'val', val_counter, "_original")
            
            # Create verification image with boxes (only for first few files)
            if bin_idx < 20:
                bev_with_boxes = bev_image.copy()
                for obj in labels:
                    if obj['type'] == 'DontCare':
                        continue
                    
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    bev_with_boxes = processor.draw_box_on_bev(
                        bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_val_original.png")
                cv2.imwrite(verify_path, bev_with_boxes)
            
            # Create augmentations if enabled - SAME AS TRAINING NOW!
            if augmentations:
                val_counter = process_augmentations(points, labels, bin_idx, bin_file, 'val', val_counter, is_verification=True)
            
            # Clear main variables after each bin file to free memory
            del points, labels, bev_image, yolo_labels
            
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue
    
    print(f"\nSuccessfully saved:")
    print(f"  Training samples: {train_counter}")
    print(f"  Validation samples: {val_counter}")
    print(f"  Total samples: {train_counter + val_counter}")
    
    # Create custom YAML dataset configuration
    dataset_config = {
        'path': dataset_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': 5,  # Updated: 5 classes instead of 6 (removed DontCare)
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Bus', 'Truck']  # Updated class order without DontCare
    }
    
    # Create a temporary YAML file with dataset configuration
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Check CUDA availability and use appropriate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    # Create a new model from scratch
    print("\n===== CREATING NEW MODEL FROM SCRATCH =====")
    model = YOLO('yolov8n.yaml')  # Create a new YOLOv8 model from scratch
    
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
        'cache': 'disk',  # Use disk cache to save memory
        'data': dataset_yaml,
        'patience': 50,  # Early stopping patience
        'save_period': 10  # Save checkpoint every 10 epochs
    }
    
    print(f"Using device: {device}")
    
    # Train the model
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    results = model.train(**train_args)
    
    # Extract training metrics from results
    try:
        # Get the training results directory
        results_dir = os.path.join(output_dir, 'bev-from-scratch', 'train')
        
        # Read the results.csv file that YOLO creates
        results_csv = os.path.join(results_dir, 'results.csv')
        
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            
            # Extract metrics (adjust column names based on actual CSV structure)
            train_losses = df['train/box_loss'].tolist() if 'train/box_loss' in df.columns else []
            val_losses = df['val/box_loss'].tolist() if 'val/box_loss' in df.columns else []
            val_maps_50 = df['metrics/mAP50(B)'].tolist() if 'metrics/mAP50(B)' in df.columns else []
            val_maps_50_95 = df['metrics/mAP50-95(B)'].tolist() if 'metrics/mAP50-95(B)' in df.columns else []
            
            # Create plots
            plot_output_dir = os.path.join(output_dir, 'training_plots')
            plot_metrics(train_losses, val_losses, val_maps_50, val_maps_50_95, plot_output_dir)
        else:
            print(f"Warning: Results CSV not found at {results_csv}")
            
    except Exception as e:
        print(f"Warning: Could not create training plots: {e}")
    
    # Return the path to best weights
    weights_dir = os.path.join(output_dir, 'bev-from-scratch', 'train', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    
    # Also save a copy with a more descriptive name
    final_weights = os.path.join(output_dir, 'bev_all_data_from_scratch_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    
    # Run inference with the trained model on a validation image (load from disk)
    print("\n===== RUNNING INFERENCE WITH TRAINED MODEL =====")
    
    # Choose a random validation image from disk
    val_images = glob.glob(os.path.join(val_img_dir, "*.png"))
    if val_images:
        val_img_path = random.choice(val_images)
        val_img = cv2.imread(val_img_path)
        
        # Get corresponding label file
        label_filename = os.path.basename(val_img_path).replace('.png', '.txt')
        val_label_path = os.path.join(val_label_dir, label_filename)
        
        val_labels = []
        if os.path.exists(val_label_path):
            with open(val_label_path, 'r') as f:
                val_labels = [line.strip() for line in f.readlines()]
        
        # Run inference with no display
        results = model.predict(val_img, conf=0.25, save=False, show=False)
        
        # Get predictions without plotting GUI
        if results and len(results) > 0:
            result = results[0]
            
            # Create result image manually without GUI
            result_img = val_img.copy()
            
            # Draw predictions on the image manually
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label and confidence
                    label = f"Class_{cls_id}: {conf:.2f}"
                    cv2.putText(result_img, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save result image
            result_path = os.path.join(verification_dir, "model_prediction.png")
            cv2.imwrite(result_path, result_img)
            
            print(f"Inference result saved to: {result_path}")
            
            # Create a comparison image with ground truth
            gt_img = val_img.copy()
            for label_str in val_labels:
                parts = label_str.split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert normalized coordinates to pixel coordinates
                    img_h, img_w = gt_img.shape[:2]
                    x_center_px = int(x_center * img_w)
                    y_center_px = int(y_center * img_h)
                    width_px = int(width * img_w)
                    height_px = int(height * img_h)
                    
                    # Calculate box corners
                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)
                    
                    # Draw box
                    color = processor.colors[processor.class_names[class_id]]
                    cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    cv2.putText(gt_img, processor.class_names[class_id], 
                              (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save ground truth image
            gt_path = os.path.join(verification_dir, "ground_truth.png")
            cv2.imwrite(gt_path, gt_img)
            
            # Create side-by-side comparison
            comparison = np.hstack((gt_img, result_img))
            comparison_path = os.path.join(verification_dir, "gt_vs_prediction.png")
            cv2.imwrite(comparison_path, comparison)
            
            print(f"Comparison image saved to: {comparison_path}")
        
        else:
            print("No predictions made")
    
    print("\n===== TRAINING COMPLETE - MEMORY OPTIMIZED =====")
    return best_weights


def train_from_checkpoint(bin_dir, label_dir, config_path, output_dir, checkpoint_path='/lidar3d_detection_ws/train/output/best.pt', epochs=100, img_size=640, batch_size=16, device='cpu', augmentations=False, augmentation_factor=3):
    """
    Continue training a YOLO model from a checkpoint
    
    Args:
        bin_dir: Directory containing bin files
        label_dir: Directory containing label files
        config_path: Path to config file
        output_dir: Output directory
        checkpoint_path: Path to model checkpoint to continue from
        epochs: Number of training epochs
        img_size: Image size
        batch_size: Batch size
        device: Device to use (cuda:0 or cpu)
        augmentations: Whether to use augmentations
        augmentation_factor: Number of augmented samples per original
        
    Returns:
        Path to best weights
    """
    print("\n===== CONTINUING TRAINING FROM CHECKPOINT =====")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using bin directory: {bin_dir}")
    print(f"Using label directory: {label_dir}")
    print(f"Using augmentations: {augmentations}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset directory
    dataset_dir = os.path.join(output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create train and val directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create images and labels directories
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Create verification directory
    verification_dir = os.path.join(output_dir, 'verification')
    os.makedirs(verification_dir, exist_ok=True)
    
    # Get list of bin files
    bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))
    print(f"Found {len(bin_files)} bin files")
    
    # Create processor
    processor = PointCloudProcessor(config_path=config_path)
    
    # Process each bin file
    all_data = []
    
    for bin_idx, bin_file in enumerate(tqdm(bin_files, desc="Processing bin files")):
        try:
            # Get label file path
            label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.txt')
            
            # Process point cloud to get clean BEV image (without boxes)
            points = processor.load_point_cloud(bin_file)
            labels = processor.load_labels(label_file)
            
            # Create clean BEV image for training
            bev_image = processor.create_bev_image(points)
            
            # Create YOLO labels
            yolo_labels = []
            
            for obj in labels:
                if obj['type'] == 'DontCare':
                    continue
                
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], bev_image.shape[:2]
                )
                yolo_labels.append(yolo_label)
            
            # Create verification image with boxes
            bev_with_boxes = bev_image.copy()
            for obj in labels:
                if obj['type'] == 'DontCare':
                    continue
                
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                bev_with_boxes = processor.draw_box_on_bev(
                    bev_with_boxes, corners_bev, center_bev, obj['type']
                )
            
            # Save verification image
            if bin_idx < 20:  # Save the first 20 images for verification
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_original.png")
                cv2.imwrite(verify_path, bev_with_boxes)
            
            # Add clean image to dataset
            all_data.append({
                'bin_file': bin_file,
                'bev_image': bev_image,  # Clean image without boxes
                'yolo_labels': yolo_labels
            })
            
            # Create augmentations if enabled
            if augmentations:
                # Import augmentation functions
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from data_processing.augmentation import (
                    rotate_points_and_labels,
                    scale_distance_points_and_labels,
                    shift_lateral_points_and_labels,
                    create_range_adapted_bev_image
                )
                
                # Augmentation 1: Rotate point cloud by different angles
                for angle in [-15, 15]:
                    # Rotate points and labels
                    rotated_points, rotated_labels = rotate_points_and_labels(points, labels, angle)
                    
                    # Create clean BEV image for training
                    rotated_bev_image = processor.create_bev_image(rotated_points)
                    
                    # Process rotated objects and create YOLO labels
                    rotated_yolo_labels = []
                    
                    for obj in rotated_labels:
                        if obj['type'] == 'DontCare':
                            continue
                            
                        # Transform 3D box to BEV
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            obj['dimensions'], obj['location'], obj['rotation_y']
                        )
                        
                        # Create YOLO label
                        yolo_label = processor.create_yolo_label(
                            corners_bev, obj['type'], rotated_bev_image.shape[:2]
                        )
                        rotated_yolo_labels.append(yolo_label)
                    
                    # Create verification image with boxes
                    rotated_bev_with_boxes = rotated_bev_image.copy()
                    for obj in rotated_labels:
                        if obj['type'] == 'DontCare':
                            continue
                            
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            obj['dimensions'], obj['location'], obj['rotation_y']
                        )
                        rotated_bev_with_boxes = processor.draw_box_on_bev(
                            rotated_bev_with_boxes, corners_bev, center_bev, obj['type']
                        )
                    
                    # Only add if we have valid labels
                    if rotated_yolo_labels:
                        all_data.append({
                            'bin_file': bin_file,
                            'bev_image': rotated_bev_image,  # Clean image without boxes
                            'yolo_labels': rotated_yolo_labels,
                            'augmentation': f'rotation_{angle}'
                        })
                    
                    # Save verification image if needed
                    if bin_idx < 5:
                        verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_rotation_{angle}.png")
                        cv2.imwrite(verify_path, rotated_bev_with_boxes)
                
                # Augmentation 2: Scale distance (move objects closer/further)
                for scale in [-2.5, 2.5]:
                    # Scale points and labels
                    scaled_points, scaled_labels = scale_distance_points_and_labels(points, labels, scale)
                    
                    # Create clean BEV image for training
                    scaled_bev_image = processor.create_bev_image(scaled_points)
                    
                    # Process scaled objects and create YOLO labels
                    scaled_yolo_labels = []
                    
                    for obj in scaled_labels:
                        if obj['type'] == 'DontCare':
                            continue
                            
                        # Transform 3D box to BEV
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            obj['dimensions'], obj['location'], obj['rotation_y']
                        )
                        
                        # Create YOLO label
                        yolo_label = processor.create_yolo_label(
                            corners_bev, obj['type'], scaled_bev_image.shape[:2]
                        )
                        scaled_yolo_labels.append(yolo_label)
                    
                    # Create verification image with boxes
                    scaled_bev_with_boxes = scaled_bev_image.copy()
                    for obj in scaled_labels:
                        if obj['type'] == 'DontCare':
                            continue
                        
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            obj['dimensions'], obj['location'], obj['rotation_y']
                        )
                        scaled_bev_with_boxes = processor.draw_box_on_bev(
                            scaled_bev_with_boxes, corners_bev, center_bev, obj['type']
                        )
                    
                    # Only add if we have valid labels
                    if scaled_yolo_labels:
                        all_data.append({
                            'bin_file': bin_file,
                            'bev_image': scaled_bev_image,  # Clean image without boxes
                            'yolo_labels': scaled_yolo_labels,
                            'augmentation': f'scale_{scale}'
                        })
                
                    # Save verification image if needed
                    if bin_idx < 5:
                        verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_scale_{scale}.png")
                        cv2.imwrite(verify_path, scaled_bev_with_boxes)
                
                # Augmentation 3: Shift laterally (left/right)
                for x_shift in [-2.5, 2.5]:
                    # Shift points and labels
                    shifted_points, shifted_labels = shift_lateral_points_and_labels(points, labels, x_shift)
                    
                    # Create clean BEV image for training
                    shifted_bev_image = processor.create_bev_image(shifted_points)
                    
                    # Process shifted objects and create YOLO labels
                    shifted_yolo_labels = []
                    
                    for obj in shifted_labels:
                        if obj['type'] == 'DontCare':
                            continue
                            
                        # Transform 3D box to BEV
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            obj['dimensions'], obj['location'], obj['rotation_y']
                        )
                        
                        # Create YOLO label
                        yolo_label = processor.create_yolo_label(
                            corners_bev, obj['type'], shifted_bev_image.shape[:2]
                        )
                        shifted_yolo_labels.append(yolo_label)
                    
                    # Create verification image with boxes
                    shifted_bev_with_boxes = shifted_bev_image.copy()
                    for obj in shifted_labels:
                        if obj['type'] == 'DontCare':
                            continue
                        
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            obj['dimensions'], obj['location'], obj['rotation_y']
                        )
                        shifted_bev_with_boxes = processor.draw_box_on_bev(
                            shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                        )
                    
                    # Only add if we have valid labels
                    if shifted_yolo_labels:
                        all_data.append({
                            'bin_file': bin_file,
                            'bev_image': shifted_bev_image,  # Clean image without boxes
                            'yolo_labels': shifted_yolo_labels,
                            'augmentation': f'x_shift_{x_shift}'
                        })
                    
                    # Save verification image if needed
                    if bin_idx < 5:
                        verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_x_shift_{x_shift}.png")
                        cv2.imwrite(verify_path, shifted_bev_with_boxes)
                
                # Augmentation 4: Fixed range zoom-in views
                # Define 7 fixed regions to zoom in
                fixed_regions = [
                    {"x_min": 0, "x_max": 10, "y_min": -5, "y_max": 5, "name": "front_center"},
                    {"x_min": 10, "x_max": 20, "y_min": -5, "y_max": 5, "name": "mid_center"}
                ]
                
                # Load config for create_range_adapted_bev_image
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                for region in fixed_regions:
                    # Create clean zoomed BEV image, BEV with boxes, and YOLO labels for this fixed region
                    zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels = create_range_adapted_bev_image(
                        points, labels, 
                        region["x_min"], region["x_max"], 
                        region["y_min"], region["y_max"], 
                        config
                    )
                    
                    # Create verification image with boxes
                    if zoom_yolo_labels:
                        # Only add if we have valid labels
                        all_data.append({
                            'bin_file': bin_file,
                            'bev_image': zoom_bev_clean,  # Use the clean image for training
                            'yolo_labels': zoom_yolo_labels,
                            'augmentation': f'zoom_{region["name"]}'
                        })
                        
                        # For verification, use the image with boxes that was already created
                        if bin_idx < 5:
                            verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_zoom_{region['name']}.png")
                            cv2.imwrite(verify_path, zoom_bev_with_boxes)
                
                print(f"Created {len(all_data) - 1} augmented samples for {os.path.basename(bin_file)}")
            
            # Augmentation 5: Height shift (new)
            for height_shift in [-10, -5, 5, 10]:
                # Convert cm to meters
                height_shift_m = height_shift / 100.0
                
                # Apply height shift to points using the existing function
                shifted_points, shifted_labels = shift_vertical_points_and_labels(points, labels, height_shift_m)
                
                # Create clean BEV image for training
                shifted_bev_image = processor.create_bev_image(shifted_points)
                
                # Process objects and create YOLO labels
                shifted_yolo_labels = []
                
                for obj in shifted_labels:
                    if obj['type'] == 'DontCare':
                        continue
                        
                    # Transform 3D box to BEV
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    
                    # Create YOLO label
                    yolo_label = processor.create_yolo_label(
                        corners_bev, obj['type'], shifted_bev_image.shape[:2]
                    )
                    shifted_yolo_labels.append(yolo_label)
                
                # Create verification image with boxes
                shifted_bev_with_boxes = shifted_bev_image.copy()
                for obj in shifted_labels:
                    if obj['type'] == 'DontCare':
                        continue
                    
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    shifted_bev_with_boxes = processor.draw_box_on_bev(
                        shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                
                # Only add if we have valid labels
                if shifted_yolo_labels:
                    all_data.append({
                        'bin_file': bin_file,
                        'bev_image': shifted_bev_image,  # Clean image without boxes
                        'yolo_labels': shifted_yolo_labels,
                        'augmentation': f'height_shift_{height_shift}'
                    })
                
                # Save verification image if needed
                if bin_idx < 5:
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_height_shift_{height_shift}.png")
                    cv2.imwrite(verify_path, shifted_bev_with_boxes)
            
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue
    load_config
    print(f"Successfully processed {len(all_data)} samples (original + augmented)")
    
    # Split into train/val sets
    # Make sure samples from the same bin file don't appear in both train and val
    unique_bin_files = list(set(item['bin_file'] for item in all_data))
    random.shuffle(unique_bin_files)
    
    # Split ratio
    train_val_split = 0.8
    
    # Split bin files into train and val
    train_bin_files = unique_bin_files[:int(train_val_split * len(unique_bin_files))]
    val_bin_files = unique_bin_files[int(train_val_split * len(unique_bin_files)):]
    
    # Split data based on bin files
    train_data = [item for item in all_data if item['bin_file'] in train_bin_files]
    val_data = [item for item in all_data if item['bin_file'] in val_bin_files]
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    # Save images and labels for training
    for i, item in enumerate(tqdm(train_data, desc="Saving training data")):
        train_img_path = os.path.join(train_img_dir, f"train_{i}.png")
        train_label_path = os.path.join(train_label_dir, f"train_{i}.txt")
        
        cv2.imwrite(train_img_path, item['bev_image'])  # Clean image without boxes
        with open(train_label_path, 'w') as f:
            for label_str in item['yolo_labels']:
                f.write(label_str + '\n')
    
    # Save images and labels for validation
    for i, item in enumerate(tqdm(val_data, desc="Saving validation data")):
        val_img_path = os.path.join(val_img_dir, f"val_{i}.png")
        val_label_path = os.path.join(val_label_dir, f"val_{i}.txt")
        
        cv2.imwrite(val_img_path, item['bev_image'])  # Clean image without boxes
        with open(val_label_path, 'w') as f:
            for label_str in item['yolo_labels']:
                f.write(label_str + '\n')
    
    # Create custom YAML dataset configuration
    dataset_config = {
        'path': dataset_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': 5,  # Updated: 5 classes instead of 6 (removed DontCare)
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Bus', 'Truck']  # Updated class order without DontCare
    }
    
    # Create a temporary YAML file with dataset configuration
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Check CUDA availability and use appropriate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    # Load the model from checkpoint
    print(f"\n===== LOADING MODEL FROM CHECKPOINT: {checkpoint_path} =====")
    model = YOLO(checkpoint_path)
    
    # Prepare training arguments
    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'workers': 8,
        'project': os.path.join(output_dir, 'bev-continued'),
        'name': 'train',
        'exist_ok': True,
        'optimizer': 'SGD',  # or 'Adam'
        'lr0': 0.001,  # Lower learning rate for fine-tuning
        'weight_decay': 0.0005,
        'cache': 'disk',
        'data': dataset_yaml,
        'patience': 10,  # Early stopping patience
        'save_period': 10  # Save checkpoint every 10 epochs
    }
    
    print(f"Using device: {device}")
    
    # Train the model
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    results = model.train(**train_args)
    
    # Extract training metrics from results
    try:
        # Get the training results directory
        results_dir = os.path.join(output_dir, 'bev-continued', 'train')
        
        # Read the results.csv file that YOLO creates
        results_csv = os.path.join(results_dir, 'results.csv')
        
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
            
            # Extract metrics (adjust column names based on actual CSV structure)
            train_losses = df['train/box_loss'].tolist() if 'train/box_loss' in df.columns else []
            val_losses = df['val/box_loss'].tolist() if 'val/box_loss' in df.columns else []
            val_maps_50 = df['metrics/mAP50(B)'].tolist() if 'metrics/mAP50(B)' in df.columns else []
            val_maps_50_95 = df['metrics/mAP50-95(B)'].tolist() if 'metrics/mAP50-95(B)' in df.columns else []
            
            # Create plots
            plot_output_dir = os.path.join(output_dir, 'training_plots')
            plot_metrics(train_losses, val_losses, val_maps_50, val_maps_50_95, plot_output_dir)
        else:
            print(f"Warning: Results CSV not found at {results_csv}")
            
    except Exception as e:
        print(f"Warning: Could not create training plots: {e}")
    
    # Return the path to best weights
    weights_dir = os.path.join(output_dir, 'bev-continued', 'train', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    
    # Also save a copy with a more descriptive name
    final_weights = os.path.join(output_dir, 'bev_continued_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    
    return best_weights

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train YOLO model on BEV images')
    
    # Dataset paths
    parser.add_argument('--bin_dir', type=str, default='/lidar3d_detection_ws/data/innoviz/', 
                        help='Directory containing bin files')
    parser.add_argument('--label_dir', type=str, default='/lidar3d_detection_ws/data/labels/', 
                        help='Directory containing label files')
    parser.add_argument('--config_path', type=str, default='/lidar3d_detection_ws/train/config/preprocessing_config.yaml', 
                        help='Path to preprocessing config file')
    parser.add_argument('--output_base', type=str, default='/lidar3d_detection_ws/train', 
                        help='Base directory for output')
    
    # Training mode (required - must choose one)
    training_mode = parser.add_mutually_exclusive_group(required=True)
    training_mode.add_argument('--all_data_from_scratch', action='store_true',
                              help='Train from scratch on all data')
    training_mode.add_argument('--continue_training', action='store_true',
                              help='Continue training from a checkpoint')
    
    # Checkpoint path (required if continue_training is selected)
    parser.add_argument('--checkpoint_path', type=str, default='/lidar3d_detection_ws/train/output/best.pt',
                        help='Path to checkpoint for continuing training (required with --continue_training)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='Image size')
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device to use (cuda:0, cpu, or auto)')
    
    # Augmentation options
    parser.add_argument('--augmentations', action='store_true', 
                        help='Enable data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=3, 
                        help='Augmentation factor (multiplier for dataset size)')

    args = parser.parse_args()

    # Validate checkpoint path if continuing training
    if args.continue_training and not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return

    # Handle device selection - auto-detect if set to "auto"
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Auto-selected device: {args.device}")
    
    # Create output directory
    output_dir = os.path.join(args.output_base, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Print training configuration
    print("\n===== TRAINING CONFIGURATION =====")
    print(f"Training mode: {'From scratch' if args.all_data_from_scratch else 'Continue from checkpoint'}")
    if args.continue_training:
        print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Bin directory: {args.bin_dir}")
    print(f"Label directory: {args.label_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    print(f"Augmentations: {'Enabled' if args.augmentations else 'Disabled'}")
    if args.augmentations:
        print(f"Augmentation factor: {args.augmentation_factor}")
    print("=" * 35)
    
    # Train the model
    if args.all_data_from_scratch:
        print("\nStarting training from scratch...")
        best_weights = train_on_all_data_from_scratch(
            bin_dir=args.bin_dir,
            label_dir=args.label_dir,
            config_path=args.config_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            augmentations=args.augmentations,
            augmentation_factor=args.augmentation_factor
        )
    
    elif args.continue_training:
        print("\nContinuing training from checkpoint...")
        best_weights = train_from_checkpoint(
            bin_dir=args.bin_dir,
            label_dir=args.label_dir,
            config_path=args.config_path,
            output_dir=output_dir,
            checkpoint_path=args.checkpoint_path,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            augmentations=args.augmentations,
            augmentation_factor=args.augmentation_factor
        )
    
    print(f"\n===== TRAINING COMPLETE =====")
    print(f"Best weights saved to: {best_weights}")
    print("=" * 30)

if __name__ == "__main__":
    main()