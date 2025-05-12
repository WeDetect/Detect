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

# Global configuration
DEFAULT_CONFIG = {
    'bin_dir': '/lidar3d_detection_ws/data/innoviz',
    'label_dir': '/lidar3d_detection_ws/data/labels',
    'config_path': '/lidar3d_detection_ws/train/config/preprocessing_config.yaml',
    'output_base': '/lidar3d_detection_ws/train/dataset',
    'train_val_split': 0.8,  # 80% training, 20% validation
    'epochs': 70,
    'batch_size': 8,
    'img_size': 608,
    'augmentations': False,
    'augmentation_factor': 3,  # Number of augmented samples per original
    'device': 'cpu',  # 'cpu' or 'cuda:0'
}

def generate_standard_dataset(bin_dir, label_dir, config_path, output_dir, img_size):
    """
    Generate standard BEV dataset without augmentation
    
    Args:
        bin_dir: Directory with .bin files
        label_dir: Directory with label .txt files
        config_path: Path to preprocessing config YAML
        output_dir: Directory to save output data
        img_size: Image size for output
        
    Returns:
        List of processed (bev_image, yolo_labels) pairs
    """
    config = load_config(config_path)
    dataset_items = []
    
    bin_files = sorted(Path(bin_dir).glob("*.bin"))
    print(f"Found {len(bin_files)} bin files")
    
    for i, bin_file in enumerate(tqdm(bin_files, desc="Processing standard dataset")):
        label_file = Path(label_dir) / f"{bin_file.stem}.txt"
        
        # Skip if label file doesn't exist
        if not label_file.exists():
            print(f"Warning: {label_file} not found, skipping {bin_file}")
            continue
        
        # Load point cloud and labels
        points = read_bin_file(bin_file)
        labels = read_label_file(label_file)
        
        # Generate BEV image (without annotations)
        bev_image, _ = create_bev_image(points, config)
        
        # Convert KITTI labels to YOLO format
        yolo_labels = convert_labels_to_yolo_format(labels, config, bev_image.shape[1], bev_image.shape[0])
        
        # Store in memory (no saving to disk)
        dataset_items.append((bev_image, yolo_labels))
        
        # Debug information for first few files
        if i < 5:
            print(f"Processed item {i}: {bin_file.stem}, with {len(yolo_labels)} labels")
    
    print(f"Processed {len(dataset_items)} standard dataset items")
    return dataset_items

def apply_visualize_style_augmentations(points, labels, config):
    """
    Create augmentations for training, including zoom, rotation, height changes, and shifts
    
    Args:
        points: Point cloud data
        labels: Label data
        config: Configuration dictionary
        
    Returns:
        List of augmented (bev_image, yolo_labels) pairs
    """
    augmented_data = []
    
    # 1. First augmentation: 10x10 meter crops in random locations
    # Original crop: 0-10m in X axis, -5 to 5m in Y axis
    x_crops = [(0, 10), (5, 15), (10, 20), (15, 25)]
    y_crops = [(-5, 5), (-10, 0), (0, 10), (-15, -5), (5, 15)]
    
    # Add random crops
    for i in range(3):  # Generate 3 random crops
        x_min = random.uniform(0, 20)
        x_max = x_min + 10
        y_min = random.uniform(-15, 5)
        y_max = y_min + 10
        
        # Generate cropped BEV image
        cropped_bev, cropped_yolo = create_range_adapted_bev_image(
            points, labels, x_min, x_max, y_min, y_max, config
        )
        if len(cropped_yolo) > 0:  # Only add if there are objects in this crop
            augmented_data.append((cropped_bev, cropped_yolo))
    
    # Add some fixed crops that are likely to contain objects
    for x_min, x_max in x_crops:
        for y_min, y_max in y_crops:
            cropped_bev, cropped_yolo = create_range_adapted_bev_image(
                points, labels, x_min, x_max, y_min, y_max, config
            )
            if len(cropped_yolo) > 0:  # Only add if there are objects in this crop
                augmented_data.append((cropped_bev, cropped_yolo))
                # Just add one good crop to avoid too many samples
                break
        
    # 2. Height adjustments (-1m to +1m in 20cm increments)
    height_offsets = np.arange(-1.0, 1.2, 0.2)  # -1.0, -0.8, ..., 0.8, 1.0
    
    for offset in height_offsets:
        if offset == 0:  # Skip original height
            continue
            
        # Apply height shift
        shifted_points, shifted_labels = shift_vertical_points_and_labels(points.copy(), labels.copy(), offset)
        
        # Generate BEV image with adjusted height
        height_bev, _ = create_bev_image(shifted_points, config)
        height_yolo = convert_labels_to_yolo_format(shifted_labels, config, 
                                                   height_bev.shape[1], height_bev.shape[0])
        augmented_data.append((height_bev, height_yolo))
    
    # 3. Rotations (45 and 90 degree increments)
    rotation_angles = [45, 90, 135, 180, 225, 270, 315]
    
    for angle in rotation_angles:
        # Apply rotation
        rotated_points, rotated_labels = rotate_points_and_labels(points.copy(), labels.copy(), angle)
        
        # Generate BEV image with rotation
        rotated_bev, _ = create_bev_image(rotated_points, config)
        rotated_yolo = convert_labels_to_yolo_format(rotated_labels, config, 
                                                    rotated_bev.shape[1], rotated_bev.shape[0])
        augmented_data.append((rotated_bev, rotated_yolo))
    
    # 4. Lateral shifts (left-right)
    lateral_shifts = [-2.0, -1.0, 1.0, 2.0]  # meters
    
    for shift in lateral_shifts:
        # Apply lateral shift
        lateral_points, lateral_labels = shift_lateral_points_and_labels(points.copy(), labels.copy(), shift)
        
        # Generate BEV image with lateral shift
        lateral_bev, _ = create_bev_image(lateral_points, config)
        lateral_yolo = convert_labels_to_yolo_format(lateral_labels, config, 
                                                   lateral_bev.shape[1], lateral_bev.shape[0])
        augmented_data.append((lateral_bev, lateral_yolo))
    
    # 5. Forward-backward shifts
    # This is equivalent to shifting all points backward/forward
    fwd_shifts = [-3.0, -1.5, 1.5, 3.0]  # meters
    
    for shift in fwd_shifts:
        # Apply forward/backward shift (negative X is forward in KITTI)
        fwd_points = points.copy()
        fwd_points[:, 0] += shift  # Shift X coordinate
        
        fwd_labels = labels.copy()
        for label in fwd_labels:
            label['location'][0] += shift  # Shift X location
        
        # Generate BEV image with forward/backward shift
        fwd_bev, _ = create_bev_image(fwd_points, config)
        fwd_yolo = convert_labels_to_yolo_format(fwd_labels, config, 
                                               fwd_bev.shape[1], fwd_bev.shape[0])
        augmented_data.append((fwd_bev, fwd_yolo))
    
    print(f"Created {len(augmented_data)} augmented data items")
    return augmented_data

def generate_augmented_dataset(bin_dir, label_dir, config_path, output_dir, img_size, augmentation_factor=3):
    """
    Generate augmented BEV dataset
    
    Args:
        bin_dir: Directory with .bin files
        label_dir: Directory with label .txt files
        config_path: Path to preprocessing config YAML
        output_dir: Directory to save output data
        img_size: Image size for output
        augmentation_factor: Number of augmented samples per original
        
    Returns:
        List of augmented (bev_image, yolo_labels) pairs
    """
    config = load_config(config_path)
    augmented_items = []
    
    bin_files = sorted(Path(bin_dir).glob("*.bin"))
    print(f"Found {len(bin_files)} bin files for augmentation")
    
    for i, bin_file in enumerate(tqdm(bin_files, desc="Processing augmented dataset")):
        label_file = Path(label_dir) / f"{bin_file.stem}.txt"
        
        # Skip if label file doesn't exist
        if not label_file.exists():
            print(f"Warning: {label_file} not found, skipping {bin_file}")
            continue
        
        # Load point cloud and labels
        points = read_bin_file(bin_file)
        labels = read_label_file(label_file)
        
        # Apply augmentations
        augmented_data = apply_visualize_style_augmentations(points, labels, config)
        
        # Store all augmentations in memory
        augmented_items.extend(augmented_data)
        
        # Debug information for first few files
        if i < 5:
            print(f"Generated {len(augmented_data)} augmentations for item {i}: {bin_file.stem}")
    
    print(f"Generated {len(augmented_items)} augmented dataset items")
    return augmented_items

def prepare_yolo_dataset(dataset_items, train_val_split=0.8, memory_dataset=False):
    """
    Prepare dataset for YOLO training - either in memory or by saving to disk
    
    Args:
        dataset_items: List of tuples (bev_image, yolo_labels)
        train_val_split: Train/validation split ratio
        memory_dataset: Whether to keep dataset in memory or save to disk
        
    Returns:
        Dictionary with dataset information
    """
    print(f"Preparing dataset with {len(dataset_items)} items ({int(train_val_split*100)}% train, {int((1-train_val_split)*100)}% val)")
    
    # Shuffle dataset
    random.shuffle(dataset_items)
    
    # Split into training and validation
    split_idx = int(len(dataset_items) * train_val_split)
    train_items = dataset_items[:split_idx]
    val_items = dataset_items[split_idx:]
    
    train_indices = list(range(len(train_items)))
    val_indices = list(range(len(val_items)))
    
    # Extract images and labels
    train_images = [item[0] for item in train_items]
    train_labels = [item[1] for item in train_items]
    val_images = [item[0] for item in val_items]
    val_labels = [item[1] for item in val_items]
    
    print(f"Dataset prepared: {len(train_indices)} training items, {len(val_indices)} validation items")
    
    return {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels
    }

def create_memory_data_yaml(dataset_info, class_names=['Car', 'Pedestrian', 'Cyclist', 'Truck']):
    """
    Create data.yaml content for in-memory dataset
    
    Args:
        dataset_info: Dictionary with train and val data
        class_names: List of class names
        
    Returns:
        Data YAML dictionary for YOLO
    """
    # Create data.yaml content
    data_yaml = {
        'train': dataset_info['train_indices'],  # Just indices for in-memory dataset
        'val': dataset_info['val_indices'],      # Just indices for in-memory dataset
        'nc': len(class_names),
        'names': class_names
    }
    
    print(f"Created in-memory data structure with {len(data_yaml['train'])} train and {len(data_yaml['val'])} val indices")
    return data_yaml

def train_yolo_model(model, dataset_info, epochs, img_size, batch_size, output_dir, device):
    """
    Train YOLO model using the prepared dataset
    
    Args:
        model: YOLO model instance
        dataset_info: Dictionary with dataset information
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Training batch size
        output_dir: Output directory for trained model
        device: Training device (cpu/cuda)
        
    Returns:
        Path to trained model weights
    """
    # Extract BEV images and labels
    train_images = dataset_info['train_images']
    train_labels = dataset_info['train_labels']
    val_images = dataset_info['val_images']
    val_labels = dataset_info['val_labels']
    
    # Create temporary directories for train and val datasets
    train_dir = os.path.join(output_dir, 'temp_train')
    val_dir = os.path.join(output_dir, 'temp_val')
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    
    # Create directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Save training images and labels
    print(f"Preparing training dataset with {len(train_images)} images...")
    for i, (img, labels) in enumerate(zip(train_images, train_labels)):
        img_path = os.path.join(train_img_dir, f'train_{i}.png')
        label_path = os.path.join(train_label_dir, f'train_{i}.txt')
        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    # Save validation images and labels
    print(f"Preparing validation dataset with {len(val_images)} images...")
    for i, (img, labels) in enumerate(zip(val_images, val_labels)):
        img_path = os.path.join(val_img_dir, f'val_{i}.png')
        label_path = os.path.join(val_label_dir, f'val_{i}.txt')
        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    # Create custom YAML dataset configuration
    dataset_config = {
        'path': output_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': 4,  # Number of classes
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Truck']
    }
    
    # Create a temporary YAML file with dataset configuration
    os.makedirs(output_dir, exist_ok=True)
    dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Check CUDA availability and use appropriate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    # Prepare training arguments
    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': output_dir,
        'name': 'bev-detection',
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
    weights_dir = os.path.join(output_dir, 'bev-detection', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    
    # Also save a copy with a more descriptive name
    if args.transfer_learning:
        final_weights = os.path.join(output_dir, 'bev_transfer_final.pt')
    else:
        final_weights = os.path.join(output_dir, 'bev_final.pt')
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    
    # Clean up temporary directories
    print("Cleaning up temporary dataset directories...")
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    
    return best_weights

def load_and_continue_training(model_path, dataset_info, epochs, img_size, batch_size, output_dir, device):
    """
    Load existing model weights and continue training
    
    Args:
        model_path: Path to existing model weights
        dataset_info: Dictionary with dataset information
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Training batch size
        output_dir: Output directory for trained model
        device: Training device (cpu/cuda)
        
    Returns:
        Path to trained model weights
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"Loading existing model from {model_path}...")
    model = YOLO(model_path)
    
    # Extract BEV images and labels
    train_images = dataset_info['train_images']
    train_labels = dataset_info['train_labels']
    val_images = dataset_info['val_images']
    val_labels = dataset_info['val_labels']
    
    # Create temporary directories for train and val datasets
    train_dir = os.path.join(output_dir, 'temp_train')
    val_dir = os.path.join(output_dir, 'temp_val')
    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    
    # Create directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Save training images and labels
    print(f"Preparing training dataset with {len(train_images)} images...")
    for i, (img, labels) in enumerate(zip(train_images, train_labels)):
        img_path = os.path.join(train_img_dir, f'train_{i}.png')
        label_path = os.path.join(train_label_dir, f'train_{i}.txt')
        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    # Save validation images and labels
    print(f"Preparing validation dataset with {len(val_images)} images...")
    for i, (img, labels) in enumerate(zip(val_images, val_labels)):
        img_path = os.path.join(val_img_dir, f'val_{i}.png')
        label_path = os.path.join(val_label_dir, f'val_{i}.txt')
        cv2.imwrite(img_path, img)
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(' '.join(map(str, label)) + '\n')
    
    # Create custom YAML dataset configuration
    dataset_config = {
        'path': output_dir,
        'train': train_dir,
        'val': val_dir,
        'nc': 4,  # Number of classes
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Truck']
    }
    
    # Create a temporary YAML file with dataset configuration
    os.makedirs(output_dir, exist_ok=True)
    dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
    
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    
    # Check CUDA availability and use appropriate device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'
    
    # Prepare training arguments
    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': output_dir,
        'name': 'bev-continued',
        'exist_ok': True,
        'cache': True,
        'data': dataset_yaml,
        'patience': 50,  # Early stopping patience
        'save_period': 10  # Save checkpoint every 10 epochs
    }
    
    print(f"Using device: {device}")
    
    # Continue training the model
    print(f"Continuing training for {epochs} epochs with batch size {batch_size}")
    model.train(**train_args)
    
    # Return the path to best weights
    weights_dir = os.path.join(output_dir, 'bev-continued', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    
    # Also save a copy with a more descriptive name
    final_weights = os.path.join(output_dir, 'bev_continued_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    
    # Clean up temporary directories
    print("Cleaning up temporary dataset directories...")
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)
    
    return best_weights

def main():
    parser = argparse.ArgumentParser(description="BEV LiDAR Object Detection Training Pipeline")
    parser.add_argument("--bin_dir", default=DEFAULT_CONFIG['bin_dir'], help="Path to bin files")
    parser.add_argument("--label_dir", default=DEFAULT_CONFIG['label_dir'], help="Path to label txt files")
    parser.add_argument("--config_path", default=DEFAULT_CONFIG['config_path'], help="Path to config YAML")
    parser.add_argument("--output_base", default=DEFAULT_CONFIG['output_base'], help="Base output directory")
    parser.add_argument("--train_val_split", type=float, default=DEFAULT_CONFIG['train_val_split'], help="Train validation split ratio")
    parser.add_argument("--img_size", type=int, default=DEFAULT_CONFIG['img_size'], help="Image size for YOLOv5")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG['epochs'], help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG['batch_size'], help="Batch size for training")
    parser.add_argument("--augmentations", type=bool, default=DEFAULT_CONFIG['augmentations'], help="Use augmentations")
    parser.add_argument("--augmentation_factor", type=int, default=DEFAULT_CONFIG['augmentation_factor'], help="Augmentation factor")
    parser.add_argument("--device", default="auto", help="Training device (cpu/cuda:0/auto)")
    parser.add_argument("--transfer_learning", action="store_true", help="Use transfer learning")
    parser.add_argument("--unfreeze_layers", type=int, default=20, help="Number of layers to unfreeze for transfer learning")
    parser.add_argument("--continue_from", default=None, help="Path to existing model weights to continue training from")
    
    args = parser.parse_args()
    
    print("BEV LiDAR Object Detection Training Pipeline")
    print("===========================================")
    
    # Handle device selection - auto-detect if set to "auto"
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Auto-selected device: {args.device}")
    
    # Step 1: Generate standard dataset
    print("\nStep 1: Generating standard BEV dataset...")
    standard_items = generate_standard_dataset(args.bin_dir, args.label_dir, args.config_path, 
                                            args.output_base, args.img_size)
    
    # Step 2: Generate augmented dataset if enabled
    all_items = standard_items[:]
    if args.augmentations:
        print("\nStep 2: Generating augmented BEV dataset...")
        augmented_items = generate_augmented_dataset(args.bin_dir, args.label_dir, args.config_path, 
                                                args.output_base, args.img_size, args.augmentation_factor)
        all_items.extend(augmented_items)
    
    # Step 3: Prepare dataset (in memory, no saving to disk)
    print("\nStep 3: Preparing dataset for training...")
    dataset_info = prepare_yolo_dataset(all_items, args.train_val_split, memory_dataset=True)
    
    # Step 4 & 5: Create/load model and train
    output_dir = os.path.join(args.output_base, 'output')
    
    if args.continue_from:
        # Continue training from existing weights
        print(f"\nStep 4 & 5: Continuing training from {args.continue_from}...")
        model_path = load_and_continue_training(
            args.continue_from, 
            dataset_info, 
            args.epochs, 
            args.img_size, 
            args.batch,
            output_dir, 
            args.device
        )
    else:
        # Start new training
        print("\nStep 4: Setting up YOLO model...")
        if args.transfer_learning:
            print("Using transfer learning with last", args.unfreeze_layers, "layers unfrozen")
            model = create_transfer_learning_model(unfreeze_layers=args.unfreeze_layers)
        else:
            print("Creating fully trainable model")
            model = create_full_trainable_model()
        
        # Step 5: Train YOLO model
        print("\nStep 5: Training YOLO model...")
        model_path = train_yolo_model(model, dataset_info, args.epochs, args.img_size, 
                                    args.batch, output_dir, args.device)
    
    print("\nTraining pipeline completed successfully!")
    print(f"Model saved to: {model_path}")
    
    return model_path

if __name__ == "__main__":
    main()