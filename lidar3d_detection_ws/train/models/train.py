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

def generate_augmented_dataset_with_visualization(bin_dir, label_dir, config_path, output_dir, img_size, augmentation_factor=3):
    """
    Generate augmented dataset from bin files and labels with visualization images
    
    Args:
        bin_dir: Directory with .bin files
        label_dir: Directory with label .txt files
        config_path: Path to preprocessing config YAML
        output_dir: Directory to save output data
        img_size: Image size for output
        augmentation_factor: Number of augmentations per original
        
    Returns:
        List of augmented (bev_image, yolo_labels) pairs
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create processor for BEV visualization
    processor = PointCloudProcessor(config_path=config_path)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'augmentation_visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get sorted list of bin files
    bin_files = sorted([os.path.join(bin_dir, f) for f in os.listdir(bin_dir) if f.endswith('.bin')])
    
    # Create augmented dataset items
    augmented_items = []
    
    # Process each file with progress bar
    for bin_file in tqdm(bin_files, desc='Processing augmented dataset'):
        try:
            # Get base name for labels
            base_name = os.path.splitext(os.path.basename(bin_file))[0]
            label_file = os.path.join(label_dir, f"{base_name}.txt")
            
            # Skip if label file doesn't exist
            if not os.path.exists(label_file):
                print(f"Warning: Label file {label_file} not found, skipping.")
                continue
            
            # Create sample visualization directory
            sample_vis_dir = os.path.join(vis_dir, base_name)
            os.makedirs(sample_vis_dir, exist_ok=True)
            
            # Load point cloud and labels
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            labels = processor.load_labels(label_file)  # Using the processor's method
            
            # Create original BEV image
            bev_image = processor.create_bev_image(points)
            
            # Create visualization with boxes for original
            bev_with_boxes = bev_image.copy()
            for label in labels:
                if label['type'] == 'DontCare':
                    continue
                
                # Transform 3D box to BEV coordinates
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    label['dimensions'], label['location'], label['rotation_y']
                )
                
                # Draw box on visualization image
                bev_with_boxes = processor.draw_box_on_bev(
                    bev_with_boxes, corners_bev, center_bev, label['type']
                )
            
            # Save original with boxes
            original_vis_path = os.path.join(sample_vis_dir, f"{base_name}_original.png")
            cv2.imwrite(original_vis_path, bev_with_boxes)
            
            # Convert labels to YOLO format for training
            img_height, img_width = bev_image.shape[:2]
            yolo_labels = convert_labels_to_yolo_format(labels, config, img_width, img_height)
            
            # Add original to dataset
            augmented_items.append((bev_image, yolo_labels))
            
            # Apply rotation augmentations
            rotations = [15, 30, 45, 60, 90, -15, -30, -45]
            for angle in random.sample(rotations, min(3, len(rotations))):
                try:
                    # Rotate points and labels
                    rotated_points, rotated_labels = rotate_points_and_labels(points, labels, angle)
                    
                    # Create BEV for rotated points
                    rotated_bev = processor.create_bev_image(rotated_points)
                    
                    # Create visualization with boxes for rotated points
                    rotated_bev_with_boxes = rotated_bev.copy()
                    for label in rotated_labels:
                        if label['type'] == 'DontCare':
                            continue
                        
                        # Transform 3D box to BEV coordinates
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            label['dimensions'], label['location'], label['rotation_y']
                        )
                        
                        # Draw box on visualization image
                        rotated_bev_with_boxes = processor.draw_box_on_bev(
                            rotated_bev_with_boxes, corners_bev, center_bev, label['type']
                        )
                    
                    # Save rotated visualization with boxes
                    rot_vis_path = os.path.join(sample_vis_dir, f"{base_name}_rot{angle}.png")
                    cv2.imwrite(rot_vis_path, rotated_bev_with_boxes)
                    
                    # Convert rotated labels to YOLO format for training
                    rot_height, rot_width = rotated_bev.shape[:2]
                    rotated_yolo_labels = convert_labels_to_yolo_format(
                        rotated_labels, config, rot_width, rot_height
                    )
                    
                    # Only add if we have valid labels
                    if len(rotated_yolo_labels) > 0:
                        # Add rotated to dataset
                        augmented_items.append((rotated_bev, rotated_yolo_labels))
                        
                except Exception as e:
                    print(f"Error processing rotation {angle} for {bin_file}: {str(e)}")
            
            # Apply lateral shift augmentations
            lateral_shifts = [-2, -1, 1, 2]  # meters
            for shift in random.sample(lateral_shifts, min(2, len(lateral_shifts))):
                try:
                    # Shift points and labels laterally
                    shifted_points, shifted_labels = shift_lateral_points_and_labels(points, labels, shift)
                    
                    # Create BEV for shifted points
                    shifted_bev = processor.create_bev_image(shifted_points)
                    
                    # Create visualization with boxes for shifted points
                    shifted_bev_with_boxes = shifted_bev.copy()
                    for label in shifted_labels:
                        if label['type'] == 'DontCare':
                            continue
                        
                        # Transform 3D box to BEV coordinates
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            label['dimensions'], label['location'], label['rotation_y']
                        )
                        
                        # Draw box on visualization image
                        shifted_bev_with_boxes = processor.draw_box_on_bev(
                            shifted_bev_with_boxes, corners_bev, center_bev, label['type']
                        )
                    
                    # Save shifted visualization with boxes
                    shift_vis_path = os.path.join(sample_vis_dir, f"{base_name}_lateral{shift}.png")
                    cv2.imwrite(shift_vis_path, shifted_bev_with_boxes)
                    
                    # Convert shifted labels to YOLO format for training
                    shift_height, shift_width = shifted_bev.shape[:2]
                    shifted_yolo_labels = convert_labels_to_yolo_format(
                        shifted_labels, config, shift_width, shift_height
                    )
                    
                    # Only add if we have valid labels
                    if len(shifted_yolo_labels) > 0:
                        # Add shifted to dataset
                        augmented_items.append((shifted_bev, shifted_yolo_labels))
                        
                except Exception as e:
                    print(f"Error processing lateral shift {shift} for {bin_file}: {str(e)}")
            
            # Apply forward/backward shifts
            vertical_shifts = [-4, -2, 2, 4]  # meters
            for shift in random.sample(vertical_shifts, min(2, len(vertical_shifts))):
                try:
                    # Shift points and labels vertically
                    shifted_points, shifted_labels = shift_vertical_points_and_labels(points, labels, shift)
                    
                    # Create BEV for shifted points
                    shifted_bev = processor.create_bev_image(shifted_points)
                    
                    # Create visualization with boxes for shifted points
                    shifted_bev_with_boxes = shifted_bev.copy()
                    for label in shifted_labels:
                        if label['type'] == 'DontCare':
                            continue
                        
                        # Transform 3D box to BEV coordinates
                        corners_bev, center_bev = processor.transform_3d_box_to_bev(
                            label['dimensions'], label['location'], label['rotation_y']
                        )
                        
                        # Draw box on visualization image
                        shifted_bev_with_boxes = processor.draw_box_on_bev(
                            shifted_bev_with_boxes, corners_bev, center_bev, label['type']
                        )
                    
                    # Save shifted visualization with boxes
                    shift_vis_path = os.path.join(sample_vis_dir, f"{base_name}_vertical{shift}.png")
                    cv2.imwrite(shift_vis_path, shifted_bev_with_boxes)
                    
                    # Convert shifted labels to YOLO format for training
                    shift_height, shift_width = shifted_bev.shape[:2]
                    shifted_yolo_labels = convert_labels_to_yolo_format(
                        shifted_labels, config, shift_width, shift_height
                    )
                    
                    # Only add if we have valid labels
                    if len(shifted_yolo_labels) > 0:
                        # Add shifted to dataset
                        augmented_items.append((shifted_bev, shifted_yolo_labels))
                        
                except Exception as e:
                    print(f"Error processing vertical shift {shift} for {bin_file}: {str(e)}")
            
            # Apply range view adaptations (zoom in on regions)
            for i in range(2):  # Try 2 random range adaptations
                try:
                    # Get random range based on point distribution
                    x_points = points[:, 0]
                    y_points = points[:, 1]
                    
                    x_range = [max(0, np.percentile(x_points, 10)), min(50, np.percentile(x_points, 90))]
                    y_range = [max(-25, np.percentile(y_points, 10)), min(25, np.percentile(y_points, 90))]
                    
                    # Filter points and labels by range
                    filtered_points, filtered_labels = filter_points_by_range(
                        points, labels, x_range, y_range
                    )
                    
                    # Check if we have enough points and labels
                    if len(filtered_points) < 100 or len(filtered_labels) < 1:
                        continue
                    
                    # Create custom config for this range
                    range_config = config.copy()
                    range_config['x_range'] = x_range
                    range_config['y_range'] = y_range
                    
                    # Create BEV for filtered points using range-adapted function
                    range_bev = create_range_adapted_bev_image(filtered_points, range_config)
                    
                    # Create visualization with boxes for range-adapted BEV
                    range_bev_with_boxes = range_bev.copy()
                    for label in filtered_labels:
                        if label['type'] == 'DontCare':
                            continue
                        
                        # For range-adapted BEV we need to adjust coordinates
                        # based on the new coordinate system
                        x_min, x_max = range_config['x_range']
                        y_min, y_max = range_config['y_range']
                        width = int((x_max - x_min) / range_config['resolution'])
                        height = int((y_max - y_min) / range_config['resolution'])
                        
                        # Create temp processor with modified config for correct conversion
                        temp_config = range_config.copy()
                        temp_config['fwd_range'] = range_config['x_range']
                        temp_config['side_range'] = range_config['y_range']
                        
                        # This is a bit tricky as we need to adapt the model
                        # Fix: Use coordinates relative to the range window
                        # Here I'll do manual calculation for the BEV projection
                        
                        # Get the dimensions and location from the label
                        dims, loc, rot = label['dimensions'], label['location'], label['rotation_y']
                        
                        # Adjust location relative to range window
                        rel_x = loc[0] - x_min
                        rel_y = loc[1] - y_min
                        
                        # Map to image coordinates
                        img_x = int(rel_x / range_config['resolution'])
                        img_y = int(height - (rel_y / range_config['resolution']))
                        
                        # Get length and width from dimensions
                        l, h, w = dims
                        
                        # Calculate corners based on rotation
                        corners = []
                        for dx, dy in [(l/2, w/2), (l/2, -w/2), (-l/2, -w/2), (-l/2, w/2)]:
                            # Rotate
                            rx = dx * np.cos(rot) - dy * np.sin(rot)
                            ry = dx * np.sin(rot) + dy * np.cos(rot)
                            
                            # Convert to image coordinates
                            cx = int((rel_x + rx) / range_config['resolution'])
                            cy = int(height - (rel_y + ry) / range_config['resolution'])
                            
                            corners.append((cx, cy))
                        
                        # Draw box
                        for i in range(4):
                            cv2.line(range_bev_with_boxes, corners[i], corners[(i+1)%4], (0, 0, 255), 2)
                        
                        # Draw center
                        center = (img_x, img_y)
                        cv2.circle(range_bev_with_boxes, center, 3, (0, 0, 255), -1)
                        
                        # Add label
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(range_bev_with_boxes, label['type'], 
                                  (center[0], center[1] - 10), 
                                  font, 0.5, (0, 0, 255), 2)
                    
                    # Save range visualization with boxes
                    range_vis_path = os.path.join(sample_vis_dir, f"{base_name}_range{i}.png")
                    cv2.imwrite(range_vis_path, range_bev_with_boxes)
                    
                    # Convert range-adapted labels to YOLO format for training
                    range_yolo_labels = convert_labels_to_yolo_format(
                        filtered_labels, temp_config, width, height
                    )
                    
                    # Only add if we have valid labels
                    if len(range_yolo_labels) > 0:
                        # Add range-adapted to dataset
                        augmented_items.append((range_bev, range_yolo_labels))
                        
                except Exception as e:
                    import traceback
                    print(f"Error processing range {i} for {bin_file}: {str(e)}")
                    print(traceback.format_exc())  # Print detailed stack trace
        
        except Exception as e:
            import traceback
            print(f"Error processing {bin_file} for augmentation: {str(e)}")
            print(traceback.format_exc())  # Print detailed stack trace
    
    print(f"Generated {len(augmented_items)} augmented dataset items in total")
    print(f"Saved visualization images to {vis_dir}")
    
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
    parser.add_argument('--train_val_split', type=float, default=0.8, 
                        help='Train/validation split ratio')
    
    # Augmentation options
    parser.add_argument('--augmentations', action='store_true', 
                        help='Enable data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=2, 
                        help='Augmentation factor (multiplier for dataset size)')
    
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
    
    # Single image test options
    parser.add_argument('--single_image_test', action='store_true', 
                        help='Train on a single image for testing')
    parser.add_argument('--bin_file', type=str, default='', 
                        help='Path to specific bin file for single image test')
    parser.add_argument('--label_file', type=str, default='', 
                        help='Path to specific label file for single image test')
    parser.add_argument('--model_path', type=str, default='', 
                        help='Path to model weights for continuing training')
    parser.add_argument('--train_from_scratch', action='store_true',
                        help='Train from scratch on a single image')
    
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
            # המשך אימון על תמונה בודדת
            model_path = train_on_single_image_continue(
                bin_file=bin_file,
                label_file=label_file,
                config_path=args.config_path,
                output_dir=os.path.join(args.output_base, "output"),
                model_path=args.model_path or "best.pt",
                epochs=args.epochs,
                img_size=args.img_size,
                batch_size=args.batch,
                device=args.device
            )
            print(f"Single image training completed. Model saved to: {model_path}")
        
        return model_path
    
    # המשך הקוד הרגיל לאימון על מערך נתונים מלא
    # Step 1: Generate standard dataset
    print("\nStep 1: Generating standard BEV dataset...")
    standard_items = generate_standard_dataset(args.bin_dir, args.label_dir, args.config_path, 
                                            args.output_base, args.img_size)
    
    # Step 2: Generate augmented dataset if enabled
    all_items = standard_items[:]
    if args.augmentations:
        print("\nStep 2: Generating augmented BEV dataset...")
        augmented_items = generate_augmented_dataset_with_visualization(args.bin_dir, args.label_dir, args.config_path, 
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