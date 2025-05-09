import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from pathlib import Path
import json

from .preprocessing import read_bin_file, read_label_file, create_bev_image, load_config
from .augmentation import (
    rotate_points_and_labels, 
    scale_distance_points_and_labels, 
    shift_lateral_points_and_labels, 
    shift_vertical_points_and_labels
)

class BEVDataset(Dataset):
    def __init__(self, bin_files, label_files, config_path, classes_json_path, img_size=608, augment=True, max_augmentations=20):
        """
        Dataset for BEV images from LiDAR point clouds
        
        Args:
            bin_files: List of paths to bin files
            label_files: List of paths to label files
            config_path: Path to config file
            classes_json_path: Path to classes JSON file
            img_size: Size of BEV image
            augment: Whether to use augmentation
            max_augmentations: Maximum number of augmentations per sample
        """
        self.bin_files = bin_files
        self.label_files = label_files
        self.config = load_config(config_path)
        self.img_size = img_size
        self.augment = augment
        self.max_augmentations = max_augmentations
        
        # Load class names
        with open(classes_json_path, 'r') as f:
            classes_data = json.load(f)
            self.class_names = [c['name'] for c in classes_data['classes']]
            self.class_dict = {c['name']: c['id'] for c in classes_data['classes']}
            print(f"Loaded class dictionary: {self.class_dict}")
        
        # Create augmented dataset if augmentation is enabled
        if augment and len(bin_files) > 0:
            self.augmented_samples = self._create_augmented_samples()
        else:
            self.augmented_samples = list(zip(bin_files, label_files))
    
    def _create_augmented_samples(self):
        """Create augmented samples from original data"""
        augmented_samples = []
        
        # Add original samples
        for bin_file, label_file in zip(self.bin_files, self.label_files):
            augmented_samples.append((bin_file, label_file))
        
        # Add augmented samples
        for bin_file, label_file in zip(self.bin_files, self.label_files):
            for _ in range(self.max_augmentations):
                # Random augmentation parameters
                rotation = random.uniform(-30, 30)
                distance_scale = random.uniform(-5, 5)
                lateral_shift = random.uniform(-5, 5)
                vertical_shift = random.uniform(-1, 1)
                
                # Add augmented sample (we'll apply augmentation at load time)
                augmented_samples.append((
                    bin_file, label_file, 
                    rotation, distance_scale, lateral_shift, vertical_shift
                ))
        
        return augmented_samples
    
    def __len__(self):
        return len(self.augmented_samples)
    
    def __getitem__(self, idx):
        sample = self.augmented_samples[idx]
        
        # Check if this is an augmented sample
        if len(sample) > 2:
            bin_file, label_file, rotation, distance_scale, lateral_shift, vertical_shift = sample
            is_augmented = True
        else:
            bin_file, label_file = sample
            is_augmented = False
        
        # Load point cloud and labels
        points = read_bin_file(bin_file)
        labels = read_label_file(label_file)
        
        # Apply augmentations if needed
        if is_augmented:
            points, labels = rotate_points_and_labels(points, labels, rotation)
            points, labels = scale_distance_points_and_labels(points, labels, distance_scale)
            points, labels = shift_lateral_points_and_labels(points, labels, lateral_shift)
            points, labels = shift_vertical_points_and_labels(points, labels, vertical_shift)
        
        # Create BEV image
        bev_result = create_bev_image(points, self.config, labels)
        
        # Check if result is a tuple (image, white_dots)
        if isinstance(bev_result, tuple):
            bev_image = bev_result[0]  # Extract the image from the tuple
        else:
            bev_image = bev_result
        
        # Resize if needed
        if bev_image.shape[0] != self.img_size or bev_image.shape[1] != self.img_size:
            bev_image = cv2.resize(bev_image, (self.img_size, self.img_size))
        
        # Convert to tensor
        img = torch.from_numpy(bev_image.transpose(2, 0, 1)).float() / 255.0
        
        # Process labels for YOLO format
        yolo_labels = []
        
        for label in labels:
            if label['type'] == 'DontCare':
                continue
            
            # Get class index from class dictionary
            class_idx = self.class_dict.get(label['type'], -1)
            if class_idx == -1:
                continue  # Skip unknown classes
            
            # Get location in BEV coordinates
            x = label['location'][0]  # Forward direction
            y = label['location'][1]  # Left-right direction
            
            # Convert to BEV pixel coordinates
            x_bev = int(x / self.config['DISCRETIZATION'])
            y_bev = int(y / self.config['DISCRETIZATION'] + self.config['BEV_WIDTH'] / 2)
            
            # Get dimensions in BEV pixels
            width = max(1, int(label['dimensions'][1] / self.config['DISCRETIZATION']))  # width = y dimension
            length = max(1, int(label['dimensions'][0] / self.config['DISCRETIZATION']))  # length = x dimension
            
            # Calculate box size (use the larger of width/length for square box)
            box_size = max(width, length)
            half_size = box_size // 2
            
            # Calculate normalized coordinates (YOLO format)
            x_center = x_bev / self.img_size
            y_center = y_bev / self.img_size
            width_norm = box_size / self.img_size
            height_norm = box_size / self.img_size
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0.001, min(0.999, width_norm))
            height_norm = max(0.001, min(0.999, height_norm))
            
            # YOLO format: [class_idx, x_center, y_center, width, height]
            yolo_labels.append([class_idx, x_center, y_center, width_norm, height_norm])
        
        # Convert labels to tensor
        if yolo_labels:
            targets = torch.FloatTensor(yolo_labels)
        else:
            # Empty tensor if no labels
            targets = torch.zeros((0, 5))
        
        # Debug info (only for first few items)
        if idx < 3:
            print(f"\nDataset item {idx}:")
            print(f"Image shape: {img.shape}, min={img.min()}, max={img.max()}")
            print(f"Number of labels: {len(yolo_labels)}")
            for i, label in enumerate(yolo_labels):
                print(f"Label {i}: class={int(label[0])}, center=({label[1]:.4f}, {label[2]:.4f}), size=({label[3]:.4f}, {label[4]:.4f})")
        
        return img, targets, bin_file

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        imgs, targets, paths = list(zip(*batch))
        
        # Remove empty targets
        targets = [boxes for boxes in targets if boxes is not None]
        
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        
        # Stack images
        imgs = torch.stack([img for img in imgs])
        
        # Concatenate targets
        if len(targets) > 0:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.zeros((0, 6))
        
        return imgs, targets, paths