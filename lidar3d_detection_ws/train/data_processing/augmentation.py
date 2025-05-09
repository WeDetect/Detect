import os
import sys
import numpy as np
from pathlib import Path
import math
import cv2

# הוסף את התיקייה הנוכחית לנתיב החיפוש
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from preprocessing import read_bin_file, read_label_file, create_bev_image, load_config

def rotate_points_and_labels(points, labels, angle_degrees):
    """
    Rotate point cloud and labels around Z axis
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        angle_degrees: Rotation angle in degrees (0-360)
    
    Returns:
        rotated_points: Rotated point cloud
        rotated_labels: Updated labels with new positions and orientations
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # Create rotation matrix around Z axis
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])
    
    # Calculate the centroid of the point cloud
    centroid = np.mean(points[:, :3], axis=0)
    
    # Translate points to origin
    translated_points = points[:, :3] - centroid
    
    # Apply rotation to XYZ coordinates (first 3 columns)
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    
    # Translate points back to original position
    rotated_points += centroid
    
    # Copy intensity values
    rotated_points = np.hstack((rotated_points, points[:, 3:4]))
    
    # Update labels
    rotated_labels = []
    if labels is None:
        return rotated_points, None
        
    for label in labels:
        new_label = label.copy()
        
        # Skip DontCare objects
        if new_label['type'] == 'DontCare':
            rotated_labels.append(new_label)
            continue
        
        # Rotate location
        x, y, z = new_label['location']
        x -= centroid[0]
        y -= centroid[1]
        new_x = x * cos_theta - y * sin_theta + centroid[0]
        new_y = x * sin_theta + y * cos_theta + centroid[1]
        new_label['location'] = [new_x, new_y, z]
        
        # Update rotation_y (add the rotation angle)
        new_label['rotation_y'] = (new_label['rotation_y'] + angle_rad) % (2 * np.pi)
        
        rotated_labels.append(new_label)
    
    return rotated_points, rotated_labels


def scale_distance_points_and_labels(points, labels, scale_factor):
    """
    Scale the distance of points and labels along X axis
    Positive scale_factor moves points away, negative brings them closer
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        scale_factor: Amount to scale (meters)
    
    Returns:
        scaled_points: Scaled point cloud
        scaled_labels: Updated labels with new positions
    """
    # Copy points to avoid modifying the original
    scaled_points = np.copy(points)
    
    # Apply scaling to X coordinate
    scaled_points[:, 0] += scale_factor
    
    # Update labels
    if labels is None:
        return scaled_points, None
        
    scaled_labels = []
    for label in labels:
        new_label = label.copy()
        
        # Skip DontCare objects
        if new_label['type'] == 'DontCare':
            scaled_labels.append(new_label)
            continue
        
        # Scale location
        x, y, z = new_label['location']
        new_label['location'] = [x + scale_factor, y, z]
        
        scaled_labels.append(new_label)
    
    return scaled_points, scaled_labels

def shift_lateral_points_and_labels(points, labels, shift_amount):
    """
    Shift points and labels laterally (left/right along Y axis)
    Positive shift moves right, negative moves left
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        shift_amount: Amount to shift laterally (meters)
    
    Returns:
        shifted_points: Shifted point cloud
        shifted_labels: Updated labels with new positions
    """
    # Copy points to avoid modifying the original
    shifted_points = np.copy(points)
    
    # Apply shift to Y coordinate
    shifted_points[:, 1] += shift_amount
    
    # Update labels
    if labels is None:
        return shifted_points, None
        
    shifted_labels = []
    for label in labels:
        new_label = label.copy()
        
        # Skip DontCare objects
        if new_label['type'] == 'DontCare':
            shifted_labels.append(new_label)
            continue
        
        # Shift location
        x, y, z = new_label['location']
        new_label['location'] = [x, y + shift_amount, z]
        
        shifted_labels.append(new_label)
    
    return shifted_points, shifted_labels

def shift_vertical_points_and_labels(points, labels, shift_amount):
    """
    Shift points and labels vertically (up/down along Z axis)
    Positive shift moves up, negative moves down
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        shift_amount: Amount to shift vertically (meters)
    
    Returns:
        shifted_points: Shifted point cloud
        shifted_labels: Updated labels with new positions
    """
    # Copy points to avoid modifying the original
    shifted_points = np.copy(points)
    
    # Apply shift to Z coordinate
    shifted_points[:, 2] += shift_amount
    
    # Update labels
    if labels is None:
        return shifted_points, None
        
    shifted_labels = []
    for label in labels:
        new_label = label.copy()
        
        # Skip DontCare objects
        if new_label['type'] == 'DontCare':
            shifted_labels.append(new_label)
            continue
        
        # Shift location
        x, y, z = new_label['location']
        new_label['location'] = [x, y, z + shift_amount]
        
        shifted_labels.append(new_label)
    
    return shifted_points, shifted_labels

def save_bin_file(points, output_path):
    """Save point cloud to binary file"""
    points.astype(np.float32).tofile(output_path)


def create_custom_bev_image(points, config, min_x, max_x, min_y, max_y, min_z, max_z, labels=None):
    """
    Create a bird's eye view image from point cloud with specified boundaries.
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        config: Configuration dictionary
        min_x, max_x: Minimum and maximum X boundaries
        min_y, max_y: Minimum and maximum Y boundaries
        min_z, max_z: Minimum and maximum Z boundaries
        labels: Optional list of label dictionaries
    
    Returns:
        bev_image: Generated BEV image
    """
    # Get dimensions from config
    Height = config['BEV_HEIGHT']
    Width = config['BEV_WIDTH']
    
    # Create a copy of points to avoid modifying the original
    points_copy = np.copy(points)
    
    # Filter points within specified boundaries
    mask = (points_copy[:, 0] >= min_x) & (points_copy[:, 0] <= max_x) & \
           (points_copy[:, 1] >= min_y) & (points_copy[:, 1] <= max_y) & \
           (points_copy[:, 2] >= min_z) & (points_copy[:, 2] <= max_z)
    points_copy = points_copy[mask]
    
    # Discretize Feature Map
    points_copy[:, 0] = np.int_(np.floor(points_copy[:, 0] / config['DISCRETIZATION']))
    points_copy[:, 1] = np.int_(np.floor(points_copy[:, 1] / config['DISCRETIZATION']) + Width / 2)
    
    # Ensure indices are within bounds
    points_copy[:, 0] = np.clip(points_copy[:, 0], 0, Height - 1)
    points_copy[:, 1] = np.clip(points_copy[:, 1], 0, Width - 1)
    
    # Initialize BEV image
    bev_image = np.zeros((Height, Width, 3), dtype=np.uint8)
    
    # Populate BEV image with intensity values
    for point in points_copy:
        x, y, z, intensity = point
        bev_image[int(x), int(y)] = (intensity * 255, intensity * 255, intensity * 255)
    
    # Optionally add labels to the BEV image
    if labels:
        for label in labels:
            if label['type'] == 'DontCare':
                continue
            
            # Get dimensions
            w = int(label['dimensions'][1] / config['DISCRETIZATION'])
            l = int(label['dimensions'][2] / config['DISCRETIZATION'])
            
            # Calculate center of the label
            center_x = int((label['location'][0] - min_x) / config['DISCRETIZATION'])
            center_y = int((label['location'][1] - min_y) / config['DISCRETIZATION'] + Width / 2)
            
            # Draw a box around the center point
            box_size = max(w, l, 10)  # Use at least 10 pixels
            half_size = box_size // 2
            
            # Draw the box
            color = (0, 255, 0)  # Green color for labels
            cv2.rectangle(bev_image, 
                         (center_x - half_size, center_y - half_size),
                         (center_x + half_size, center_y + half_size),
                         color, 2)
            
            # Add label text
            cv2.putText(bev_image, label['type'], 
                       (center_x - half_size, center_y - half_size - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return bev_image


def save_label_file(labels, output_path):
    """Save labels to KITTI format text file"""
    with open(output_path, 'w') as f:
        for label in labels:
            line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                label['type'],
                label['truncated'],
                label['occluded'],
                label['alpha'],
                *label['bbox'],
                *label['dimensions'],
                *label['location'],
                label['rotation_y']
            )
            f.write(line + '\n')

def apply_augmentation(bin_file, label_file, config_file, output_dir, 
                       rotation=0, distance_scale=0, lateral_shift=0, vertical_shift=0,
                       visualize=False):
    """
    Apply augmentation to a point cloud and its labels
    
    Args:
        bin_file: Path to input bin file
        label_file: Path to input label file
        config_file: Path to config file
        output_dir: Directory to save augmented files
        rotation: Rotation angle in degrees (0-360)
        distance_scale: Amount to scale distance (meters)
        lateral_shift: Amount to shift laterally (meters)
        vertical_shift: Amount to shift vertically (meters)
        visualize: Whether to visualize the result
    
    Returns:
        output_bin_path: Path to augmented bin file
        output_label_path: Path to augmented label file
    """
    # Load data
    points = read_bin_file(bin_file)
    labels = read_label_file(label_file)
    config = load_config(config_file)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    bin_filename = Path(bin_file).name
    label_filename = Path(label_file).name
    
    # Add augmentation info to filename
    augmentation_info = f"_r{rotation}_d{distance_scale}_l{lateral_shift}_v{vertical_shift}"
    output_bin_path = output_dir / f"{Path(bin_filename).stem}{augmentation_info}.bin"
    output_label_path = output_dir / f"{Path(label_filename).stem}{augmentation_info}.txt"
    
    # Apply augmentations in sequence
    augmented_points = points
    augmented_labels = labels
    
    if rotation != 0:
        augmented_points, augmented_labels = rotate_points_and_labels(
            augmented_points, augmented_labels, rotation)
    
    if distance_scale != 0:
        augmented_points, augmented_labels = scale_distance_points_and_labels(
            augmented_points, augmented_labels, distance_scale)
    
    if lateral_shift != 0:
        augmented_points, augmented_labels = shift_lateral_points_and_labels(
            augmented_points, augmented_labels, lateral_shift)
    
    if vertical_shift != 0:
        augmented_points, augmented_labels = shift_vertical_points_and_labels(
            augmented_points, augmented_labels, vertical_shift)
    
    # Save augmented data
    #save_bin_file(augmented_points, output_bin_path)
    #save_label_file(augmented_labels, output_label_path)
    
    # Visualize if requested
    if visualize:
        original_bev = create_bev_image(points, config, labels)
        augmented_bev = create_bev_image(augmented_points, config, augmented_labels)
        
        # Combine images side by side
        combined_img = np.hstack((original_bev, augmented_bev))
        
        # Add labels
        cv2.putText(combined_img, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_img, f"Augmented (r:{rotation}, d:{distance_scale}, l:{lateral_shift}, v:{vertical_shift})", 
                   (original_bev.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Augmentation Comparison", combined_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return str(output_bin_path), str(output_label_path)

