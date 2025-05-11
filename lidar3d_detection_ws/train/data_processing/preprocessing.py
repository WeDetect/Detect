import numpy as np
import cv2
import yaml
from pathlib import Path
import struct
import sys
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


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
    """Create bird's eye view image from point cloud without label boxes"""
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
    
    # Prepare white dots for label points (but don't draw them)
    white_dots_mask = np.zeros((Height, Width), dtype=bool)
    
    # Process labels and mark their positions (for tracking, not for display)
    if labels is not None:
        for label in labels:
            try:
                # Skip DontCare objects
                if label['type'] == 'DontCare':
                    continue
                
                # Get location (x, y, z in KITTI format)
                x = label['location'][0]  # x
                y = label['location'][1]  # y
                
                # Convert to BEV pixel coordinates before rotation
                x_bev = int(x / config['DISCRETIZATION'])
                y_bev = int(y / config['DISCRETIZATION'] + Width / 2)
                
                # Ensure within bounds
                if 0 <= x_bev < Height and 0 <= y_bev < Width:
                    # Mark a 3x3 area around the point
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = x_bev + dx, y_bev + dy
                            if 0 <= nx < Height and 0 <= ny < Width:
                                white_dots_mask[nx, ny] = True
            except Exception as e:
                print(f"Error processing label point: {e}")
                continue
    
    # Rotate the map by 180 degrees for proper orientation
    RGB_Map = np.rot90(RGB_Map, 2, axes=(1, 2))
    rotated_white_dots_mask = np.rot90(white_dots_mask, 2)
    
    # Convert to image format (H, W, C) and scale to 0-255
    bev_image = (RGB_Map.transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Find the non-zero region of the image
    non_zero_mask = np.any(bev_image > 0, axis=2)
    crop_info = None
    
    if np.any(non_zero_mask):
        rows = np.any(non_zero_mask, axis=1)
        cols = np.any(non_zero_mask, axis=0)
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if len(row_indices) > 0 and len(col_indices) > 0:
            y_min, y_max = row_indices[[0, -1]]
            x_min, x_max = col_indices[[0, -1]]
            
            # Add some padding
            padding = 10
            y_min = max(0, y_min - padding)
            y_max = min(Height - 1, y_max + padding)
            x_min = max(0, x_min - padding)
            x_max = min(Width - 1, x_max + padding)
            
            # Save crop information for later use
            crop_info = (y_min, y_max, x_min, x_max)
            
            # Crop to the non-zero region
            cropped_image = bev_image[y_min:y_max+1, x_min:x_max+1]
            cropped_white_dots = rotated_white_dots_mask[y_min:y_max+1, x_min:x_max+1]
            
            # Resize to the original dimensions
            bev_image = cv2.resize(cropped_image, (Width, Height), interpolation=cv2.INTER_LINEAR)
            
            # Also resize the white dots mask
            resized_white_dots = cv2.resize(cropped_white_dots.astype(np.uint8), (Width, Height), interpolation=cv2.INTER_NEAREST)
            white_dots_positions = np.where(resized_white_dots > 0)
    
    return bev_image, white_dots_positions


def calculate_anchors_kmeans(label_files, config, num_clusters=9):
    """
    Calculate anchors using K-means clustering on the dataset
    
    Args:
        label_files: List of paths to label files
        config: Configuration dictionary
        num_clusters: Number of clusters (anchors)
        
    Returns:
        List of anchors in (width, height) format
    """
    print("Calculating anchors using K-means...")
    
    # Collect all bounding box dimensions
    widths = []
    heights = []
    
    for label_file in label_files:
        labels = read_label_file(label_file)
        
        for label in labels:
            if label['type'] == 'DontCare':
                continue
            
            # Get dimensions in BEV pixels
            width = max(1, int(label['dimensions'][1] / config['DISCRETIZATION']))  # width = y dimension
            length = max(1, int(label['dimensions'][0] / config['DISCRETIZATION']))  # length = x dimension
            
            # Use the larger dimension for both width and height (square boxes)
            box_size = max(width, length)
            
            widths.append(box_size)
            heights.append(box_size)
    
    # Convert to numpy arrays
    widths = np.array(widths)
    heights = np.array(heights)
    
    # Combine width and height
    boxes = np.stack([widths, heights], axis=1)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    
    # Sort anchors by area (smallest to largest)
    areas = anchors[:, 0] * anchors[:, 1]
    indices = np.argsort(areas)
    anchors = anchors[indices]
    
    # Round to integers
    anchors = np.round(anchors).astype(int)
    
    # Ensure minimum size
    anchors = np.maximum(anchors, 1)
    
    # Split into 3 groups for different scales
    small_anchors = anchors[:3].tolist()
    medium_anchors = anchors[3:6].tolist()
    large_anchors = anchors[6:].tolist()
    
    print(f"Small anchors: {small_anchors}")
    print(f"Medium anchors: {medium_anchors}")
    print(f"Large anchors: {large_anchors}")
    
    return [small_anchors, medium_anchors, large_anchors]

