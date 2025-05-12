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

def filter_points_by_range(points, labels, x_min, x_max, y_min, y_max):
    """
    Filter points to keep only those within the specified X and Y range
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        x_min, x_max: X-axis range boundaries (forward direction)
        y_min, y_max: Y-axis range boundaries (lateral direction)
    
    Returns:
        filtered_points: Filtered point cloud
        filtered_labels: Updated labels that fall within the range
    """
    # Copy points to avoid modifying the original
    filtered_points = np.copy(points)
    
    # Create a mask for points that fall within the specified range
    mask = (filtered_points[:, 0] >= x_min) & (filtered_points[:, 0] <= x_max) & \
           (filtered_points[:, 1] >= y_min) & (filtered_points[:, 1] <= y_max)
    
    # Apply the mask to filter points
    filtered_points = filtered_points[mask]
    
    # Update labels
    if labels is None:
        return filtered_points, None
        
    filtered_labels = []
    for label in labels:
        new_label = label.copy()
        
        # Skip DontCare objects
        if new_label['type'] == 'DontCare':
            filtered_labels.append(new_label)
            continue
        
        # Get label location
        x, y, z = new_label['location']
        
        # Check if the center of the label is within the range
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered_labels.append(new_label)
    
    return filtered_points, filtered_labels

def create_range_adapted_bev_image(points, labels, x_min, x_max, y_min, y_max, config):
    """
    Create a BEV image for a specific range that fills the entire image space
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        x_min, x_max: Forward range boundaries (meters)
        y_min, y_max: Lateral range boundaries (meters)
        config: Configuration dictionary
    
    Returns:
        adapted_bev: BEV image that maximizes the use of available space
        yolo_labels: YOLO format labels
    """
    # Get the output dimensions
    Height = config['BEV_HEIGHT']
    Width = config['BEV_WIDTH']
    
    # Get z-range from config
    z_min = config['HEIGHT_RANGE'][0] if 'HEIGHT_RANGE' in config else -1.0
    z_max = config['HEIGHT_RANGE'][1] if 'HEIGHT_RANGE' in config else 5.0
    z_resolution = config['Z_RESOLUTION'] if 'Z_RESOLUTION' in config else 0.2
    
    # Calculate range dimensions
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate resolution to fill the entire image
    resolution_x = x_range / Height  # Note: Height for x because we're using the KITTI convention
    resolution_y = y_range / Width   # Width for y because of the coordinate conversion
    resolution = max(resolution_x, resolution_y)  # Use the max to maintain aspect ratio
    
    # Create a custom processor class for this specific range
    class RangeSpecificProcessor:
        def __init__(self):
            # Fixed dimensions based on the range
            self.x_max = Height - 1
            self.y_max = Width - 1
            self.z_max = int((z_max - z_min) / z_resolution)
            
            # Store range parameters
            self.fwd_range = [x_min, x_max]
            self.side_range = [y_min, y_max]
            self.height_range = [z_min, z_max]
            self.resolution = resolution
            self.z_resolution = z_resolution
            
            # Color mapping for different object types (same as in preprocessing_0.py)
            self.color_map = {
                'Car': (0, 0, 255),        # Red
                'Pedestrian': (0, 255, 0), # Green
                'Cyclist': (255, 0, 0),    # Blue
                'Van': (255, 0, 255),      # Magenta
                'Truck': (255, 255, 0),    # Cyan
                'Person': (0, 255, 255),   # Yellow
                'Tram': (128, 128, 255),   # Pink
                'Misc': (128, 128, 128)    # Gray
            }
        
        def create_bev_image(self, points):
            """Create a BEV image from points (same logic as in preprocessing_0.py)"""
            # Initialize BEV image array
            bev_image = np.zeros((self.x_max + 1, self.y_max + 1, self.z_max + 2), dtype=np.float32)
            
            # Extract point coordinates
            x_points = points[:, 0]
            y_points = points[:, 1]
            z_points = points[:, 2]
            intensity = points[:, 3]
            
            # Filter points that are within the specified ranges
            f_filter = np.logical_and((x_points > self.fwd_range[0]), (x_points < self.fwd_range[1]))
            s_filter = np.logical_and((y_points > self.side_range[0]), (y_points < self.side_range[1]))
            filt = np.logical_and(f_filter, s_filter)
            z_filter = np.logical_and((z_points > self.height_range[0]), (z_points < self.height_range[1]))
            filt = np.logical_and(filt, z_filter)
            
            # Extract filtered points
            indices = np.where(filt)[0]
            x_filt = x_points[indices]
            y_filt = y_points[indices]
            z_filt = z_points[indices]
            intensity_filt = intensity[indices]
            
            # Convert coordinates to pixel positions
            x_img = (-y_filt / self.resolution).astype(np.int32)
            y_img = (-x_filt / self.resolution).astype(np.int32)
            
            # Shift coordinates to image space
            x_img -= int(np.floor(self.side_range[0] / self.resolution))
            y_img += int(np.floor(self.fwd_range[1] / self.resolution))
            
            # Height slices
            for i, height in enumerate(np.arange(self.height_range[0], self.height_range[1], self.z_resolution)):
                z_slice = np.logical_and((z_filt >= height), (z_filt < height + self.z_resolution))
                if np.any(z_slice):
                    z_indices = np.where(z_slice)[0]
                    # Make sure indices are within bounds
                    valid_indices = (x_img[z_indices] >= 0) & (x_img[z_indices] <= self.y_max) & \
                                   (y_img[z_indices] >= 0) & (y_img[z_indices] <= self.x_max)
                    valid_z_indices = z_indices[valid_indices]
                    if len(valid_z_indices) > 0:
                        bev_image[y_img[valid_z_indices], x_img[valid_z_indices], i] = 1
            
            # Add intensity channel (with bounds checking)
            valid_indices = (x_img >= 0) & (x_img <= self.y_max) & (y_img >= 0) & (y_img <= self.x_max)
            valid_idx = np.where(valid_indices)[0]
            if len(valid_idx) > 0:
                bev_image[y_img[valid_idx], x_img[valid_idx], -1] = intensity_filt[valid_idx] / 255.0
            
            # Create a colored height map for better visualization
            height_map = np.zeros((self.x_max + 1, self.y_max + 1, 3), dtype=np.uint8)
            
            # Assign different colors based on height
            for i in range(self.z_max):
                # Create a color gradient from blue (lower) to red (higher)
                b = max(0, 255 - (i * 255 // self.z_max))
                r = min(255, i * 255 // self.z_max)
                g = min(100, i * 100 // self.z_max)
                
                mask = bev_image[:, :, i] > 0
                height_map[mask, 0] = r
                height_map[mask, 1] = g
                height_map[mask, 2] = b
            
            # Use intensity for empty cells
            empty = np.sum(bev_image[:, :, :-1], axis=2) == 0
            intensity_map = (bev_image[:, :, -1] * 255).astype(np.uint8)
            height_map[empty, 0] = intensity_map[empty]
            height_map[empty, 1] = intensity_map[empty]
            height_map[empty, 2] = intensity_map[empty]
            
            return height_map
        
        def transform_3d_box_to_bev(self, dimensions, location, rotation_y):
            """Transform 3D box to BEV coordinates (same as in preprocessing_0.py)"""
            # Extract dimensions
            h, w, l = dimensions
            x, y, z = location
            
            # Calculate 3D box corners (exactly as in the original)
            corners_3d = np.array([
                [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],
                [0, 0, 0, 0, -h, -h, -h, -h],
                [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            ])
            
            # Rotation matrix
            R = np.array([
                [np.cos(rotation_y), 0, np.sin(rotation_y)],
                [0, 1, 0],
                [-np.sin(rotation_y), 0, np.cos(rotation_y)]
            ])
            
            # Apply rotation and translation
            corners_3d = np.dot(R, corners_3d)
            corners_3d[0, :] += x
            corners_3d[1, :] += y
            corners_3d[2, :] += z
            
            # Get only the base of the box for BEV
            base_corners_3d = corners_3d[:, :4]
            
            # Transform to BEV image coordinates
            x_img = (-base_corners_3d[1, :] / self.resolution).astype(np.int32)
            y_img = (-base_corners_3d[0, :] / self.resolution).astype(np.int32)
            
            # Shift to image coordinates for the specific range
            x_img -= int(np.floor(self.side_range[0] / self.resolution))
            y_img += int(np.floor(self.fwd_range[1] / self.resolution))
            
            # Center point in BEV
            center_x = (-y / self.resolution) - int(np.floor(self.side_range[0] / self.resolution))
            center_y = (-x / self.resolution) + int(np.floor(self.fwd_range[1] / self.resolution))
            
            corners_bev = list(zip(x_img, y_img))
            center_bev = (int(center_x), int(center_y))
            
            return corners_bev, center_bev
        
        def draw_box_on_bev(self, bev_image, corners_bev, center_bev, obj_type):
            """Draw bounding box on BEV image (same as in preprocessing_0.py)"""
            # Create a copy of the image to draw on
            bev_with_box = bev_image.copy()
            
            # Get color for this object type
            color = self.color_map.get(obj_type, (255, 255, 255))  # Default: white
            
            # Connect corners with lines
            for i in range(4):
                cv2.line(bev_with_box, 
                          corners_bev[i], 
                          corners_bev[(i+1)%4], 
                          color, 
                          2)
            
            # Draw center point
            cv2.circle(bev_with_box, center_bev, 3, color, -1)
            
            # Add label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(bev_with_box, obj_type, 
                      (center_bev[0], center_bev[1] - 10), 
                      font, 0.5, color, 2)
            
            return bev_with_box
        
        def create_yolo_label(self, corners_bev, obj_type, img_shape=(608, 608)):
            """Create YOLO format label based on BEV corners (enhanced from preprocessing_0.py)"""
            # Get image dimensions
            img_height, img_width = img_shape
            
            # Convert corners to numpy array
            corners = np.array(corners_bev)
            
            # Calculate bounding box
            x_min = np.min(corners[:, 0])
            y_min = np.min(corners[:, 1])
            x_max = np.max(corners[:, 0])
            y_max = np.max(corners[:, 1])
            
            # Calculate center and dimensions (YOLO format)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            
            # Normalize coordinates (YOLO format)
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            # Map class to ID
            class_id = 0  # Default to Car
            if obj_type == 'Pedestrian':
                class_id = 1
            elif obj_type == 'Cyclist':
                class_id = 2
            elif obj_type == 'Truck':
                class_id = 3
            
            # Create YOLO label string
            yolo_label = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            
            return yolo_label
    
    # Create processor instance for this specific range
    processor = RangeSpecificProcessor()
    
    # Create BEV image from points
    bev_image = processor.create_bev_image(points)
    
    # Process labels
    yolo_labels = []
    
    if labels:
        bev_with_boxes = bev_image.copy()
        
        for label in labels:
            if label['type'] == 'DontCare':
                continue
                
            # Get label location
            x, y, z = label['location']
            
            # Skip if not in range
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue
            
            # Transform 3D box to BEV
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                label['dimensions'],
                label['location'],
                label['rotation_y']
            )
            
            # Make sure box is visible in the image
            if all((0 <= x < Width and 0 <= y < Height) for x, y in corners_bev):
                # Draw box on the image
                bev_with_boxes = processor.draw_box_on_bev(
                    bev_with_boxes,
                    corners_bev,
                    center_bev,
                    label['type']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev,
                    label['type'],
                    (Height, Width)
                )
                
                yolo_labels.append(yolo_label)
        
        return bev_with_boxes, yolo_labels
    
    return bev_image, yolo_labels