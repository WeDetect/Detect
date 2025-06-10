import numpy as np
import math
import cv2
import copy




def rotate_points_and_labels(points, labels, angle_degrees):
    """
    Efficiently rotate point cloud and KITTI labels (3D boxes) around Z-axis.

    Args:
        points (np.ndarray): Nx4 array of points (x, y, z, intensity)
        labels (list of dict): KITTI-style label dicts
        angle_degrees (float): Rotation angle in degrees

    Returns:
        rotated_points: Nx4 array
        rotated_labels: list of dicts
    """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Rotation matrix around Z
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=np.float32)

    # Rotate points using matrix multiplication (fast!)
    rotated_points = points.copy()
    rotated_points[:, :3] = points[:, :3] @ R.T

    # Efficient label processing: extract all locations to a matrix
    if labels:
        locs = np.array([label['location'] for label in labels], dtype=np.float32)  # (N, 3)
        rotated_locs = locs @ R.T  # Rotate all locations at once

        # Adjust yaws
        yaws = np.array([label['rotation_y'] for label in labels], dtype=np.float32)
        new_yaws = (yaws + angle_rad + np.pi) % (2 * np.pi) - np.pi

        # Build new labels
        rotated_labels = []
        for i, label in enumerate(labels):
            new_label = label.copy()
            new_label['location'] = rotated_locs[i].tolist()
            new_label['rotation_y'] = float(new_yaws[i])
            rotated_labels.append(new_label)
    else:
    rotated_labels = []

    return rotated_points, rotated_labels

def scale_distance_points_and_labels(points, labels, scale_factor):
    """
    Efficiently scale the distance of points and labels along X axis
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        scale_factor: Amount to scale (meters)
    
    Returns:
        scaled_points: Scaled point cloud
        scaled_labels: Updated labels with new positions
    """
    # Scale points efficiently using array operations
    scaled_points = points.copy()
    scaled_points[:, 0] += scale_factor
    
    if labels is None:
        return scaled_points, None
        
    # Process all non-DontCare labels at once
    scaled_labels = []
    valid_labels = [label for label in labels if label['type'] != 'DontCare']
    
    if valid_labels:
        # Extract locations as array
        locs = np.array([label['location'] for label in valid_labels])
        locs[:, 0] += scale_factor  # Scale X coordinates
        
        # Create new labels with scaled locations
        for i, label in enumerate(valid_labels):
        new_label = label.copy()
            new_label['location'] = locs[i].tolist()
            scaled_labels.append(new_label)
    
    # Add back DontCare labels unchanged
    scaled_labels.extend([label.copy() for label in labels if label['type'] == 'DontCare'])
    
    return scaled_points, scaled_labels

def shift_lateral_points_and_labels(points, labels, shift_amount):
    """
    Efficiently shift points and labels laterally (left/right along Y axis)
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        shift_amount: Amount to shift laterally (meters)
    
    Returns:
        shifted_points: Shifted point cloud
        shifted_labels: Updated labels with new positions
    """
    # Shift points efficiently using array operations
    shifted_points = points.copy()
    shifted_points[:, 1] += shift_amount
    
    if labels is None:
        return shifted_points, None
        
    # Process all non-DontCare labels at once
    shifted_labels = []
    valid_labels = [label for label in labels if label['type'] != 'DontCare']
    
    if valid_labels:
        # Extract locations as array
        locs = np.array([label['location'] for label in valid_labels])
        locs[:, 1] += shift_amount  # Shift Y coordinates
        
        # Create new labels with shifted locations
        for i, label in enumerate(valid_labels):
        new_label = label.copy()
            new_label['location'] = locs[i].tolist()
            shifted_labels.append(new_label)
    
    # Add back DontCare labels unchanged
    shifted_labels.extend([label.copy() for label in labels if label['type'] == 'DontCare'])
    
    return shifted_points, shifted_labels

def shift_vertical_points_and_labels(points, labels, shift_amount):
    """
    Efficiently shift points and labels vertically (up/down along Z axis)
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        shift_amount: Amount to shift vertically (meters)
    
    Returns:
        shifted_points: Shifted point cloud
        shifted_labels: Updated labels with new positions
    """
    # Shift points efficiently using array operations
    shifted_points = points.copy()
    shifted_points[:, 2] += shift_amount
    
    if labels is None:
        return shifted_points, None
        
    # Process all non-DontCare labels at once
    shifted_labels = []
    valid_labels = [label for label in labels if label['type'] != 'DontCare']
    
    if valid_labels:
        # Extract locations as array
        locs = np.array([label['location'] for label in valid_labels])
        locs[:, 2] += shift_amount  # Shift Z coordinates
        
        # Create new labels with shifted locations
        for i, label in enumerate(valid_labels):
        new_label = label.copy()
            new_label['location'] = locs[i].tolist()
            shifted_labels.append(new_label)
    
    # Add back DontCare labels unchanged
    shifted_labels.extend([label.copy() for label in labels if label['type'] == 'DontCare'])
    
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
    Efficiently filter points within specified X and Y range
    
    Args:
        points: Nx4 point cloud array (x, y, z, intensity)
        labels: List of label dictionaries
        x_min, x_max: X-axis range boundaries (forward direction)
        y_min, y_max: Y-axis range boundaries (lateral direction)
    
    Returns:
        filtered_points: Filtered point cloud
        filtered_labels: Updated labels that fall within the range
    """
    # Create mask for points using vectorized operations
    x_mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    y_mask = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    mask = x_mask & y_mask
    
    # Apply mask to points
    filtered_points = points[mask]
    
    if labels is None:
        return filtered_points, None
        
    # Process labels efficiently
    filtered_labels = []
    valid_labels = [label for label in labels if label['type'] != 'DontCare']
    
    if valid_labels:
        # Extract locations as array
        locs = np.array([label['location'] for label in valid_labels])
        
        # Create mask for labels
        label_x_mask = (locs[:, 0] >= x_min) & (locs[:, 0] <= x_max)
        label_y_mask = (locs[:, 1] >= y_min) & (locs[:, 1] <= y_max)
        label_mask = label_x_mask & label_y_mask
        
        # Add labels that fall within range
        for i, label in enumerate(valid_labels):
            if label_mask[i]:
                filtered_labels.append(label.copy())
    
    # Add back DontCare labels
    filtered_labels.extend([label.copy() for label in labels if label['type'] == 'DontCare'])
    
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
        bev_clean: BEV image without labels
        bev_with_boxes: BEV image with labeled bounding boxes
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
    
    # Create a processor for this specific range
    processor = RangeSpecificProcessor(x_min, x_max, y_min, y_max, z_min, z_max, 
                                      z_resolution, Height, Width)
    
    # Create clean BEV image from points
    bev_clean = processor.create_bev_image(points)
    
    # If no labels, return clean image and empty labels
    if not labels:
        return bev_clean, bev_clean.copy(), []
    
    # Create a copy for drawing boxes
    bev_with_boxes = bev_clean.copy()
    yolo_labels = []
    
    for label in labels:
        if label['type'] == 'DontCare':
            continue
            
        # Get label dimensions and location
        h, w, l = label['dimensions']  # height, width, length
        x, y, z = label['location']    # center coordinates
        
        # Calculate the corners of the 3D box (before rotation)
        # We'll check if any part of the box is within our range
        corners_3d = np.array([
            [l/2, w/2, 0],   # front-right
            [l/2, -w/2, 0],  # front-left
            [-l/2, -w/2, 0], # back-left
            [-l/2, w/2, 0]   # back-right
        ])
        
        # Apply rotation
        rotation_y = label['rotation_y']
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])
        
        # Rotate corners and add center position
        corners_3d = np.dot(R, corners_3d.T).T
        corners_3d[:, 0] += x
        corners_3d[:, 1] += y
        
        # Check if any corner is within range
        x_coords = corners_3d[:, 0]
        y_coords = corners_3d[:, 1]
        
        # If any part of the box is within range, include it
        if (np.any((x_coords >= x_min) & (x_coords <= x_max)) and 
            np.any((y_coords >= y_min) & (y_coords <= y_max))):
            
            # Transform 3D box to BEV
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                label['dimensions'],
                label['location'],
                label['rotation_y']
            )
            
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
    
    return bev_clean, bev_with_boxes, yolo_labels


class RangeSpecificProcessor:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, z_resolution, height, width):
        """
        Initialize processor for a specific range
        
        Args:
            x_min, x_max: Forward range boundaries (meters)
            y_min, y_max: Lateral range boundaries (meters)
            z_min, z_max: Height range boundaries (meters)
            z_resolution: Resolution for height slices
            height, width: Output image dimensions
        """
        # Fixed dimensions based on the range
        self.x_max = height - 1
        self.y_max = width - 1
        self.z_max = int((z_max - z_min) / z_resolution)
        
        # Store range parameters
        self.fwd_range = [x_min, x_max]
        self.side_range = [y_min, y_max]
        self.height_range = [z_min, z_max]
        
        # Calculate resolution to fill the entire image
        resolution_x = (x_max - x_min) / height
        resolution_y = (y_max - y_min) / width
        self.resolution = max(resolution_x, resolution_y)  # Use the max to maintain aspect ratio
        self.z_resolution = z_resolution
        
        # Color mapping for different object types
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
        """Create a BEV image from points"""
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
        
        # Define floor height range (-0.5m to 0.3m)
        floor_min_height = -2.0
        floor_max_height = 0.3
        
        # Calculate which height slices correspond to the floor
        floor_min_idx = max(0, int((floor_min_height - self.height_range[0]) / self.z_resolution))
        floor_max_idx = min(self.z_max, int((floor_max_height - self.height_range[0]) / self.z_resolution))
        
        # Assign different colors based on height, but make floor points black
        for i in range(self.z_max):
            # Skip floor points - they will remain black (0,0,0)
            if i >= floor_min_idx and i <= floor_max_idx:
                continue
            
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
        """Transform 3D box to BEV coordinates"""
        # Extract dimensions
        h, w, l = dimensions
        x, y, z = location
        
        # Calculate 3D box corners
        corners_3d = np.array([
            [l/2, w/2, 0],   # front-right
            [l/2, -w/2, 0],  # front-left
            [-l/2, -w/2, 0], # back-left
            [-l/2, w/2, 0]   # back-right
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])
        
        # Apply rotation and translation
        corners_3d = np.dot(R, corners_3d.T).T
        corners_3d[:, 0] += x
        corners_3d[:, 1] += y
        corners_3d[:, 2] += z
        
        # Transform to BEV image coordinates
        x_img = (-corners_3d[:, 1] / self.resolution).astype(np.int32)
        y_img = (-corners_3d[:, 0] / self.resolution).astype(np.int32)
        
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
        """Draw bounding box on BEV image"""
        # Create a copy of the image to draw on
        bev_with_box = bev_image.copy()
        
        # Get color for this object type
        color = self.color_map.get(obj_type, (255, 255, 255))  # Default: white
        
        # Convert corners to numpy array for drawing
        corners_np = np.array(corners_bev, dtype=np.int32)
        
        # Draw filled polygon with transparency
        overlay = bev_with_box.copy()
        cv2.fillPoly(overlay, [corners_np], color)
        
        # Apply the overlay with transparency
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, bev_with_box, 1 - alpha, 0, bev_with_box)
        
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
    
    def create_yolo_label(self, corners_bev, obj_type, img_shape):
        """Create YOLO format label based on BEV corners"""
        # Get image dimensions
        img_height, img_width = img_shape
        
        # Convert corners to numpy array
        corners = np.array(corners_bev)
        
        # Calculate bounding box
        x_min = np.min(corners[:, 0])
        y_min = np.min(corners[:, 1])
        x_max = np.max(corners[:, 0])
        y_max = np.max(corners[:, 1])
        
        # Ensure the box has width and height
        if x_max <= x_min:
            x_max = x_min + 1
        if y_max <= y_min:
            y_max = y_min + 1
        
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