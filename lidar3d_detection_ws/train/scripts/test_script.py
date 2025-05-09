import os
import sys
import numpy as np
import cv2
from pathlib import Path
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_processing.preprocessing import read_bin_file, read_label_file, create_bev_image, draw_labels_on_bev, create_labeled_bev_image, load_config
from data_processing.augmentation import rotate_points_and_labels, scale_distance_points_and_labels, shift_lateral_points_and_labels, shift_vertical_points_and_labels, create_custom_bev_image

def is_valid_lidar_bin(file_path):
    """Check if a file is a valid LiDAR point cloud bin file"""
    try:
        # Try to load the file as a point cloud
        points = np.fromfile(file_path, dtype=np.float32)
        # Check if the size is divisible by 4 (x, y, z, intensity)
        if len(points) % 4 != 0:
            return False
        # Reshape and check if it looks like a point cloud (has reasonable number of points)
        points = points.reshape(-1, 4)
        return len(points) > 100  # Assume a valid point cloud has at least 100 points
    except Exception:
        return False

def main():
    # Load configuration
    config_path = os.path.join(parent_dir, 'config', 'preprocessing_config.yaml')
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Define the paths to the bin and label files
    bin_file = '/lidar3d_detection_ws/data/innoviz/innoviz_00006.bin'
    label_file = '/lidar3d_detection_ws/data/labels/innoviz_00006.txt'
    
    # Check if the bin file is valid
    if not is_valid_lidar_bin(bin_file):
        print("Error: No valid LiDAR .bin files found in the specified path")
        return
    
    print(f"Using point cloud file: {bin_file}")
    
    # Load point cloud
    print("Loading point cloud...")
    original_points = read_bin_file(bin_file)
    print(f"Loaded {len(original_points)} points")
    
    # Try to load the label file
    original_labels = None
    if os.path.exists(label_file):
        print(f"Found label file: {label_file}")
        original_labels = read_label_file(label_file)
        print(f"Loaded {len(original_labels)} labels")
    else:
        print("No label file found. Rotation will be shown without labels.")
    
    # Create original BEV image without labels
    original_bev_no_labels, white_dots_positions = create_bev_image(original_points, config, original_labels)
    
    # Create original BEV image with labels
    original_bev_with_labels = draw_labels_on_bev(original_bev_no_labels.copy(), original_labels, config, white_dots_positions)
    
    # Display both versions side by side
    combined_img = np.hstack((original_bev_no_labels, original_bev_with_labels))
    
    # Add title and labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_img, "BEV without labels", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined_img, "BEV with labels", (original_bev_no_labels.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Create window
    window_name = "BEV Comparison - Without vs With Labels"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    # Display the combined image
    cv2.imshow(window_name, combined_img)
    cv2.waitKey(0)  # Wait until a key is pressed
    
    # Rotate and display for each angle from -30 to +30 degrees
    for angle in range(-30, 35, 5):
        print(f"Rotating by {angle} degrees...")
        rotated_points, rotated_labels = rotate_points_and_labels(original_points, original_labels or [], angle)
        
        # Create BEV image for rotated points (without labels)
        rotated_bev_no_labels, rotated_white_dots = create_bev_image(rotated_points, config, rotated_labels)
        
        # Create BEV image for rotated points (with labels)
        rotated_bev_with_labels = draw_labels_on_bev(rotated_bev_no_labels.copy(), rotated_labels, config, rotated_white_dots)
        
        # Combine original and rotated images side by side (both with labels)
        combined_img = np.hstack((original_bev_with_labels, rotated_bev_with_labels))
        
        # Add title and labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_img, "Original (0°)", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(combined_img, f"Rotated {angle}°", (original_bev_with_labels.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Create window
        window_name = f"BEV Comparison - Original vs {angle}° Rotation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 600)
        
        # Display the combined image
        cv2.imshow(window_name, combined_img)
        
        # Wait for 3 seconds before moving to the next angle
        cv2.waitKey(3000)
        
    # Rest of the code remains similar, just update to use the new functions
    # For brevity, I'll update just one more example:
    
    # Example of using create_custom_bev_image
    min_x, max_x = 0, 30
    min_y, max_y = -30, 30
    min_z, max_z = -2, 6
    print(f"Creating custom BEV image with boundaries X:({min_x},{max_x}), Y:({min_y},{max_y}), Z:({min_z},{max_z})")
    
    # Create custom BEV without labels
    custom_config = config.copy()
    custom_config['boundary'] = {
        'minX': min_x, 'maxX': max_x,
        'minY': min_y, 'maxY': max_y,
        'minZ': min_z, 'maxZ': max_z
    }
    custom_bev_no_labels, custom_white_dots = create_bev_image(original_points, custom_config, original_labels)
    
    # Create custom BEV with labels
    custom_bev_with_labels = draw_labels_on_bev(custom_bev_no_labels.copy(), original_labels, custom_config, custom_white_dots)
    
    # Display both versions side by side
    combined_img = np.hstack((custom_bev_no_labels, custom_bev_with_labels))
    
    # Add title and labels
    cv2.putText(combined_img, "Custom BEV without labels", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined_img, "Custom BEV with labels", (custom_bev_no_labels.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
    
    # Create window
    window_name = "Custom BEV Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    # Display the combined image
    cv2.imshow(window_name, combined_img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()