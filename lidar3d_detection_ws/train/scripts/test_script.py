import os
import sys
import numpy as np
import cv2
import yaml

# Add parent directory to path for importing modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the PointCloudProcessor from our preprocessing file
from data_processing.preproccesing_0 import PointCloudProcessor
from data_processing.augmentation import (rotate_points_and_labels, 
                                         scale_distance_points_and_labels,
                                         shift_lateral_points_and_labels,
                                         shift_vertical_points_and_labels,
                                         filter_points_by_range,
                                         save_label_file,
                                         create_range_adapted_bev_image)
from data_processing.preprocessing import read_bin_file, read_label_file, load_config

def create_processor():
    """Create and configure a PointCloudProcessor instance"""
    config_path = "/lidar3d_detection_ws/train/config/preprocessing_config.yaml"
    data_config_path = "/lidar3d_detection_ws/train/config/data.yaml"
    processor = PointCloudProcessor(config_path, data_config_path)
    return processor

def visualize_dataset(data_dir, labels_dir):
    """Visualize all bin files in the dataset with their corresponding labels"""
    # Create a processor instance
    processor = create_processor()
    
    # Get all bin files
    bin_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')])
    
    if not bin_files:
        print("No bin files found in directory")
        return
        
    print(f"Found {len(bin_files)} bin files")
    
    # Create window for visualization
    window_name = "BEV Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    
    # Visualize each bin file
    for bin_file in bin_files:
        base_name = os.path.basename(bin_file)
        print(f"Processing: {base_name}")
        
        # Define the corresponding label file
        label_base = os.path.splitext(base_name)[0] + '.txt'
        label_file = os.path.join(labels_dir, label_base)
        
        if not os.path.exists(label_file):
            print(f"No label file found for {base_name}")
            continue
            
        print(f"Using label file: {label_file}")
        
        # Process the point cloud and create BEV image with labels
        bev_image, _ = processor.process_point_cloud(bin_file, label_file)
        
        # Add file name to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bev_image, f"File: {base_name}", (10, bev_image.shape[0] - 20), 
                   font, 0.8, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow(window_name, bev_image)
        
        # Wait for 3 seconds or until a key is pressed
        key = cv2.waitKey(3000)
        if key == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

def visualize_specific_file(bin_file, label_file):
    """Visualize a specific bin file with its labels"""
    # Check if the bin file exists
    if not os.path.exists(bin_file):
        print(f"Error: File not found: {bin_file}")
        return
    
    print(f"Using point cloud file: {bin_file}")
    
    # Check if the label file exists
    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
        return
    
    print(f"Using label file: {label_file}")
    
    # Create a processor instance
    processor = create_processor()
    
    # Process the point cloud and get the BEV image with labels
    print("Processing point cloud and generating BEV image...")
    bev_image, yolo_labels = processor.process_point_cloud(bin_file, label_file)
    
    print(f"Generated {len(yolo_labels)} YOLO labels")
    for label in yolo_labels:
        print(f"  {label}")
    
    # Create window
    window_name = "BEV Visualization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    
    # Display file name on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bev_image, f"File: {os.path.basename(bin_file)}", 
               (10, bev_image.shape[0] - 20), font, 0.8, (255, 255, 255), 2)
    
    # Display the image
    cv2.imshow(window_name, bev_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

def visualize_with_controls(
        bin_file, 
        label_file, 
        config_path="train/config/preprocessing_config.yaml", 
        data_config_path="train/config/data.yaml"
    ):
    """Visualize a specific file with interactive controls for adjusting visualization params"""
    # Check if files exist
    if not os.path.exists(bin_file) or not os.path.exists(label_file):
        print(f"Error: Files not found")
        return
    
    print(f"Using point cloud file: {bin_file}")
    print(f"Using label file: {label_file}")
    
    # Load initial config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initial parameters from config
    resolution = float(config['DISCRETIZATION'])
    z_resolution = float(config['Z_RESOLUTION'])
    boundary = config['boundary']
    minX = float(boundary['minX'])
    maxX = float(boundary['maxX'])
    minY = float(boundary['minY'])
    maxY = float(boundary['maxY'])
    minZ = float(boundary['minZ'])
    maxZ = float(boundary['maxZ'])
    
    side_range = (minY, maxY)
    fwd_range = (minX, maxX)
    height_range = (minZ, maxZ)
    
    def update_view():
        """Update the BEV visualization with current parameters"""
        # Create temporary config for this view
        temp_config = config.copy()
        temp_config['DISCRETIZATION'] = resolution
        temp_config['Z_RESOLUTION'] = z_resolution
        temp_config['boundary']['minX'] = fwd_range[0]
        temp_config['boundary']['maxX'] = fwd_range[1]
        temp_config['boundary']['minY'] = side_range[0]
        temp_config['boundary']['maxY'] = side_range[1]
        temp_config['boundary']['minZ'] = height_range[0]
        temp_config['boundary']['maxZ'] = height_range[1]
        
        # Create a temporary config file
        temp_config_path = "/tmp/temp_preprocessing_config.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        # Create processor with the temporary config
        processor = PointCloudProcessor(temp_config_path, data_config_path)
        
        # Process the point cloud and get the BEV image with labels
        bev_image, _ = processor.process_point_cloud(bin_file, label_file)
        
        # Add parameter info to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = [
            f"Resolution: {resolution:.2f} m/pixel",
            f"Side range: {side_range[0]:.1f} to {side_range[1]:.1f} m",
            f"Forward range: {fwd_range[0]:.1f} to {fwd_range[1]:.1f} m",
            f"Height range: {height_range[0]:.1f} to {height_range[1]:.1f} m",
            f"Z resolution: {z_resolution:.2f} m"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(bev_image, text, (10, 30 + i*30), font, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(window_name, bev_image)
        
        # Remove the temporary file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    # Create window
    window_name = "BEV Visualization (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 608, 608)
    
    # Initial update
    update_view()
    
    print("\nControls:")
    print("  W/S - Increase/decrease resolution")
    print("  A/D - Increase/decrease side range")
    print("  Q/E - Increase/decrease forward range")
    print("  Z/X - Increase/decrease height range")
    print("  R/F - Increase/decrease Z resolution")
    print("  ESC - Exit")
    
    # Control loop
    while True:
        key = cv2.waitKey(0)
        
        # ESC to exit
        if key == 27:
            break
        
        # Resolution controls
        elif key == ord('w'):
            resolution = max(0.05, resolution - 0.05)
        elif key == ord('s'):
            resolution = min(0.5, resolution + 0.05)
        
        # Side range controls
        elif key == ord('a'):
            side_width = side_range[1] - side_range[0]
            side_range = (side_range[0] - 5, side_range[1] + 5)
        elif key == ord('d'):
            if side_range[1] - side_range[0] > 20:
                side_range = (side_range[0] + 5, side_range[1] - 5)
        
        # Forward range controls
        elif key == ord('q'):
            fwd_range = (fwd_range[0], fwd_range[1] + 10)
        elif key == ord('e'):
            if fwd_range[1] - fwd_range[0] > 20:
                fwd_range = (fwd_range[0], fwd_range[1] - 10)
        
        # Height range controls
        elif key == ord('z'):
            height_range = (height_range[0] - 0.5, height_range[1] + 0.5)
        elif key == ord('x'):
            if height_range[1] - height_range[0] > 1:
                height_range = (height_range[0] + 0.5, height_range[1] - 0.5)
        
        # Z resolution controls
        elif key == ord('r'):
            z_resolution = max(0.1, z_resolution - 0.1)
        elif key == ord('f'):
            z_resolution = min(1.0, z_resolution + 0.1)
        
        # Update view with new parameters
        update_view()
    
    cv2.destroyAllWindows()

def visualize_augmentations(data_dir, labels_dir, config_path="train/config/preprocessing_config.yaml"):
    """Visualize various augmentations on the first point cloud file in the dataset"""
    print("Visualizing different augmentations on the first bin file")
    # Get the first bin file
    bin_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')])
    if not bin_files:
        print("No bin files found in directory")
        return
    
    bin_file = bin_files[0]
    base_name = os.path.basename(bin_file)
    print(f"Using point cloud file: {base_name}")
    
    # Define the corresponding label file
    label_base = os.path.splitext(base_name)[0] + '.txt'
    label_file = os.path.join(labels_dir, label_base)
    
    if not os.path.exists(label_file):
        print(f"No label file found for {base_name}")
        return
    
    print(f"Using label file: {label_file}")
    
    # Create processor
    processor = create_processor()
    
    # Get config
    config = load_config(config_path)
    
    # Load point cloud and labels
    points = read_bin_file(bin_file)
    labels = read_label_file(label_file)
    
    # Create BEV image of original point cloud for comparison
    original_bev, _ = processor.process_point_cloud(bin_file, label_file)
    
    demonstrate_rotation(points, labels, processor, original_bev, config, bin_file, label_file)
    demonstrate_range_filtering(points, labels, processor, original_bev, config, bin_file, label_file)
    # Show different augmentations
    demonstrate_distance_scaling(points, labels, processor, original_bev, config, bin_file, label_file)
    demonstrate_lateral_shift(points, labels, processor, original_bev, config, bin_file, label_file)
    demonstrate_vertical_shift(points, labels, processor, original_bev, config, bin_file, label_file)
    
    # Add the new range filtering demonstration
    
    print("Augmentation demonstration complete")

def demonstrate_rotation(points, labels, processor, original_bev, config, bin_file, label_file):
    """Demonstrate rotation augmentation with different angles"""
    # עדכון הזוויות לפי הבקשה
    rotation_angles = [-45, -30, -15, 15, 30, 45]
    
    print("\nDemonstrating rotation augmentation...")
    
    # Create a single window to reuse
    window_name = "Rotation Augmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    for angle in rotation_angles:
        print(f"  Rotating by {angle} degrees")
        
        # Apply rotation
        rotated_points, rotated_labels = rotate_points_and_labels(points, labels, angle)
        
        # Save temporary bin file
        temp_bin = "/tmp/rotated_points.bin"
        rotated_points.astype(np.float32).tofile(temp_bin)
        
        # Save temporary label file
        temp_label = "/tmp/rotated_labels.txt"
        save_label_file(rotated_labels, temp_label)
        
        # Instead of using process_point_cloud, let's create our own BEV image with boxes
        # This will bypass any issues in the processor's transform_3d_box_to_bev function
        bev_image = processor.create_bev_image(rotated_points)
        bev_with_boxes = bev_image.copy()
        
        # Draw boxes directly
        for label in rotated_labels:
            if label['type'] == 'DontCare':
                continue
                
            # Transform 3D box to BEV using our fixed function
            corners_bev, center_bev = transform_3d_box_to_bev_fixed(
                label['dimensions'],
                label['location'],
                label['rotation_y'],
                processor.resolution,
                processor.side_range,
                processor.fwd_range
            )
            
            # Draw box on the image
            color = processor.colors.get(label['type'], (255, 255, 255))
            
            # Draw filled polygon with transparency
            corners_np = np.array(corners_bev, dtype=np.int32)
            overlay = bev_with_boxes.copy()
            cv2.fillPoly(overlay, [corners_np], color)
            
            # Apply the overlay with transparency
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, bev_with_boxes, 1 - alpha, 0, bev_with_boxes)
            
            # Draw outline
            for i in range(4):
                cv2.line(bev_with_boxes, 
                         corners_bev[i], 
                         corners_bev[(i+1)%4], 
                         color, 2)
            
            # Draw center
            cv2.circle(bev_with_boxes, center_bev, 3, color, -1)
        
        # Create combined visualization
        combined_vis = np.hstack((original_bev, bev_with_boxes))
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_vis, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_vis, f"Rotated by {angle} degrees", 
                   (original_bev.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the comparison
        cv2.imshow(window_name, combined_vis)
        
        # Wait for 4 seconds or key press
        key = cv2.waitKey(4000)
        
        # Remove temp files
        if os.path.exists(temp_bin):
            os.remove(temp_bin)
        if os.path.exists(temp_label):
            os.remove(temp_label)
            
        # Exit if ESC pressed
        if key == 27:
            cv2.destroyWindow(window_name)
            return
    
    # Close window after all rotations
    cv2.destroyWindow(window_name)

def transform_3d_box_to_bev_fixed(dimensions, location, rotation_y, resolution, side_range, fwd_range):
    """Fixed version of transform_3d_box_to_bev that correctly handles rotations"""
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
    x_img = (-corners_3d[:, 1] / resolution).astype(np.int32)
    y_img = (-corners_3d[:, 0] / resolution).astype(np.int32)
    
    # Shift to image coordinates for the specific range
    x_img -= int(np.floor(side_range[0] / resolution))
    y_img += int(np.floor(fwd_range[1] / resolution))
    
    # Center point in BEV
    center_x = int((-y / resolution) - int(np.floor(side_range[0] / resolution)))
    center_y = int((-x / resolution) + int(np.floor(fwd_range[1] / resolution)))
    
    corners_bev = list(zip(x_img, y_img))
    center_bev = (center_x, center_y)
    
    return corners_bev, center_bev

def demonstrate_distance_scaling(points, labels, processor, original_bev, config, bin_file, label_file):
    """Demonstrate distance scaling augmentation"""
    scaling_factors = [0.9, 0.8, 1.1, 1.2]
    
    print("\nDemonstrating distance scaling...")
    
    # Create a single window to reuse
    window_name = "Distance Scaling Augmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    for scale in scaling_factors:
        print(f"  Scaling by factor {scale}")
        
        # Apply scaling
        scaled_points, scaled_labels = scale_distance_points_and_labels(points, labels, scale)
        
        # Save temporary bin file
        temp_bin = "/tmp/scaled_points.bin"
        scaled_points.astype(np.float32).tofile(temp_bin)
        
        # Save temporary label file
        temp_label = "/tmp/scaled_labels.txt"
        save_label_file(scaled_labels, temp_label)
        
        # Create BEV image from augmented points
        augmented_bev, _ = processor.process_point_cloud(temp_bin, temp_label)
        
        # Create combined visualization
        combined_vis = np.hstack((original_bev, augmented_bev))
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_vis, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_vis, f"Distance scaled by {scale}", 
                   (original_bev.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the comparison
        cv2.imshow(window_name, combined_vis)
        
        # Wait for 4 seconds or key press
        key = cv2.waitKey(4000)
        
        # Remove temp files
        if os.path.exists(temp_bin):
            os.remove(temp_bin)
        if os.path.exists(temp_label):
            os.remove(temp_label)
            
        # Exit if ESC pressed
        if key == 27:
            cv2.destroyWindow(window_name)
            return

    # Close window after all scalings
    cv2.destroyWindow(window_name)


def demonstrate_lateral_shift(points, labels, processor, original_bev, config, bin_file, label_file):
    """Demonstrate lateral shifting augmentation"""
    shift_amounts = [-5, -2, 2, 5]
    
    print("\nDemonstrating lateral shifting...")
    
    # Create a single window to reuse
    window_name = "Lateral Shift Augmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    for shift in shift_amounts:
        print(f"  Shifting laterally by {shift} meters")
        
        # Apply shifting
        shifted_points, shifted_labels = shift_lateral_points_and_labels(points, labels, shift)
        
        # Save temporary bin file
        temp_bin = "/tmp/shifted_points.bin"
        shifted_points.astype(np.float32).tofile(temp_bin)
        
        # Save temporary label file
        temp_label = "/tmp/shifted_labels.txt"
        save_label_file(shifted_labels, temp_label)
        
        # Create BEV image from augmented points
        augmented_bev, _ = processor.process_point_cloud(temp_bin, temp_label)
        
        # Create combined visualization
        combined_vis = np.hstack((original_bev, augmented_bev))
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_vis, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_vis, f"Laterally shifted by {shift}m", 
                   (original_bev.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the comparison
        cv2.imshow(window_name, combined_vis)
        
        # Wait for 4 seconds or key press
        key = cv2.waitKey(4000)
        
        # Remove temp files
        if os.path.exists(temp_bin):
            os.remove(temp_bin)
        if os.path.exists(temp_label):
            os.remove(temp_label)
            
        # Exit if ESC pressed
        if key == 27:
            cv2.destroyWindow(window_name)
            return

    # Close window after all shifts
    cv2.destroyWindow(window_name)


def demonstrate_vertical_shift(points, labels, processor, original_bev, config, bin_file, label_file):
    """Demonstrate vertical shifting augmentation"""
    shift_amounts = [-1, -0.5, 0.5, 1]
    
    print("\nDemonstrating vertical shifting...")
    
    # Create a single window to reuse
    window_name = "Vertical Shift Augmentation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    for shift in shift_amounts:
        print(f"  Shifting vertically by {shift} meters")
        
        # Apply shifting
        shifted_points, shifted_labels = shift_vertical_points_and_labels(points, labels, shift)
        
        # Save temporary bin file
        temp_bin = "/tmp/shifted_points.bin"
        shifted_points.astype(np.float32).tofile(temp_bin)
        
        # Save temporary label file
        temp_label = "/tmp/shifted_labels.txt"
        save_label_file(shifted_labels, temp_label)
        
        # Create BEV image from augmented points
        augmented_bev, _ = processor.process_point_cloud(temp_bin, temp_label)
        
        # Create combined visualization
        combined_vis = np.hstack((original_bev, augmented_bev))
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_vis, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_vis, f"Vertically shifted by {shift}m", 
                   (original_bev.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the comparison
        cv2.imshow(window_name, combined_vis)
        
        # Wait for 4 seconds or key press
        key = cv2.waitKey(4000)
        
        # Remove temp files
        if os.path.exists(temp_bin):
            os.remove(temp_bin)
        if os.path.exists(temp_label):
            os.remove(temp_label)
            
        # Exit if ESC pressed
        if key == 27:
            cv2.destroyWindow(window_name)
            return

    # Close window after all shifts
    cv2.destroyWindow(window_name)

def demonstrate_range_filtering(points, labels, processor, original_bev, config, bin_file, label_file):
    """Demonstrate range filtering augmentation with adaptive resolution"""
    # Define different range filter options
    # Format: (x_min, x_max, y_min, y_max, description)
    range_filters = [
        (0, 5, -3, 3, "Close range (0-5m forward, ±3m lateral)"),
        (5, 10, -5, 5, "Medium range (5-10m forward, ±5m lateral)"),
        (10, 20, -10, 10, "Far range (10-20m forward, ±10m lateral)"),
        (0, 30, -2, 2, "Center corridor (0-30m forward, ±2m lateral)")
    ]
    
    print("\nDemonstrating range filtering with adaptive resolution...")
    
    # Create a single window to reuse
    window_name = "Range Filtering (Adaptive Resolution)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    # Resize original BEV to 608x608 if needed
    if original_bev.shape[0] != 608 or original_bev.shape[1] != 608:
        display_original = cv2.resize(original_bev, (608, 608))
    else:
        display_original = original_bev.copy()
    
    for x_min, x_max, y_min, y_max, desc in range_filters:
        print(f"  Filtering to range: {desc}")
        
        # Apply range filtering
        filtered_points, filtered_labels = filter_points_by_range(
            points, labels, x_min, x_max, y_min, y_max)
        
        # Skip if no points were found in this range
        if len(filtered_points) == 0:
            print(f"    No points found in range: {x_min}-{x_max}m (X), {y_min}-{y_max}m (Y)")
            continue
        
        # Create BEV image with adaptive resolution to fill the entire view
        # Now returns both clean and labeled images
        bev_clean, bev_with_boxes, yolo_labels = create_range_adapted_bev_image(
            points, labels, x_min, x_max, y_min, y_max, config)
        
        print(f"    Generated {len(yolo_labels)} YOLO labels")
        
        # Create side-by-side comparison of original vs clean vs with boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # First show original vs clean
        combined_vis1 = np.hstack((display_original, bev_clean))
        cv2.putText(combined_vis1, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_vis1, f"Filtered & Scaled (Clean): {desc}", 
                   (display_original.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the comparison
        cv2.imshow(window_name, combined_vis1)
        key = cv2.waitKey(2000)  # Show for 2 seconds
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return
            
        # Then show clean vs with boxes
        combined_vis2 = np.hstack((bev_clean, bev_with_boxes))
        cv2.putText(combined_vis2, "Clean BEV", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined_vis2, f"With Bounding Boxes: {desc}", 
                   (bev_clean.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the comparison
        cv2.imshow(window_name, combined_vis2)
        
        # Wait for 4 seconds or key press
        key = cv2.waitKey(4000)
        
        # Exit if ESC pressed
        if key == 27:
            cv2.destroyWindow(window_name)
            return

    # Close window after all filters
    cv2.destroyWindow(window_name)

def visualize_without_bounding_boxes(data_dir):
    """Visualize BEV images of point clouds without bounding boxes"""
    print("Visualizing dataset without bounding boxes...")
    # Get the bin files
    bin_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')])
    
    if not bin_files:
        print("No bin files found in directory")
        return
    
    # Create a processor
    processor = create_processor()
    
    # Create a window
    window_name = "BEV Visualization (No Bounding Boxes)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 608, 608)
    
    # Visualization loop
    for i, bin_file in enumerate(bin_files):
        base_name = os.path.basename(bin_file)
        print(f"Processing file {i+1}/{len(bin_files)}: {base_name}")
        
        # Load point cloud data directly
        points = read_bin_file(bin_file)
        
        # Create BEV image without any labels
        bev_image = processor.create_bev_image(points)
        
        # Display the BEV image
        cv2.imshow(window_name, bev_image)
        
        # Display file name on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bev_image, base_name, (10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Wait for key press or 4 seconds
        key = cv2.waitKey(4000)
        
        # Check if user wants to quit
        if key == 27:  # ESC key
            break
        # Pause if space is pressed
        elif key == 32:  # SPACE key
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def visualize_with_filled_boxes(data_dir, labels_dir):
    """Visualize all bin files in the dataset with filled bounding boxes"""
    # Create a processor instance
    processor = create_processor()
    
    # Get all bin files
    bin_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.bin')])
    
    if not bin_files:
        print("No bin files found in directory")
        return
        
    print(f"Found {len(bin_files)} bin files")
    
    # Create window for visualization
    window_name = "BEV Visualization with Filled Boxes"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    
    # Visualize each bin file
    for i, bin_file in enumerate(bin_files):
        base_name = os.path.basename(bin_file)
        label_base = os.path.splitext(base_name)[0] + '.txt'
        label_file = os.path.join(labels_dir, label_base)
        
        if not os.path.exists(label_file):
            print(f"Skipping {base_name}: No corresponding label file found")
            continue
        
        print(f"Processing file {i+1}/{len(bin_files)}: {base_name}")
        
        # Process the point cloud and get the BEV image with filled boxes
        bev_image, _ = processor.process_point_cloud_with_filled_boxes(bin_file, label_file)
        
        # Display file name on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bev_image, base_name, (10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Display the BEV image
        cv2.imshow(window_name, bev_image)
        
        # Wait for key press or timeout (4 seconds)
        key = cv2.waitKey(4000)
        
        # Check if user wants to quit
        if key == 27:  # ESC key
            break
        # Pause if space is pressed
        elif key == 32:  # SPACE key
            print("Paused. Press any key to continue...")
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()

def visualize_zoomed_regions(bin_file, label_file, config_path="train/config/preprocessing_config.yaml"):
    """Visualize zoomed-in regions of a point cloud with and without bounding boxes"""
    print("Visualizing zoomed-in regions of point cloud...")
    # Check if files exist
    if not os.path.exists(bin_file):
        print(f"Error: Bin file not found: {bin_file}")
        return
    if not os.path.exists(label_file):
        print(f"Error: Label file not found: {label_file}")
        return
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    print(f"Using point cloud file: {bin_file}")
    print(f"Using label file: {label_file}")
    
    # Load config
    config = load_config(config_path)
    
    # Load point cloud and labels
    points = read_bin_file(bin_file)
    labels = read_label_file(label_file)
    
    # Create processor for full view
    processor = create_processor()
    
    # Create full BEV image for reference
    full_bev, _ = processor.process_point_cloud(bin_file, label_file)
    
    # Define zoom regions to visualize
    # Format: (x_min, x_max, y_min, y_max, description)
    zoom_regions = [
        (0, 10, -5, 5, "Close range (0-10m forward, ±5m lateral)"),
        (10, 20, -10, 10, "Medium range (10-20m forward, ±10m lateral)"),
        (20, 40, -15, 15, "Far range (20-40m forward, ±15m lateral)"),
        (0, 50, -3, 3, "Center corridor (0-50m forward, ±3m lateral)")
    ]
    
    # Create window
    window_name = "Zoomed BEV Regions"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 600)
    
    # Show full BEV first
    full_bev_display = full_bev.copy()
    cv2.putText(full_bev_display, "Full BEV Image (Auto-advancing in 2 seconds)", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(window_name, full_bev_display)
    print("Showing full BEV image. Auto-advancing in 2 seconds...")
    
    # Wait for 2 seconds, but still allow ESC to exit
    key = cv2.waitKey(2000)
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        return
    
    for x_min, x_max, y_min, y_max, desc in zoom_regions:
        print(f"\nZooming to region: {desc}")
        
        # Create BEV images for this region
        bev_clean, bev_with_boxes, yolo_labels = create_range_adapted_bev_image(
            points, labels, x_min, x_max, y_min, y_max, config)
        
        print(f"  Generated {len(yolo_labels)} YOLO labels")
        for label in yolo_labels:
            print(f"    {label}")
        
        # Make sure we have valid images
        if bev_clean is None or bev_with_boxes is None:
            print("  Error: Failed to create BEV images for this region")
            continue
            
        print(f"  Clean BEV shape: {bev_clean.shape}, With boxes shape: {bev_with_boxes.shape}")
        
        # First show: Original vs Clean Zoomed
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Resize images to same height if needed
        if full_bev.shape[0] != bev_clean.shape[0]:
            display_height = 400
            h, w = full_bev.shape[:2]
            scale = display_height / h
            full_display = cv2.resize(full_bev, (int(w * scale), display_height))
            
            h, w = bev_clean.shape[:2]
            scale = display_height / h
            clean_display = cv2.resize(bev_clean, (int(w * scale), display_height))
        else:
            full_display = full_bev.copy()
            clean_display = bev_clean.copy()
        
        # Create side-by-side comparison
        comparison1 = np.hstack((full_display, clean_display))
        
        # Add text labels
        cv2.putText(comparison1, "Original Full BEV", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison1, f"Zoomed Clean BEV: {desc}", 
                   (full_display.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison1, "Auto-advancing in 2 seconds...", 
                   (10, full_display.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        
        # Show the comparison
        cv2.imshow(window_name, comparison1)
        print("  Showing Original vs Clean Zoomed. Auto-advancing in 2 seconds...")
        
        # Wait for 2 seconds, but still allow ESC to exit
        key = cv2.waitKey(2000)
        if key == 27:  # ESC
            break
            
        # Then show: Clean Zoomed vs With Boxes
        if bev_clean.shape[0] != bev_with_boxes.shape[0]:
            h, w = bev_clean.shape[:2]
            scale = display_height / h
            clean_display = cv2.resize(bev_clean, (int(w * scale), display_height))
            
            h, w = bev_with_boxes.shape[:2]
            scale = display_height / h
            boxes_display = cv2.resize(bev_with_boxes, (int(w * scale), display_height))
        else:
            clean_display = bev_clean.copy()
            boxes_display = bev_with_boxes.copy()
        
        # Create side-by-side comparison
        comparison2 = np.hstack((clean_display, boxes_display))
        
        # Add text labels
        cv2.putText(comparison2, "Zoomed Clean BEV", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison2, f"Zoomed BEV with Boxes: {desc}", 
                   (clean_display.shape[1] + 10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison2, "Auto-advancing in 2 seconds...", 
                   (10, clean_display.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        
        # Show the comparison
        cv2.imshow(window_name, comparison2)
        print("  Showing Clean Zoomed vs With Boxes. Auto-advancing in 2 seconds...")
        
        # Wait for 2 seconds, but still allow ESC to exit
        key = cv2.waitKey(2000)
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def main():
    point_cloud_dir = '/home/user/deep_learning/Detect/lidar3d_detection_ws/data/innoviz'
    label_dir = '/home/user/deep_learning/Detect/lidar3d_detection_ws/data/labels'
    
    print("Point Cloud BEV Visualization")
    print("Choose an option:")
    print("1: Visualize specific file (innoviz_00010.bin)")
    print("2: Visualize all files in dataset")
    print("3: Interactive visualization with controls")
    print("4: Demonstrate augmentation techniques")
    print("5: Visualize dataset without bounding boxes")
    print("6: Visualize all files in dataset with filled bounding boxes")
    print("7: Visualize zoomed-in regions with and without bounding boxes")
    
    choice = input("Enter your choice (1-7): ")
    
    if choice == '1':
        visualize_specific_file(point_cloud_dir, label_dir)
    elif choice == '2':
        visualize_dataset(point_cloud_dir, label_dir)
    elif choice == '3':
        visualize_with_controls(point_cloud_dir, label_dir)
    elif choice == '4':
        visualize_augmentations(point_cloud_dir, label_dir)
    elif choice == '5':
        visualize_without_bounding_boxes(point_cloud_dir)
    elif choice == '6':
        visualize_with_filled_boxes(point_cloud_dir, label_dir)
    elif choice == '7':
        visualize_zoomed_regions(point_cloud_dir, label_dir)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
