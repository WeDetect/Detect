import glob
import os

import cv2
import numpy as np
from tqdm import tqdm

from lidar3d_detection_ws.train.data_processing.augmentation import create_range_adapted_bev_image, rotate_points_and_labels, shift_vertical_points_and_labels
from lidar3d_detection_ws.train.data_processing.preproccesing_0 import PointCloudProcessor
from lidar3d_detection_ws.train.data_processing.preprocessing import read_bin_file, read_label_file


def generate_augmented_dataset(bin_dir=None, label_dir=None, bin_file=None, label_file=None, config_path=None, output_dir=None, verification_dir=None):
    """
    Generate augmented dataset from bin files
    
    Args:
        bin_dir: Directory containing bin files
        label_dir: Directory containing label files
        bin_file: Path to specific bin file (if processing a single file)
        label_file: Path to specific label file (if processing a single file)
        config_path: Path to config file
        output_dir: Output directory for augmented dataset
        verification_dir: Directory for verification images
        
    Returns:
        List of dictionaries with augmented data
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if verification_dir:
        os.makedirs(verification_dir, exist_ok=True)
    
    # Initialize processor
    processor = PointCloudProcessor(config_path=config_path)
    
    # Get list of bin files
    if bin_file and label_file:
        # Process a single file
        bin_files = [bin_file]
        print(f"Processing single file: {bin_file}")
    else:
        # Process all files in directory
        bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))
        print(f"Found {len(bin_files)} bin files")
    
    # Create augmented dataset
    all_augmented_data = []
    
    # Process each bin file
    for bin_idx, bin_file_path in enumerate(tqdm(bin_files, desc="Processing bin files")):
        try:
            # Get corresponding label file
            if bin_file and label_file:
                # Use provided label file
                label_file_path = label_file
            else:
                # Find corresponding label file in directory
                base_name = os.path.basename(bin_file_path)
                label_file_path = os.path.join(label_dir, os.path.splitext(base_name)[0] + '.txt')
            
            if not os.path.exists(label_file_path):
                print(f"Warning: Label file not found for {bin_file_path}")
                continue
            
            # Read point cloud and labels
            points = read_bin_file(bin_file_path)
            labels = read_label_file(label_file_path)
            
            # Add original data
            bev_image = processor.create_bev_image(points)
            
            # Process objects and create YOLO labels
            yolo_labels = []
            bev_with_boxes = bev_image.copy()
            
            for obj in labels:
                # Transform 3D box to BEV
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                
                # Draw box on the visualization image
                bev_with_boxes = processor.draw_box_on_bev(
                    bev_with_boxes, corners_bev, center_bev, obj['type']
                )
                
                # Create YOLO label
                yolo_label = processor.create_yolo_label(
                    corners_bev, obj['type'], bev_image.shape[:2]
                )
                yolo_labels.append(yolo_label)
            
            all_augmented_data.append({
                'bin_file': bin_file_path,
                'label_file': label_file_path,
                'bev_image': bev_with_boxes,
                'yolo_labels': yolo_labels,
                'augmentation': 'original'
            })
            
            # Save verification image if needed
            if verification_dir and bin_idx < 5:  # Limit to first 5 files
                verify_img = bev_with_boxes.copy()
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_original.png")
                cv2.imwrite(verify_path, verify_img)
            
            # Augmentation 1: Rotate 20 degrees left and right in 5-degree increments
            for angle in range(-20, 25, 5):
                if angle == 0:  # Skip 0 degrees as it's the original
                    continue
                
                # Rotate points and labels
                rotated_points, rotated_labels = rotate_points_and_labels(points, labels, angle)
                
                # Create BEV image
                rotated_bev_image = processor.create_bev_image(rotated_points)
                rotated_bev_with_boxes = rotated_bev_image.copy()
                
                # Process rotated objects and create YOLO labels
                rotated_yolo_labels = []
                
                for obj in rotated_labels:
                    # Transform 3D box to BEV
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    
                    # Draw box on the visualization image
                    rotated_bev_with_boxes = processor.draw_box_on_bev(
                        rotated_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                    
                    # Create YOLO label
                    yolo_label = processor.create_yolo_label(
                        corners_bev, obj['type'], rotated_bev_image.shape[:2]
                    )
                    rotated_yolo_labels.append(yolo_label)
                
                all_augmented_data.append({
                    'bin_file': bin_file_path,
                    'label_file': label_file_path,
                    'bev_image': rotated_bev_with_boxes,
                    'yolo_labels': rotated_yolo_labels,
                    'augmentation': f'rotate_{angle}'
                })
                
                # Save verification image if needed
                if verification_dir and bin_idx < 5:  # Limit to first 5 files
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_rotate_{angle}.png")
                    cv2.imwrite(verify_path, rotated_bev_with_boxes)
            
            # Augmentation 2: Shift height from -2m to +2m in 10cm increments
            for height_shift in np.arange(-2.0, 2.1, 0.1):
                if abs(height_shift) < 0.01:  # Skip near-zero shifts
                    continue
                
                # Shift points in height
                shifted_points = points.copy()
                shifted_points[:, 2] += height_shift  # Add to z coordinate
                
                # Labels remain the same, just update z position
                shifted_labels = []
                for label in labels:
                    shifted_label = label.copy()
                    x, y, z = label['location']
                    shifted_label['location'] = [x, y, z + height_shift]
                    shifted_labels.append(shifted_label)
                
                # Create BEV image
                shifted_bev_image = processor.create_bev_image(shifted_points)
                shifted_bev_with_boxes = shifted_bev_image.copy()
                
                # Process shifted objects and create YOLO labels
                shifted_yolo_labels = []
                
                for obj in shifted_labels:
                    # Transform 3D box to BEV
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    
                    # Draw box on the visualization image
                    shifted_bev_with_boxes = processor.draw_box_on_bev(
                        shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                    
                    # Create YOLO label
                    yolo_label = processor.create_yolo_label(
                        corners_bev, obj['type'], shifted_bev_image.shape[:2]
                    )
                    shifted_yolo_labels.append(yolo_label)
                
                # Only add if we have valid labels
                if shifted_yolo_labels:
                    all_augmented_data.append({
                        'bin_file': bin_file_path,
                        'label_file': label_file_path,
                        'bev_image': shifted_bev_with_boxes,
                        'yolo_labels': shifted_yolo_labels,
                        'augmentation': f'height_shift_{height_shift:.1f}'
                    })
                
                # Save verification image if needed (only for a few samples)
                if verification_dir and bin_idx < 5 and abs(height_shift) % 1.0 < 0.01:  # Save only for whole-meter shifts
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_height_shift_{height_shift:.1f}.png")
                    cv2.imwrite(verify_path, shifted_bev_with_boxes)
            
            # Augmentation 3: Shift horizontally from -5m to +5m in 1m increments
            for x_shift in range(-5, 6):
                if x_shift == 0:  # Skip zero shift
                    continue
                
                # Shift points horizontally
                shifted_points = points.copy()
                shifted_points[:, 0] += x_shift  # Add to x coordinate
                
                # Shift labels
                shifted_labels = []
                for label in labels:
                    shifted_label = label.copy()
                    x, y, z = label['location']
                    shifted_label['location'] = [x + x_shift, y, z]
                    shifted_labels.append(shifted_label)
                
                # Create BEV image
                shifted_bev_image = processor.create_bev_image(shifted_points)
                shifted_bev_with_boxes = shifted_bev_image.copy()
                
                # Process shifted objects and create YOLO labels
                shifted_yolo_labels = []
                
                for obj in shifted_labels:
                    # Transform 3D box to BEV
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    
                    # Draw box on the visualization image
                    shifted_bev_with_boxes = processor.draw_box_on_bev(
                        shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                    
                    # Create YOLO label
                    yolo_label = processor.create_yolo_label(
                        corners_bev, obj['type'], shifted_bev_image.shape[:2]
                    )
                    shifted_yolo_labels.append(yolo_label)
                
                # Only add if we have valid labels
                if shifted_yolo_labels:
                    all_augmented_data.append({
                        'bin_file': bin_file_path,
                        'label_file': label_file_path,
                        'bev_image': shifted_bev_with_boxes,
                        'yolo_labels': shifted_yolo_labels,
                        'augmentation': f'x_shift_{x_shift}'
                    })
                
                # Save verification image if needed
                if verification_dir and bin_idx < 5:
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_x_shift_{x_shift}.png")
                    cv2.imwrite(verify_path, shifted_bev_with_boxes)
            
            # Augmentation 4: Shift forward/backward from -5m to +5m in 1m increments
            for y_shift in range(-5, 6):
                if y_shift == 0:  # Skip zero shift
                    continue
                
                # Shift points forward/backward
                shifted_points = points.copy()
                shifted_points[:, 1] += y_shift  # Add to y coordinate
                
                # Shift labels
                shifted_labels = []
                for label in labels:
                    shifted_label = label.copy()
                    x, y, z = label['location']
                    shifted_label['location'] = [x, y + y_shift, z]
                    shifted_labels.append(shifted_label)
                
                # Create BEV image
                shifted_bev_image = processor.create_bev_image(shifted_points)
                shifted_bev_with_boxes = shifted_bev_image.copy()
                
                # Process shifted objects and create YOLO labels
                shifted_yolo_labels = []
                
                for obj in shifted_labels:
                    # Transform 3D box to BEV
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    
                    # Draw box on the visualization image
                    shifted_bev_with_boxes = processor.draw_box_on_bev(
                        shifted_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                    
                    # Create YOLO label
                    yolo_label = processor.create_yolo_label(
                        corners_bev, obj['type'], shifted_bev_image.shape[:2]
                    )
                    shifted_yolo_labels.append(yolo_label)
                
                # Only add if we have valid labels
                if shifted_yolo_labels:
                    all_augmented_data.append({
                        'bin_file': bin_file_path,
                        'label_file': label_file_path,
                        'bev_image': shifted_bev_with_boxes,
                        'yolo_labels': shifted_yolo_labels,
                        'augmentation': f'y_shift_{y_shift}'
                    })
                
                # Save verification image if needed
                if verification_dir and bin_idx < 5:
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_y_shift_{y_shift}.png")
                    cv2.imwrite(verify_path, shifted_bev_with_boxes)
            
            # Augmentation 5: Zoom on each object
            for label_idx, label in enumerate(labels):
                if label['type'] == 'DontCare':
                    continue
                
                # Get object location
                x, y, z = label['location']
                
                # Create a zoomed view centered on the object
                # Define a 30x30 meter box around the object
                x_min, x_max = x - 15, x + 15
                y_min, y_max = y - 15, y + 15
                
                # Create zoomed BEV image, BEV with boxes, and YOLO labels
                zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels = create_range_adapted_bev_image(
                    points, labels, x_min, x_max, y_min, y_max, config
                )
                
                # Only add if we have valid labels
                if zoom_yolo_labels:
                    all_augmented_data.append({
                        'bin_file': bin_file_path,
                        'label_file': label_file_path,
                        'bev_image': zoom_bev_clean,  # Use the clean image for training
                        'yolo_labels': zoom_yolo_labels,
                        'augmentation': f'zoom_obj_{label_idx}'
                    })
                
                # Save verification image if needed
                if verification_dir and bin_idx < 5 and label_idx < 3:  # Limit to first 5 files and 3 objects
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_zoom_obj_{label_idx}.png")
                    cv2.imwrite(verify_path, zoom_bev_with_boxes)  # Use the image with boxes for verification
            
            # Augmentation 6: Fixed range zoom-in views (7 different views)
            # Define 7 fixed regions to zoom in
            fixed_regions = [
                {"x_min": 0, "x_max": 10, "y_min": -5, "y_max": 5, "name": "front_center"},
                {"x_min": 0, "x_max": 10, "y_min": 5, "y_max": 15, "name": "front_right"},
                {"x_min": 0, "x_max": 10, "y_min": -15, "y_max": -5, "name": "front_left"},
                {"x_min": 10, "x_max": 20, "y_min": -5, "y_max": 5, "name": "mid_center"},
                {"x_min": 10, "x_max": 20, "y_min": 5, "y_max": 15, "name": "mid_right"},
                {"x_min": 10, "x_max": 20, "y_min": -15, "y_max": -5, "name": "mid_left"},
                {"x_min": 20, "x_max": 30, "y_min": -5, "y_max": 5, "name": "far_center"},
            ]
            
            print(f"Creating 7 fixed zoom-in views for {os.path.basename(bin_file_path)}")
            
            for region in fixed_regions:
                # Create zoomed BEV image, BEV with boxes, and YOLO labels
                zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels = create_range_adapted_bev_image(
                    points, labels, 
                    region["x_min"], region["x_max"], 
                    region["y_min"], region["y_max"], 
                    config
                )
                
                # Only add if we have valid labels
                if zoom_yolo_labels:
                    all_augmented_data.append({
                        'bin_file': bin_file_path,
                        'label_file': label_file_path,
                        'bev_image': zoom_bev_clean,  # Use the clean image for training
                        'yolo_labels': zoom_yolo_labels,
                        'augmentation': f'zoom_{region["name"]}'
                    })
                    
                    # For verification, use the image with boxes that was already created
                    if bin_idx < 5:
                        verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_zoom_{region['name']}.png")
                        cv2.imwrite(verify_path, zoom_bev_with_boxes)
            
            print(f"Created {len(all_augmented_data) - 1} augmented samples for {os.path.basename(bin_file_path)}")
            
            # Augmentation 7: Height shift (new)
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
                    all_augmented_data.append({
                        'bin_file': bin_file_path,
                        'label_file': label_file_path,
                        'bev_image': shifted_bev_image,  # Clean image without boxes
                        'yolo_labels': shifted_yolo_labels,
                        'augmentation': f'height_shift_{height_shift}'
                    })
                
                # Save verification image if needed
                if verification_dir and bin_idx < 5:
                    verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file_path))[0]}_height_shift_{height_shift}.png")
                    cv2.imwrite(verify_path, shifted_bev_with_boxes)
        
        except Exception as e:
            print(f"Error processing {bin_file_path}: {e}")
            continue
    
    print(f"Generated {len(all_augmented_data)} augmented samples")
    return all_augmented_data
