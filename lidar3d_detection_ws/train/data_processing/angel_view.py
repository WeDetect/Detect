import numpy as np
import cv2
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.preproccesing_0 import PointCloudProcessor

def load_point_cloud(bin_file):
    """
    Load point cloud data from binary file
    
    Args:
        bin_file: Path to binary file
        
    Returns:
        points: Nx3 array of point cloud coordinates (x, y, z)
        intensity: N array of intensity values
    """
    # Load point cloud from binary file
    data = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    
    # Extract coordinates and intensity
    points = data[:, :3]  # x, y, z
    intensity = data[:, 3]  # intensity
    
    return points, intensity

def create_elevated_angled_view_optimized(points, intensity, img_width=800, img_height=600, 
                               fov_h=100, camera_height=5.0, pitch_angle=-25,
                               min_distance=1.0, max_distance=70.0):
    """
    Create a view from an elevated position with a downward angle (optimized version)
    
    Args:
        points: Nx3 array of point cloud coordinates (x, y, z)
        intensity: N array of intensity values
        img_width: Width of output image
        img_height: Height of output image
        fov_h: Horizontal field of view in degrees
        camera_height: Height in meters above the LiDAR
        pitch_angle: Downward pitch angle in degrees (negative is looking down)
        min_distance: Minimum distance for filtering
        max_distance: Maximum distance for filtering
        
    Returns:
        image: RGB image of elevated angled view
    """
    # Create empty image with background gradient
    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    y_coords = np.arange(img_height).reshape(-1, 1)
    ratio = y_coords / img_height
    image[:, :, 0] = np.broadcast_to((20 + 30 * ratio).astype(np.uint8), (img_height, img_width))
    image[:, :, 1] = np.broadcast_to((20 + 30 * ratio).astype(np.uint8), (img_height, img_width))
    image[:, :, 2] = np.broadcast_to((40 + 40 * ratio).astype(np.uint8), (img_height, img_width))
    
    # Calculate focal length based on field of view
    fx = img_width / (2 * np.tan(np.radians(fov_h/2)))
    fy = fx  # Same focal length for y (square pixels)
    
    # Principal point at image center
    cx = img_width / 2
    cy = img_height / 2
    
    # Create rotation matrix for pitch angle (rotation around Y-axis)
    pitch_rad = np.radians(pitch_angle)
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    # Fast pre-filtering of points: First calculate distances
    distances = np.sqrt(np.sum(points**2, axis=1))
    mask_dist = (distances >= min_distance) & (distances <= max_distance)
    
    # Early sampling for speed - keep only every N-th point
    sampling_rate = 3  # Process only every 3rd point for speed
    mask_sampling = np.zeros_like(mask_dist, dtype=bool)
    mask_sampling[::sampling_rate] = True
    
    combined_mask = mask_dist & mask_sampling
    filtered_points = points[combined_mask]
    filtered_intensities = intensity[combined_mask]
    filtered_distances = distances[combined_mask]
    
    if len(filtered_points) == 0:
        # Add view information and return if no points
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"Aerial View: {camera_height}m height, {pitch_angle}° pitch"
        cv2.putText(image, info_text, (img_width - 350, 30), font, 0.7, (220, 220, 220), 1)
        return image
    
    # Transform points in one vectorized operation
    # 1. Translate up by camera height
    translated_points = filtered_points.copy()
    translated_points[:, 2] -= camera_height
    
    # 2. Apply rotation matrix to all points at once
    rotated_points = np.dot(translated_points, R_pitch.T)
    
    # 3. Further filtering after rotation
    x_rot = rotated_points[:, 0]
    y_rot = rotated_points[:, 1]
    z_rot = rotated_points[:, 2]
    
    # Keep only points in front and within reasonable field of view
    mask_view = (x_rot > 0) & (np.abs(y_rot) < max_distance * 0.8) & (np.abs(z_rot) < max_distance * 0.8)
    
    rotated_points = rotated_points[mask_view]
    filtered_intensities = filtered_intensities[mask_view]
    filtered_distances = filtered_distances[mask_view]
    
    if len(rotated_points) == 0:
        # Add view information and return if no points after second filtering
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"Aerial View: {camera_height}m height, {pitch_angle}° pitch"
        cv2.putText(image, info_text, (img_width - 350, 30), font, 0.7, (220, 220, 220), 1)
        return image
    
    # Normalize distances and intensities for coloring
    max_intensity = np.max(filtered_intensities)
    normalized_distances = (max_distance - filtered_distances) / (max_distance - min_distance)
    normalized_distances = np.clip(normalized_distances, 0, 1)
    normalized_intensities = filtered_intensities / (max_intensity + 1e-10)
    
    # Project to image plane
    x_rot = rotated_points[:, 0]
    y_rot = rotated_points[:, 1]
    z_rot = rotated_points[:, 2]
    
    # Vectorized calculation of u, v coordinates
    # Only use points with positive x
    pos_x_mask = x_rot > 0
    
    if np.sum(pos_x_mask) == 0:
        # Add view information and return if no points in front
        font = cv2.FONT_HERSHEY_SIMPLEX
        info_text = f"Aerial View: {camera_height}m height, {pitch_angle}° pitch"
        cv2.putText(image, info_text, (img_width - 350, 30), font, 0.7, (220, 220, 220), 1)
        return image
    
    # Calculate image coordinates
    u = (cx - fx * 0.8 * (y_rot[pos_x_mask] / x_rot[pos_x_mask])).astype(np.int32)
    v = (cy - fy * 0.9 * (z_rot[pos_x_mask] / x_rot[pos_x_mask])).astype(np.int32)
    
    # Filter points within image boundaries
    in_img = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    u = u[in_img]
    v = v[in_img]
    
    distances_to_use = filtered_distances[pos_x_mask][in_img]
    intensities_to_use = normalized_intensities[pos_x_mask][in_img]
    norm_dist_to_use = normalized_distances[pos_x_mask][in_img]
    
    # Sort by distance to handle occlusion (farther points first)
    depth_order = np.argsort(-distances_to_use)
    u = u[depth_order]
    v = v[depth_order]
    norm_dist_to_use = norm_dist_to_use[depth_order]
    intensities_to_use = intensities_to_use[depth_order]
    
    # Render the points to the image (optimized loop)
    # Only process points in blocks for efficiency
    block_size = min(5000, len(u))  # Process in blocks to avoid memory issues
    for i in range(0, len(u), block_size):
        end_idx = min(i + block_size, len(u))
        
        # Enhanced color calculation to emphasize intensity differences
        # Increase the intensity influence on the color
        intensity_factor = intensities_to_use[i:end_idx] ** 0.7  # Apply power to enhance contrast
        
        # Create more distinct colors based on intensity and distance
        r = (200 * norm_dist_to_use[i:end_idx] + 55 * intensity_factor).astype(np.uint8)
        g = (220 * intensity_factor).astype(np.uint8)  # Make green channel more dependent on intensity
        b = (180 * (1 - norm_dist_to_use[i:end_idx]) + 75 * intensity_factor).astype(np.uint8)
        
        # Simplified point rendering with slightly larger points for better visibility
        for j in range(i, end_idx):
            # Use variable point size based on intensity for better differentiation
            point_size = max(1, min(3, int(1 + intensities_to_use[j] * 2)))
            cv2.circle(image, (u[j], v[j]), point_size, (int(b[j-i]), int(g[j-i]), int(r[j-i])), -1)
    
    # Minimal post-processing for speed
    info_text = f"Aerial View: {camera_height}m height, {pitch_angle}° pitch"
    cv2.putText(image, info_text, (img_width - 350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
    
    return image

def main():
    # Create output directory
    output_dir = "/lidar3d_detection_ws/train/output/view_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process 5 different point cloud files
    file_numbers = ["00000", "00001", "00002", "00003", "00004"]
    total_time = 0
    processing_times = []
    
    print("\n" + "="*50)
    print(" PERFORMANCE TEST: OPTIMIZED ELEVATED VIEW GENERATION ")
    print("="*50)
    
    for file_num in file_numbers:
        # Path to bin file
        bin_file = f"/lidar3d_detection_ws/data/augmented/{file_num}.bin"
        
        if not os.path.exists(bin_file):
            print(f"File {bin_file} not found. Skipping.")
            continue
            
        print(f"\nProcessing file: {bin_file}")
        
        # Load point cloud with intensity values
        points, intensity = load_point_cloud(bin_file)
        
        # Measure processing time
        start_time = time.time()
        
        # Create elevated angled view with optimized function
        elevated_view = create_elevated_angled_view_optimized(
            points, intensity, 
            img_width=800, img_height=600, 
            fov_h=100, camera_height=2.0, pitch_angle=-1,
            min_distance=1.0, max_distance=70.0
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        processing_times.append(processing_time)
        total_time += processing_time
        
        # Save the elevated view
        output_file = os.path.join(output_dir, f'elevated_view_{file_num}.png')
        cv2.imwrite(output_file, elevated_view)
        
        # Print processing time with highlighted color
        print(f"\033[1;32m>>> Processing time: {processing_time:.2f} milliseconds <<<\033[0m")
        print(f"Image saved to: {output_file}")
    
    # Print summary statistics
    if processing_times:
        avg_time = total_time / len(processing_times)
        print("\n" + "="*50)
        print(f"\033[1;33m PERFORMANCE SUMMARY \033[0m")
        print(f"Total files processed: {len(processing_times)}")
        print(f"Average processing time: \033[1;32m{avg_time:.2f} milliseconds\033[0m")
        print(f"Min processing time: \033[1;32m{min(processing_times):.2f} milliseconds\033[0m")
        print(f"Max processing time: \033[1;32m{max(processing_times):.2f} milliseconds\033[0m")
        print("="*50)
        print(f"\nAll images saved to: {output_dir}")
    else:
        print("\nNo files were processed successfully.")

if __name__ == "__main__":
    main()
