#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm

# Class mapping for special cases
CLASS_MAPPING = {
    'Van': 'Car',
    'Tram': 'Bus',
    'Person_sitting': 'Pedestrian'
}

VALID_CLASSES = {'Car', 'Pedestrian', 'Cyclist', 'Bus', 'Truck'}

# Only fixed heights - width and length will be taken from original labels
CLASS_FIXED_DIMENSIONS = {
    'Car':        {'h': 2.36, 'z_center': 1.05},
    'Pedestrian': {'h': 2.00, 'z_center': 1.0},
    'Cyclist':    {'h': 2.00, 'z_center': 1.0},
    'Bus':        {'h': 4.00, 'z_center': 2.0},
    'Truck':      {'h': 4.50, 'z_center': 2.25}
}

def find_unique_filename(base_name, bin_dir, label_dir, max_suffix=9999):
    """
    בודק אם קובץ קיים ומחפש שם חדש עם סיומת ייחודית, כמו 09999.txt
    """
    for suffix in range(max_suffix, -1, -1):
        name = f"{suffix:05d}"
        bin_path = os.path.join(bin_dir, name + ".bin")
        label_path = os.path.join(label_dir, name + ".txt")
        if not os.path.exists(bin_path) and not os.path.exists(label_path):
            return name
    raise RuntimeError("⚠️ All possible filenames are taken!")

def get_mapped_class(original_class):
    """Map special classes to their target class"""
    return CLASS_MAPPING.get(original_class, original_class)

def shift_vertical_points_and_labels(points, labels, shift_amount):
    shifted_points = np.copy(points)
    shifted_points[:, 2] += shift_amount  # Z axis של ה־LiDAR

    if labels is None:
        return shifted_points, None

    shifted_labels = []
    for label in labels:
        if label['type'] not in VALID_CLASSES:
            continue

        new_label = label.copy()
        x, y, z = new_label['location']
        new_label['location'] = [x, y, z + shift_amount]
        shifted_labels.append(new_label)

    return shifted_points, shifted_labels

def load_bin_file(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def save_bin_file(points, output_path):
    points.astype(np.float32).tofile(output_path)

def load_label_file(label_path):
    """Load KITTI format labels with class mapping"""
    if not os.path.exists(label_path):
        return None

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
                
            # Apply class mapping
            original_type = parts[0]
            mapped_type = get_mapped_class(original_type)
            
            label = {
                'type': mapped_type,
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],  # h, w, l
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],  # x, y, z
                'rotation_y': float(parts[14])
            }
            labels.append(label)
    return labels

def save_label_file_converted_format(labels, output_path):
    with open(output_path, 'w') as f:
        for label in labels:
            if label['type'] not in VALID_CLASSES:
                continue

            x_kitti, y_kitti, z_kitti = label['location']
            yaw_kitti = label['rotation_y']
            obj_type = label['type']

            if obj_type in CLASS_FIXED_DIMENSIONS:
                # Use fixed height but original width and length
                dims = CLASS_FIXED_DIMENSIONS[obj_type]
                h = dims['h']  # Fixed height
                _, w, l = label['dimensions']  # Original width and length
                z_new = dims['z_center']
            else:
                continue  # Skip unknown classes

            # המרת מערכת קואורדינטות:
            x_new = z_kitti - 1.5          # עומק קדימה ← X
            y_new = -x_kitti               # צדדים ← Y עם שיקוף
            z_new = z_new                  # גובה ← Z

            # תיקון YAW לפי השיקוף והמרה:
            yaw_new = -yaw_kitti + np.pi / 2

            # שמירה לפורמט המותאם
            line = f"{obj_type} 0 0 0 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {x_new:.8f} {y_new:.8f} {z_new:.8f} {yaw_new:.8f}"
            f.write(line + "\n")

def normalize_kitti_dataset(bin_dir, label_dir, output_bin_dir, output_label_dir, z_shift):
    os.makedirs(output_bin_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    bin_files = sorted([f for f in os.listdir(bin_dir) if f.endswith('.bin')])

    print(f"Processing {len(bin_files)} files with Z shift: {z_shift}m")

    for bin_file in tqdm(bin_files, desc="Normalizing KITTI dataset"):
        file_name = os.path.splitext(bin_file)[0]
        bin_path = os.path.join(bin_dir, bin_file)
        label_path = os.path.join(label_dir, file_name + '.txt')

        points = load_bin_file(bin_path)
        labels = load_label_file(label_path)

        shifted_points, shifted_labels = shift_vertical_points_and_labels(points, labels, z_shift)

        # מצא שם חדש אם יש כפילות
        unique_name = find_unique_filename(file_name, output_bin_dir, output_label_dir)

        output_bin_path = os.path.join(output_bin_dir, unique_name + ".bin")
        output_label_path = os.path.join(output_label_dir, unique_name + ".txt")

        save_bin_file(shifted_points, output_bin_path)

        if shifted_labels:
            save_label_file_converted_format(shifted_labels, output_label_path)
        else:
            open(output_label_path, 'w').close()

def main():
    bin_dir = "/lidar3d_detection_ws/data/kitti_velodyne_bins"
    label_dir = "/lidar3d_detection_ws/data/kitti_velodune_labels"
    output_bin_dir = "/lidar3d_detection_ws/data/kitti_velodyne_bins_normal"
    output_label_dir = "/lidar3d_detection_ws/data/kitti_velodune_labels_normal"
    z_shift = 1.7  # העלאה של כל הנקודות

    if not os.path.exists(bin_dir):
        print(f"Error: BIN directory does not exist: {bin_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"Error: LABEL directory does not exist: {label_dir}")
        return

    normalize_kitti_dataset(bin_dir, label_dir, output_bin_dir, output_label_dir, z_shift)

    print("✅ Normalization completed!")
    print(f"BIN files saved to: {output_bin_dir}")
    print(f"LABEL files saved to: {output_label_dir}")

if __name__ == "__main__":
    main()
