import glob
import os
import random
import shutil
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
import yaml

# Assuming these imports are available in your environment
from config.config import CLASS_NAMES, FIXED_ZOOM_REGIONS, HEIGHT_SHIFTS_CM, LATERAL_SHIFTS, NUM_CLASSES, ROTATION_ANGLES, SCALE_DISTANCES, TRAIN_VAL_SPLIT_RATIO, VERIFICATION_AUGMENTATION_LIMIT, VERIFICATION_ORIGINAL_LIMIT
from lidar3d_detection_ws.train.data_processing.augmentation import (
    shift_vertical_points_and_labels,
    rotate_points_and_labels,
    scale_distance_points_and_labels,
    shift_lateral_points_and_labels,
    create_range_adapted_bev_image
)
from lidar3d_detection_ws.train.data_processing.preproccesing_0 import PointCloudProcessor


def _create_output_directories(output_dir):
    """
    Creates necessary output directories for training.
    """
    print("Creating output directories...")
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = os.path.join(output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_img_dir = os.path.join(train_dir, 'images')
    train_label_dir = os.path.join(train_dir, 'labels')
    val_img_dir = os.path.join(val_dir, 'images')
    val_label_dir = os.path.join(val_dir, 'labels')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    verification_dir = os.path.join(output_dir, 'verification')
    os.makedirs(verification_dir, exist_ok=True)

    model_output_dir = os.path.join(output_dir, 'bev-from-scratch')
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(os.path.join(model_output_dir, 'train'), exist_ok=True)
    print("Output directories created.")
    return dataset_dir, train_img_dir, train_label_dir, val_img_dir, val_label_dir, verification_dir


def _process_single_sample(processor, bin_file, label_file, verification_dir, bin_idx):
    """
    Processes a single bin/label file pair, generates BEV images and YOLO labels.
    """
    try:
        points = processor.load_point_cloud(bin_file)
        labels = processor.load_labels(label_file)

        # Original BEV image and labels
        bev_image = processor.create_bev_image(points)
        yolo_labels = []
        for obj in labels:
            if obj['type'] == 'DontCare': # This 'DontCare' string could also be a constant if used more broadly
                continue
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            yolo_label = processor.create_yolo_label(
                corners_bev, obj['type'], bev_image.shape[:2]
            )
            yolo_labels.append(yolo_label)

        sample_data = [{
            'bin_file': bin_file,
            'bev_image': bev_image,
            'yolo_labels': yolo_labels,
            'augmentation': 'original'
        }]

        # Save verification image for original
        if bin_idx < VERIFICATION_ORIGINAL_LIMIT:
            bev_with_boxes = bev_image.copy()
            for obj in labels:
                if obj['type'] == 'DontCare':
                    continue
                corners_bev, center_bev = processor.transform_3d_box_to_bev(
                    obj['dimensions'], obj['location'], obj['rotation_y']
                )
                bev_with_boxes = processor.draw_box_on_bev(
                    bev_with_boxes, corners_bev, center_bev, obj['type']
                )
            verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_original.png")
            cv2.imwrite(verify_path, bev_with_boxes)

        return sample_data, points, labels

    except Exception as e:
        print(f"Error processing {bin_file}: {e}")
        return [], None, None


def _apply_augmentations(processor, points, labels, bin_file, verification_dir, bin_idx, config):
    """
    Applies various augmentations to the point cloud and labels.
    Returns a list of augmented samples.
    """
    augmented_samples = []

    # Augmentation 1: Rotate point cloud by different angles
    for angle in ROTATION_ANGLES:
        rotated_points, rotated_labels = rotate_points_and_labels(points, labels, angle)
        rotated_bev_image = processor.create_bev_image(rotated_points)
        rotated_yolo_labels = []
        for obj in rotated_labels:
            if obj['type'] == 'DontCare':
                continue
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            yolo_label = processor.create_yolo_label(
                corners_bev, obj['type'], rotated_bev_image.shape[:2]
            )
            rotated_yolo_labels.append(yolo_label)

        if rotated_yolo_labels:
            augmented_samples.append({
                'bin_file': bin_file,
                'bev_image': rotated_bev_image,
                'yolo_labels': rotated_yolo_labels,
                'augmentation': f'rotation_{angle}'
            })
            if bin_idx < VERIFICATION_AUGMENTATION_LIMIT:
                rotated_bev_with_boxes = rotated_bev_image.copy()
                for obj in rotated_labels:
                    if obj['type'] == 'DontCare':
                        continue
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    rotated_bev_with_boxes = processor.draw_box_on_bev(
                        rotated_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_rotation_{angle}.png")
                cv2.imwrite(verify_path, rotated_bev_with_boxes)

    # Augmentation 2: Scale distance (move objects closer/further)
    for scale in SCALE_DISTANCES:
        scaled_points, scaled_labels = scale_distance_points_and_labels(points, labels, scale)
        scaled_bev_image = processor.create_bev_image(scaled_points)
        scaled_yolo_labels = []
        for obj in scaled_labels:
            if obj['type'] == 'DontCare':
                continue
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            yolo_label = processor.create_yolo_label(
                corners_bev, obj['type'], scaled_bev_image.shape[:2]
            )
            scaled_yolo_labels.append(yolo_label)

        if scaled_yolo_labels:
            augmented_samples.append({
                'bin_file': bin_file,
                'bev_image': scaled_bev_image,
                'yolo_labels': scaled_yolo_labels,
                'augmentation': f'scale_{scale}'
            })
            if bin_idx < VERIFICATION_AUGMENTATION_LIMIT:
                scaled_bev_with_boxes = scaled_bev_image.copy()
                for obj in scaled_labels:
                    if obj['type'] == 'DontCare':
                        continue
                    corners_bev, center_bev = processor.transform_3d_box_to_bev(
                        obj['dimensions'], obj['location'], obj['rotation_y']
                    )
                    scaled_bev_with_boxes = processor.draw_box_on_bev(
                        scaled_bev_with_boxes, corners_bev, center_bev, obj['type']
                    )
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_scale_{scale}.png")
                cv2.imwrite(verify_path, scaled_bev_with_boxes)

    # Augmentation 3: Shift laterally (left/right)
    for x_shift in LATERAL_SHIFTS:
        shifted_points, shifted_labels = shift_lateral_points_and_labels(points, labels, x_shift)
        shifted_bev_image = processor.create_bev_image(shifted_points)
        shifted_yolo_labels = []
        for obj in shifted_labels:
            if obj['type'] == 'DontCare':
                continue
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            yolo_label = processor.create_yolo_label(
                corners_bev, obj['type'], shifted_bev_image.shape[:2]
            )
            shifted_yolo_labels.append(yolo_label)

        if shifted_yolo_labels:
            augmented_samples.append({
                'bin_file': bin_file,
                'bev_image': shifted_bev_image,
                'yolo_labels': shifted_yolo_labels,
                'augmentation': f'x_shift_{x_shift}'
            })
            if bin_idx < VERIFICATION_AUGMENTATION_LIMIT:
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
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_x_shift_{x_shift}.png")
                cv2.imwrite(verify_path, shifted_bev_with_boxes)

    # Augmentation 4: Fixed range zoom-in views
    for region in FIXED_ZOOM_REGIONS:
        zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels = create_range_adapted_bev_image(
            points, labels,
            region["x_min"], region["x_max"],
            region["y_min"], region["y_max"],
            config
        )
        if zoom_yolo_labels:
            augmented_samples.append({
                'bin_file': bin_file,
                'bev_image': zoom_bev_clean,
                'yolo_labels': zoom_yolo_labels,
                'augmentation': f'zoom_{region["name"]}'
            })
            if bin_idx < VERIFICATION_AUGMENTATION_LIMIT:
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_zoom_{region['name']}.png")
                cv2.imwrite(verify_path, zoom_bev_with_boxes)

    # Augmentation 5: Height shift
    for height_shift_cm in HEIGHT_SHIFTS_CM:
        height_shift_m = height_shift_cm / 100.0
        shifted_points, shifted_labels = shift_vertical_points_and_labels(points, labels, height_shift_m)
        shifted_bev_image = processor.create_bev_image(shifted_points)
        shifted_yolo_labels = []
        for obj in shifted_labels:
            if obj['type'] == 'DontCare':
                continue
            corners_bev, center_bev = processor.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            yolo_label = processor.create_yolo_label(
                corners_bev, obj['type'], shifted_bev_image.shape[:2]
            )
            shifted_yolo_labels.append(yolo_label)

        if shifted_yolo_labels:
            augmented_samples.append({
                'bin_file': bin_file,
                'bev_image': shifted_bev_image,
                'yolo_labels': shifted_yolo_labels,
                'augmentation': f'height_shift_{height_shift_cm}'
            })
            if bin_idx < VERIFICATION_AUGMENTATION_LIMIT:
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
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_height_shift_{height_shift_cm}.png")
                cv2.imwrite(verify_path, shifted_bev_with_boxes)
    return augmented_samples


def _split_and_save_dataset(all_data, dataset_dir, train_img_dir, train_label_dir, val_img_dir, val_label_dir):
    """
    Splits the processed data into training and validation sets and saves them.
    """
    print("\nSplitting and saving dataset...")
    unique_bin_files = list(set(item['bin_file'] for item in all_data))
    random.shuffle(unique_bin_files)

    # Use constant for split ratio
    train_bin_files = unique_bin_files[:int(TRAIN_VAL_SPLIT_RATIO * len(unique_bin_files))]
    val_bin_files = unique_bin_files[int(TRAIN_VAL_SPLIT_RATIO * len(unique_bin_files)):]

    train_data = [item for item in all_data if item['bin_file'] in train_bin_files]
    val_data = [item for item in all_data if item['bin_file'] in val_bin_files]

    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")

    for i, item in enumerate(tqdm(train_data, desc="Saving training data")):
        train_img_path = os.path.join(train_img_dir, f"train_{i}.png")
        train_label_path = os.path.join(train_label_dir, f"train_{i}.txt")
        cv2.imwrite(train_img_path, item['bev_image'])
        with open(train_label_path, 'w') as f:
            for label_str in item['yolo_labels']:
                f.write(label_str + '\n')

    for i, item in enumerate(tqdm(val_data, desc="Saving validation data")):
        val_img_path = os.path.join(val_img_dir, f"val_{i}.png")
        val_label_path = os.path.join(val_label_dir, f"val_{i}.txt")
        cv2.imwrite(val_img_path, item['bev_image'])
        with open(val_label_path, 'w') as f:
            for label_str in item['yolo_labels']:
                f.write(label_str + '\n')

    # Use constants for dataset config
    dataset_config = {
        'path': dataset_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': NUM_CLASSES,
        'names': CLASS_NAMES
    }
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    print("Dataset split and saved.")
    return dataset_yaml, val_data


def _train_yolo_model(dataset_yaml, output_dir, epochs, img_size, batch_size, device):
    """
    Initializes and trains the YOLOv8 model.
    """
    print("\n===== CREATING NEW MODEL FROM SCRATCH =====")
    model = YOLO('yolov8n.yaml') # This could also be a constant if you only use yolov8n

    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'

    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'workers': 8, # This could be a constant
        'project': os.path.join(output_dir, 'bev-from-scratch'),
        'name': 'train',
        'exist_ok': True,
        'pretrained': False,
        'optimizer': 'SGD', # This could be a constant
        'lr0': 0.01, # This could be a constant
        'weight_decay': 0.0005, # This could be a constant
        'cache': True,
        'data': dataset_yaml,
        'patience': 50, # This could be a constant
        'save_period': 10 # This could be a constant
    }

    print(f"Using device: {device}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    model.train(**train_args)

    weights_dir = os.path.join(output_dir, 'bev-from-scratch', 'train', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    final_weights = os.path.join(output_dir, 'bev_all_data_from_scratch_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
    return model, best_weights


def _run_inference_and_verify(model, val_data, verification_dir, processor):
    """
    Runs inference on a validation image and saves prediction and ground truth for verification.
    """
    print("\n===== RUNNING INFERENCE WITH TRAINED MODEL =====")
    if not val_data:
        print("No validation data available for inference.")
        return

    val_item = random.choice(val_data)
    val_img = val_item['bev_image']
    val_labels = val_item['yolo_labels']
    # val_bin_file = val_item['bin_file'] # Not used, can remove

    results = model.predict(val_img, conf=0.25) # This confidence threshold could be a constant
    result_img = results[0].plot()
    result_path = os.path.join(verification_dir, "model_prediction.png")
    cv2.imwrite(result_path, result_img)
    print(f"Inference result saved to: {result_path}")

    gt_img = val_img.copy()
    for label_str in val_labels:
        parts = label_str.split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            img_h, img_w = gt_img.shape[:2]
            x1 = int((x_center - width / 2) * img_w)
            y1 = int((y_center - height / 2) * img_h)
            x2 = int((x_center + width / 2) * img_w)
            y2 = int((y_center + height / 2) * img_h)

            color = processor.colors[processor.class_names[class_id]]
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(gt_img, processor.class_names[class_id],
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    gt_path = os.path.join(verification_dir, "ground_truth.png")
    cv2.imwrite(gt_path, gt_img)

    comparison = np.hstack((gt_img, result_img))
    comparison_path = os.path.join(verification_dir, "gt_vs_prediction.png")
    cv2.imwrite(comparison_path, comparison)
    print(f"Comparison image saved to: {comparison_path}")


def train_on_all_data_from_scratch(bin_dir, label_dir, config_path, output_dir, epochs=100, img_size=640, batch_size=16, device='cpu', augmentations=False, augmentation_factor=3):
    """
    Train a model from scratch on all data

    Args:
        bin_dir: Directory containing bin files
        label_dir: Directory containing label files
        config_path: Path to config file
        output_dir: Output directory
        epochs: Number of training epochs
        img_size: Image size
        batch_size: Batch size
        device: Device to use (cuda:0 or cpu)
        augmentations: Whether to use augmentations
        augmentation_factor: Number of augmented samples per original

    Returns:
        Path to best weights
    """
    print("\n===== TRAINING FROM SCRATCH ON ALL DATA =====")
    print(f"Using bin directory: {bin_dir}")
    print(f"Using label directory: {label_dir}")
    print(f"Using augmentations: {augmentations}")

    dataset_dir, train_img_dir, train_label_dir, val_img_dir, val_label_dir, verification_dir = \
        _create_output_directories(output_dir)

    processor = PointCloudProcessor(config_path=config_path)
    bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))
    print(f"Found {len(bin_files)} bin files")

    all_data = []
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for bin_idx, bin_file in enumerate(tqdm(bin_files, desc="Processing bin files")):
        label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.txt')
        
        sample_data, points, labels = _process_single_sample(processor, bin_file, label_file, verification_dir, bin_idx)
        if sample_data:
            all_data.extend(sample_data)
            if augmentations and points is not None and labels is not None:
                augmented_samples = _apply_augmentations(processor, points, labels, bin_file, verification_dir, bin_idx, config)
                all_data.extend(augmented_samples)
                print(f"Created {len(augmented_samples)} augmented samples for {os.path.basename(bin_file)}")

    print(f"Successfully processed {len(all_data)} samples (original + augmented)")

    dataset_yaml, val_data = _split_and_save_dataset(all_data, dataset_dir, train_img_dir, train_label_dir, val_img_dir, val_label_dir)

    model, best_weights = _train_yolo_model(dataset_yaml, output_dir, epochs, img_size, batch_size, device)

    _run_inference_and_verify(model, val_data, verification_dir, processor)

    return best_weights