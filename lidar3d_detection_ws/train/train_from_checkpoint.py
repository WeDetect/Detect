import os
import glob
import cv2
import yaml
import random
import shutil
import sys
from tqdm import tqdm
import torch
from ultralytics import YOLO

from lidar3d_detection_ws.train.data_processing.augmentation import create_range_adapted_bev_image, rotate_points_and_labels, scale_distance_points_and_labels, shift_lateral_points_and_labels, shift_vertical_points_and_labels
from lidar3d_detection_ws.train.data_processing.preproccesing_0 import PointCloudProcessor


def setup_directories(output_dir):
    """
    Creates necessary output and dataset directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset_dir = os.path.join(output_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)
    verification_dir = os.path.join(output_dir, 'verification')
    os.makedirs(verification_dir, exist_ok=True)
    return dataset_dir, verification_dir

def process_single_sample(bin_file, label_file, processor, verification_dir, bin_idx, augmentations, config_path):
    """
    Processes a single bin and label file, generating BEV images and YOLO labels,
    and applying augmentations if enabled.
    """
    all_data_for_sample = []
    
    # Process original data
    points = processor.load_point_cloud(bin_file)
    labels = processor.load_labels(label_file)
    bev_image = processor.create_bev_image(points)
    yolo_labels = []
    for obj in labels:
        if obj['type'] == 'DontCare':
            continue
        corners_bev, center_bev = processor.transform_3d_box_to_bev(
            obj['dimensions'], obj['location'], obj['rotation_y']
        )
        yolo_label = processor.create_yolo_label(
            corners_bev, obj['type'], bev_image.shape[:2]
        )
        yolo_labels.append(yolo_label)
    all_data_for_sample.append({
        'bin_file': bin_file,
        'bev_image': bev_image,
        'yolo_labels': yolo_labels,
        'augmentation': 'none'
    })

    # Create and save verification image for original data
    if bin_idx < 20:
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

    if augmentations:
        # Augmentation 1: Rotate point cloud by different angles
        for angle in [-15, 15]:
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
                all_data_for_sample.append({
                    'bin_file': bin_file,
                    'bev_image': rotated_bev_image,
                    'yolo_labels': rotated_yolo_labels,
                    'augmentation': f'rotation_{angle}'
                })
            if bin_idx < 5:
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
        for scale in [-2.5, 2.5]:
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
                all_data_for_sample.append({
                    'bin_file': bin_file,
                    'bev_image': scaled_bev_image,
                    'yolo_labels': scaled_yolo_labels,
                    'augmentation': f'scale_{scale}'
                })
            if bin_idx < 5:
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
        for x_shift in [-2.5, 2.5]:
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
                all_data_for_sample.append({
                    'bin_file': bin_file,
                    'bev_image': shifted_bev_image,
                    'yolo_labels': shifted_yolo_labels,
                    'augmentation': f'x_shift_{x_shift}'
                })
            if bin_idx < 5:
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
        fixed_regions = [
            {"x_min": 0, "x_max": 10, "y_min": -5, "y_max": 5, "name": "front_center"},
            {"x_min": 10, "x_max": 20, "y_min": -5, "y_max": 5, "name": "mid_center"}
        ]
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for region in fixed_regions:
            zoom_bev_clean, zoom_bev_with_boxes, zoom_yolo_labels = create_range_adapted_bev_image(
                points, labels,
                region["x_min"], region["x_max"],
                region["y_min"], region["y_max"],
                config
            )
            if zoom_yolo_labels:
                all_data_for_sample.append({
                    'bin_file': bin_file,
                    'bev_image': zoom_bev_clean,
                    'yolo_labels': zoom_yolo_labels,
                    'augmentation': f'zoom_{region["name"]}'
                })
            if bin_idx < 5:
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_zoom_{region['name']}.png")
                cv2.imwrite(verify_path, zoom_bev_with_boxes)
        
        # Augmentation 5: Height shift
        for height_shift in [-10, -5, 5, 10]:
            height_shift_m = height_shift / 100.0
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
                all_data_for_sample.append({
                    'bin_file': bin_file,
                    'bev_image': shifted_bev_image,
                    'yolo_labels': shifted_yolo_labels,
                    'augmentation': f'height_shift_{height_shift}'
                })
            if bin_idx < 5:
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
                verify_path = os.path.join(verification_dir, f"{os.path.splitext(os.path.basename(bin_file))[0]}_height_shift_{height_shift}.png")
                cv2.imwrite(verify_path, shifted_bev_with_boxes)

    print(f"Created {len(all_data_for_sample) - 1} augmented samples for {os.path.basename(bin_file)}")
    return all_data_for_sample


def prepare_dataset(bin_dir, label_dir, config_path, output_dir, augmentations):
    """
    Prepares the dataset by processing bin files, generating BEV images and YOLO labels,
    and optionally applying augmentations.
    """
    dataset_dir, verification_dir = setup_directories(output_dir)
    bin_files = sorted(glob.glob(os.path.join(bin_dir, '*.bin')))
    print(f"Found {len(bin_files)} bin files")

    processor = PointCloudProcessor(config_path=config_path)
    all_data = []

    for bin_idx, bin_file in enumerate(tqdm(bin_files, desc="Processing bin files")):
        try:
            label_file = os.path.join(label_dir, os.path.splitext(os.path.basename(bin_file))[0] + '.txt')
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for {os.path.basename(bin_file)}, skipping.")
                continue
            
            sample_data = process_single_sample(bin_file, label_file, processor, verification_dir, bin_idx, augmentations, config_path)
            all_data.extend(sample_data)
        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue

    print(f"Successfully processed {len(all_data)} samples (original + augmented)")
    return all_data, dataset_dir

def split_and_save_data(all_data, dataset_dir):
    """
    Splits the processed data into training and validation sets and saves them to disk.
    """
    train_img_dir = os.path.join(dataset_dir, 'train', 'images')
    train_label_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_img_dir = os.path.join(dataset_dir, 'val', 'images')
    val_label_dir = os.path.join(dataset_dir, 'val', 'labels')

    unique_bin_files = list(set(item['bin_file'] for item in all_data))
    random.shuffle(unique_bin_files)

    train_val_split = 0.8
    train_bin_files = unique_bin_files[:int(train_val_split * len(unique_bin_files))]
    val_bin_files = unique_bin_files[int(train_val_split * len(unique_bin_files)):]

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

    dataset_config = {
        'path': dataset_dir,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'nc': 5,
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Bus', 'Truck']
    }
    dataset_yaml = os.path.join(dataset_dir, 'dataset.yaml')
    with open(dataset_yaml, 'w') as f:
        yaml.dump(dataset_config, f)
    return dataset_yaml

def train_yolo_model(checkpoint_path, output_dir, epochs, img_size, batch_size, device, dataset_yaml):
    """
    Loads a YOLO model from a checkpoint and trains it.
    """
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        device = 'cpu'

    print(f"\n===== LOADING MODEL FROM CHECKPOINT: {checkpoint_path} =====")
    model = YOLO(checkpoint_path)

    train_args = {
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'workers': 8,
        'project': os.path.join(output_dir, 'bev-continued'),
        'name': 'train',
        'exist_ok': True,
        'optimizer': 'SGD',
        'lr0': 0.001,
        'weight_decay': 0.0005,
        'cache': True,
        'data': dataset_yaml,
        'patience': 10,
        'save_period': 10
    }

    print(f"Using device: {device}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    model.train(**train_args)

    weights_dir = os.path.join(output_dir, 'bev-continued', 'train', 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    final_weights = os.path.join(output_dir, 'bev_continued_final.pt')
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)

    return best_weights

def train_from_checkpoint(bin_dir, label_dir, config_path, output_dir, checkpoint_path='train/output/best.pt', epochs=100, img_size=640, batch_size=16, device='cpu', augmentations=False):
    """
    Continue training a YOLO model from a checkpoint.

    Args:
        bin_dir (str): Directory containing bin files.
        label_dir (str): Directory containing label files.
        config_path (str): Path to config file.
        output_dir (str): Output directory.
        checkpoint_path (str, optional): Path to model checkpoint to continue from. Defaults to 'train/output/best.pt'.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        img_size (int, optional): Image size. Defaults to 640.
        batch_size (int, optional): Batch size. Defaults to 16.
        device (str, optional): Device to use (cuda:0 or cpu). Defaults to 'cpu'.
        augmentations (bool, optional): Whether to use augmentations. Defaults to False.

    Returns:
        str: Path to best weights, or None if checkpoint not found.
    """
    print("\n===== CONTINUING TRAINING FROM CHECKPOINT =====")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using bin directory: {bin_dir}")
    print(f"Using label directory: {label_dir}")
    print(f"Using augmentations: {augmentations}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None

    # Prepare dataset
    all_data, dataset_dir = prepare_dataset(bin_dir, label_dir, config_path, output_dir, augmentations)
    if not all_data:
        print("No data processed. Exiting training.")
        return None

    # Split and save data
    dataset_yaml = split_and_save_data(all_data, dataset_dir)

    # Train the YOLO model
    best_weights_path = train_yolo_model(checkpoint_path, output_dir, epochs, img_size, batch_size, device, dataset_yaml)

    return best_weights_path