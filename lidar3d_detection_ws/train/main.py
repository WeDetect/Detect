import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from data_processing.preprocessing import (
    load_config, read_bin_file, read_label_file,
    create_bev_image
)


def convert_to_yolo_format(label, class_id, config):
    """
    המרה מפורמט KITTI לפורמט YOLO
    תיקון: שינוי חישוב הגודל והמיקום של התיבות
    """
    x, y, z = label['location']
    h, w, l = label['dimensions']  # גובה, רוחב, אורך במטרים
    
    # המרה לקואורדינטות BEV
    x_bev = x / config['DISCRETIZATION']
    y_bev = y / config['DISCRETIZATION'] + config['BEV_WIDTH'] / 2
    
    # חישוב רוחב ואורך בפיקסלים
    width_px = w / config['DISCRETIZATION']
    height_px = l / config['DISCRETIZATION']
    
    # נרמול לטווח [0,1]
    x_center = x_bev / config['BEV_WIDTH']
    y_center = y_bev / config['BEV_HEIGHT']
    width = width_px / config['BEV_WIDTH']
    height = height_px / config['BEV_HEIGHT']
    
    # הגבלת הערכים לטווח תקין
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    
    # תיקון: הגדלת הגודל המינימלי של התיבות
    width = max(0.01, min(1, width))     # מינימום 1% מרוחב התמונה
    height = max(0.01, min(1, height))   # מינימום 1% מגובה התמונה
    
    # הדפסת דיבאג
    print(f"Debug - Class: {class_id}, Center: ({x_center:.4f}, {y_center:.4f}), Size: ({width:.4f}, {height:.4f})")
    
    return [class_id, x_center, y_center, width, height]


def save_yolo_labels(yolo_labels, label_path):
    with open(label_path, 'w') as f:
        for label in yolo_labels:
            f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def generate_dataset(bin_dir, label_dir, config_path, output_images, output_labels, img_size):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    config = load_config(config_path)
    
    # וידוא שהגדרות התמונה תואמות
    if config['BEV_HEIGHT'] != img_size or config['BEV_WIDTH'] != img_size:
        print(f"Warning: BEV dimensions ({config['BEV_HEIGHT']}x{config['BEV_WIDTH']}) " 
              f"don't match img_size ({img_size}). Updating config.")
        config['BEV_HEIGHT'] = img_size
        config['BEV_WIDTH'] = img_size

    bin_files = sorted(Path(bin_dir).glob("*.bin"))
    for bin_file in bin_files:
        stem = bin_file.stem
        label_file = Path(label_dir) / f"{stem}.txt"

        points = read_bin_file(bin_file)
        labels = read_label_file(label_file)
        bev_image, _ = create_bev_image(points, config, labels)

        # בדיקת גודל התמונה
        if bev_image.shape[0] != img_size or bev_image.shape[1] != img_size:
            print(f"Warning: Resizing BEV image from {bev_image.shape[:2]} to {img_size}x{img_size}")
            bev_image = cv2.resize(bev_image, (img_size, img_size))

        yolo_labels = []
        for label in labels:
            if label['type'] == 'DontCare':
                continue
            elif label['type'] == 'Car':
                class_id = 0
            elif label['type'] == 'Pedestrian':
                class_id = 1
            elif label['type'] == 'Cyclist':
                class_id = 2
            elif label['type'] == 'Truck':
                class_id = 3
            else:
                continue

            yolo_label = convert_to_yolo_format(label, class_id, config)
            yolo_labels.append(yolo_label)

        out_img_path = Path(output_images) / f"{stem}.png"
        cv2.imwrite(str(out_img_path), bev_image)

        out_lbl_path = Path(output_labels) / f"{stem}.txt"
        save_yolo_labels(yolo_labels, out_lbl_path)

        print(f"Saved: {out_img_path}, {out_lbl_path}")


def create_data_yaml(img_output, class_names, output_path="config/data.yaml"):
    data_yaml = {
        'train': str(Path(img_output).resolve()),
        'val': str(Path(img_output).resolve()),
        'nc': len(class_names),
        'names': class_names
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data_yaml, f)
    print(f"Created data YAML at {output_path}")


def setup_yolo_model(pretrained_model="yolov5s.pt", num_classes=4):
    model = YOLO(pretrained_model)
    model.model.model[-1].nc = num_classes
    model.model.model[-1].detect = True

    # שחרור כל השכבות לאימון
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    print(f"Model set up with {num_classes} classes")
    return model


def train_yolo_model(model, data_yaml_path, epochs, img_size, batch_size, output_dir):
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="bev-transfer",
        project=output_dir,
        device="cpu",
        single_cls=False,
        rect=True
        )
    model.save(f"{output_dir}/bev_transfer_final.pt")
    print(f"Model saved to {output_dir}/bev_transfer_final.pt")


def main():
    parser = argparse.ArgumentParser(description="BEV LiDAR Object Detection Pipeline")
    parser.add_argument("--bin_dir", default="/lidar3d_detection_ws/data/innoviz", help="Path to bin files")
    parser.add_argument("--label_dir", default="/lidar3d_detection_ws/data/labels", help="Path to label txt files")
    parser.add_argument("--config_path", default="/lidar3d_detection_ws/train/config/preprocessing_config.yaml", help="Path to config YAML")
    parser.add_argument("--img_output", default="/lidar3d_detection_ws/train/images", help="Output directory for BEV images")
    parser.add_argument("--label_output", default="/lidar3d_detection_ws/train/labels", help="Output directory for YOLO labels")
    parser.add_argument("--img_size", type=int, default=608, help="Image size for YOLOv5")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for training")
    args = parser.parse_args()

    print("Step 1: Generating BEV images and YOLO labels...")
    generate_dataset(args.bin_dir, args.label_dir, args.config_path, args.img_output, args.label_output, args.img_size)

    print("Step 2: Creating data YAML...")
    class_names = ['Car', 'Pedestrian', 'Cyclist', 'Truck']
    create_data_yaml(args.img_output, class_names)

    print("Step 3: Setting up YOLO model...")
    model = setup_yolo_model(num_classes=len(class_names))

    print("Step 4: Training YOLO model...")
    train_yolo_model(model, "config/data.yaml", args.epochs, args.img_size, args.batch, "output")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()