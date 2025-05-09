import os
import sys
import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from data_processing.preprocessing import load_config, read_bin_file, read_label_file, create_bev_image, convert_labels_to_yolo_format, save_yolo_labels

def generate_dataset(bin_dir, label_dir, config_path, output_images, output_labels, img_size):
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    config = load_config(config_path)

    bin_files = sorted(Path(bin_dir).glob("*.bin"))
    for i, bin_file in enumerate(bin_files):
        stem = f"{i:06d}"
        label_file = Path(label_dir) / f"{bin_file.stem}.txt"

        # Load point cloud and labels
        points = read_bin_file(bin_file)
        labels = read_label_file(label_file)

        # Generate BEV image
        bev_image, _ = create_bev_image(points, config, labels)

        # Convert KITTI labels to YOLO
        yolo_labels = convert_labels_to_yolo_format(labels, config, bev_image.shape[1], bev_image.shape[0])

        # Save image
        out_img_path = Path(output_images) / f"{stem}.png"
        cv2.imwrite(str(out_img_path), bev_image)

        # Save labels
        out_lbl_path = Path(output_labels) / f"{stem}.txt"
        save_yolo_labels(yolo_labels, out_lbl_path)

        print(f"Saved: {out_img_path}, {out_lbl_path}")
        
        # Debug: Print label content to verify
        with open(out_lbl_path, 'r') as f:
            content = f.read()
            print(f"Label content: {content}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_dir", default="/lidar3d_detection_ws/data/innoviz", help="Path to .bin files")
    parser.add_argument("--label_dir", default="/lidar3d_detection_ws/data/labels", help="Path to KITTI label .txt files")
    parser.add_argument("--config_path", default="/lidar3d_detection_ws/train/config/preprocessing_config.yaml", help="Path to preprocessing YAML")
    parser.add_argument("--img_output", default="/lidar3d_detection_ws/train/images", help="Output dir for BEV images")
    parser.add_argument("--label_output", default="/lidar3d_detection_ws/train/labels", help="Output dir for YOLO labels")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=608)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    args = parser.parse_args()

    # Step 1: Convert .bin + .txt to YOLO-format images and labels
    generate_dataset(args.bin_dir, args.label_dir, args.config_path, args.img_output, args.label_output, args.img_size)

    # Step 2: Create data.yaml
    data_yaml = {
        'train': str(Path(args.img_output).resolve()),
        'val': str(Path(args.img_output).resolve()),
        'nc': 4,
        'names': ['Car', 'Pedestrian', 'Cyclist', 'Truck']
    }
    os.makedirs("train/config", exist_ok=True)
    with open("train/config/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    # Step 3: Load YOLO model and modify for transfer learning
    model = YOLO("yolov5s.pt")
    model.model.model[-1].nc = 5
    model.model.model[-1].detect = True
    
    # Unfreeze more layers for better training
    # First freeze all layers
    for name, param in model.model.named_parameters():
        param.requires_grad = False
        
    # Unfreeze the last 10 layers for better transfer learning
    for i in range(10):  # Changed from 3 to 10
        if i < len(model.model.model):  # Safety check
            for name, param in model.model.model[-(i+1)].named_parameters():
                param.requires_grad = True
                print(f"Unfreezing layer: {name}")

    # Step 4: Train model
    model.train(
        data="train/config/data.yaml",
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        name="bev-transfer",
        project="train/output",
        device=args.device,
        verbose=True,
        single_cls=False
    )

    # Step 5: Save model
    model.save("train/output/bev_transfer_final.pt")

if __name__ == "__main__":
    main()