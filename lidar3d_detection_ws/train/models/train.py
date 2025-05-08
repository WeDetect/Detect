import os
import sys
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import glob

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_processing.dataset import BEVDataset
from models.yolo_bev import YOLOBEV
from models.loss import YOLOLoss
from models.evaluate import non_max_suppression
from data_processing.preprocessing import load_config, calculate_anchors_kmeans
from sklearn.cluster import KMeans

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load class names
    with open(args.classes_json, 'r') as f:
        import json
        classes_data = json.load(f)
        num_classes = len(classes_data['classes'])
    
    # Find all bin files
    bin_files = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
    label_files = [os.path.join(args.label_dir, Path(f).stem + ".txt") for f in bin_files]
    
    # Filter to only include files with corresponding labels
    valid_pairs = [(b, l) for b, l in zip(bin_files, label_files) if os.path.exists(l)]
    bin_files = [b for b, _ in valid_pairs]
    label_files = [l for _, l in valid_pairs]
    
    print(f"Found {len(bin_files)} valid bin/label pairs")
    
    # Load config
    config = load_config(args.config_path)
    
    # Calculate anchors using K-means if requested
    if hasattr(args, 'calculate_anchors') and args.calculate_anchors:
        anchors = calculate_anchors_kmeans(label_files, config)
        print("Using K-means calculated anchors")
    else:
        # Default anchors (should be replaced with calculated ones)
        anchors = [
            # Small objects (76x76 grid)
            [(8, 8), (12, 12), (16, 16)],
            # Medium objects (38x38 grid)
            [(20, 20), (28, 28), (36, 36)],
            # Large objects (19x19 grid)
            [(45, 45), (60, 60), (80, 80)]
        ]
        print("Using default anchors")
    
    # Initialize model
    model = YOLOBEV(num_classes=num_classes, img_size=args.img_size)
    model.anchors = anchors  # Set the calculated anchors
    model.to(device)
    
    # Initialize loss function
    loss_fn = YOLOLoss(anchors, num_classes, args.img_size)
    loss_fn.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create dataset and dataloader
    dataset = BEVDataset(
        bin_files=bin_files,
        label_files=label_files,
        config_path=args.config_path,
        classes_json_path=args.classes_json,
        img_size=args.img_size,
        augment=True,
        max_augmentations=args.max_augmentations
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_i, (imgs, targets, _) in enumerate(dataloader):
            # Move data to device
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
            
            # Print batch progress
            print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        # Print epoch progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s, Avg Loss: {epoch_loss/len(dataloader):.6f}")
        
        # Validation (if we have enough data)
        if len(bin_files) > 5 and (epoch + 1) % 10 == 0:
            model.eval()
            print("Running validation...")
            
            with torch.no_grad():
                for i, (imgs, targets, _) in enumerate(dataloader):
                    if i >= 5:  # Only validate on a few batches
                        break
                    
                    imgs = imgs.to(device)
                    
                    # Forward pass
                    outputs = model(imgs)
                    
                    # Process each scale output
                    processed_outputs = []
                    for i, output in enumerate(outputs):
                        # Reshape output
                        batch_size, num_anchors, grid_size, grid_size, box_attrs = output.shape
                        output_reshaped = output.view(batch_size, num_anchors * grid_size * grid_size, box_attrs)
                        
                        # Apply activations
                        output_reshaped[..., 0:2] = torch.sigmoid(output_reshaped[..., 0:2])  # x, y
                        output_reshaped[..., 4] = torch.sigmoid(output_reshaped[..., 4])  # confidence
                        output_reshaped[..., 5:] = torch.sigmoid(output_reshaped[..., 5:])  # class predictions
                        
                        processed_outputs.append(output_reshaped)
                    
                    # Process detections with NMS across all scales
                    detections = non_max_suppression(processed_outputs, conf_thres=0.5, nms_thres=0.4)
                    
                    # Count detections
                    detection_count = sum(1 for det in detections if det is not None)
                    print(f"Validation: {detection_count} detections found")
        elif (epoch + 1) % 10 == 0:
            print("Skipping validation - not enough data for meaningful validation")
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"yolo_bev_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "yolo_bev_final.pth"))
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/lidar3d_detection_ws/data/innoviz", help="Directory containing bin files")
    parser.add_argument("--label_dir", type=str, default="/lidar3d_detection_ws/data/labels", help="Directory containing label files")
    parser.add_argument("--config_path", type=str, default="../config/preprocessing_config.yaml", help="Path to config file")
    parser.add_argument("--classes_json", type=str, default="../../labels/_classes.json", help="Path to classes JSON file")
    parser.add_argument("--output_dir", type=str, default="../output", help="Directory to save models")
    parser.add_argument("--img_size", type=int, default=608, help="Size of input images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max_augmentations", type=int, default=50, help="Maximum number of augmentations per sample")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--calculate_anchors", action="store_true", help="Calculate anchors using K-means")
    
    args = parser.parse_args()
    train(args)