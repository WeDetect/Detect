#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.train import train

def main():
    """
    Main entry point for training YOLO BEV model with augmentations
    """
    # Default paths - עדכון הנתיבים למיקומים הנכונים
    default_data_dir = os.path.join(os.path.dirname(current_dir), "data", "innoviz")
    default_label_dir = os.path.join(os.path.dirname(current_dir), "labels")
    default_config_path = os.path.join(current_dir, "config", "preprocessing_config.yaml")
    default_classes_json = os.path.join(os.path.dirname(current_dir), "labels", "_classes.json")
    default_output_dir = os.path.join(current_dir, "output")
    
    # Create parser
    parser = argparse.ArgumentParser(description="Train YOLO BEV model on LIDAR data with augmentations")
    
    # Add arguments with default values
    parser.add_argument("--data_dir", type=str, default=default_data_dir, 
                        help="Directory containing bin files")
    parser.add_argument("--label_dir", type=str, default=default_label_dir, 
                        help="Directory containing label files")
    parser.add_argument("--config_path", type=str, default=default_config_path, 
                        help="Path to config file")
    parser.add_argument("--classes_json", type=str, default=default_classes_json, 
                        help="Path to classes JSON file")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, 
                        help="Directory to save models")
    parser.add_argument("--img_size", type=int, default=608, 
                        help="Size of input images")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--max_augmentations", type=int, default=50, 
                        help="Maximum number of augmentations per sample")
    parser.add_argument("--save_interval", type=int, default=10, 
                        help="Save model every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU training")
    parser.add_argument("--evaluate_after_training", action="store_true", 
                        help="Run evaluation after training")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=== Training Configuration ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Label directory: {args.label_dir}")
    print(f"Config path: {args.config_path}")
    print(f"Classes JSON: {args.classes_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max augmentations: {args.max_augmentations}")
    print(f"Save interval: {args.save_interval}")
    print(f"Number of workers: {args.num_workers}")
    print(f"Force CPU: {args.cpu}")
    print(f"Evaluate after training: {args.evaluate_after_training}")
    print("============================")
    
    # Check if files exist
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        return
    
    if not os.path.exists(args.classes_json):
        print(f"Error: Classes JSON file not found at {args.classes_json}")
        return
    
    # Check if data directory contains bin files
    bin_files = list(Path(args.data_dir).glob("*.bin"))
    if len(bin_files) == 0:
        print(f"Error: No bin files found in {args.data_dir}")
        return
    
    print(f"Found {len(bin_files)} bin file(s) in {args.data_dir}")
    
    # Start training
    print("Starting training...")
    train(args)
    print("Training completed!")
    
    # Run evaluation if requested
    if args.evaluate_after_training:
        print("Running evaluation...")
        from models.evaluate import evaluate
        # Create evaluation args with the same parameters
        eval_args = argparse.Namespace(
            data_dir=args.data_dir,
            label_dir=args.label_dir,
            config_path=args.config_path,
            classes_json=args.classes_json,
            weights=os.path.join(args.output_dir, "yolo_bev_final.pth"),
            output_dir=os.path.join(args.output_dir, "eval"),
            img_size=args.img_size,
            conf_thres=0.5,
            nms_thres=0.4,
            display=False,
            cpu=args.cpu
        )
        os.makedirs(eval_args.output_dir, exist_ok=True)
        evaluate(eval_args)
        print("Evaluation completed!")

if __name__ == "__main__":
    main()