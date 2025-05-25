import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model on BEV images')
    
    # Dataset generation options
    parser.add_argument('--bin_dir', type=str, default='lidar3d_detection_ws/data/innoviz/', 
                        help='Directory containing bin files')
    parser.add_argument('--label_dir', type=str, default='lidar3d_detection_ws/data/labels/', 
                        help='Directory containing label files')
    parser.add_argument('--config_path', type=str, default='config/preprocessing_config.yaml', 
                        help='Path to preprocessing config file')
    parser.add_argument('--output_base', type=str, default='train', 
                        help='Base directory for output')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, 
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640, 
                        help='Image size')
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device to use (cuda:0, cpu, or auto)')
    
    # Model options
    parser.add_argument('--transfer_learning', action='store_true', 
                        help='Use transfer learning')
    parser.add_argument('--unfreeze_layers', type=int, default=10, 
                        help='Number of layers to unfreeze for transfer learning')
    parser.add_argument('--continue_from', type=str, default='', 
                        help='Path to model weights to continue training from')
    
    # Augmentation options
    parser.add_argument('--augmentations', action='store_true', 
                        help='Enable data augmentation')
    parser.add_argument('--augmentation_factor', type=int, default=2, 
                        help='Augmentation factor (multiplier for dataset size)')
    
    # Single image test options
    parser.add_argument('--single_image_test', action='store_true', 
                        help='Train on a single image for testing')
    parser.add_argument('--bin_file', type=str, default='', 
                        help='Path to specific bin file for single image test')
    parser.add_argument('--label_file', type=str, default='', 
                        help='Path to specific label file for single image test')
    parser.add_argument('--train_from_scratch', action='store_true',
                        help='Train from scratch on a single image')
    
    # All data training option
    parser.add_argument('--all_data_from_scratch', action='store_true',
                        help='Train from scratch on all data with detailed visualization')
    
    # New option for generating augmented dataset
    parser.add_argument('--generate_augmented_dataset', action='store_true',
                        help='Generate augmented dataset without training')
    
    # New option for continuing training from a checkpoint
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from a checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='/lidar3d_detection_ws/train/output/best.pt',
                        help='Path to checkpoint for continuing training')

    args = parser.parse_args()
    return args