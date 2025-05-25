import os
from lidar3d_detection_ws.train.models.train import generate_augmented_dataset
# from lidar3d_detection_ws.train.data_processing.generate_augmented_dataset import generate_augmented_dataset
from utils.dataset_saver import save_augmented_dataset

def handle_generate_augmented_dataset(args, output_dir, verification_dir):
    print("Generating augmented dataset...")
    if args.bin_file and args.label_file:
        bin_files = [args.bin_file]
        augmented_data = generate_augmented_dataset(
            bin_file=args.bin_file,
            label_file=args.label_file,
            config_path=args.config_path,
            output_dir=os.path.join(output_dir, 'augmented_dataset'),
            verification_dir=verification_dir
        )
    else:
        augmented_data = generate_augmented_dataset(
            bin_dir=args.bin_dir,
            label_dir=args.label_dir,
            config_path=args.config_path,
            output_dir=os.path.join(output_dir, 'augmented_dataset'),
            verification_dir=verification_dir
        )
    save_augmented_dataset(augmented_data, output_dir)
