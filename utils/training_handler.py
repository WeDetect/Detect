# from lidar3d_detection_ws.train.models.train import train_from_checkpoint, train_on_single_image_from_scratch
from lidar3d_detection_ws.train.train_from_checkpoint import train_from_checkpoint
from lidar3d_detection_ws.train.train_on_all_from_scratch import train_on_all_data_from_scratch


def handle_single_image_training(args, output_dir):
    if not args.bin_file or not args.label_file:
        print("Error: Must provide bin_file and label_file for single image test")
        return
    if args.train_from_scratch:
        return train_on_single_image_from_scratch(
            bin_file=args.bin_file,
            label_file=args.label_file,
            config_path=args.config_path,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device
        )

def handle_train_all_data(args, output_dir):
    return train_on_all_data_from_scratch(
        bin_dir=args.bin_dir,
        label_dir=args.label_dir,
        config_path=args.config_path,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        augmentations=args.augmentations,
        augmentation_factor=args.augmentation_factor
    )

def handle_continue_training(args, output_dir):
    return train_from_checkpoint(
        bin_dir=args.bin_dir,
        label_dir=args.label_dir,
        config_path=args.config_path,
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        augmentations=args.augmentations,
        augmentation_factor=args.augmentation_factor
    )
