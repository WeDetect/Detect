from utils.args_parser import parse_args
from utils.device_utils import auto_select_device
from utils.io_utils import create_output_dirs
from utils.augmentation_handler import handle_generate_augmented_dataset
from utils.training_handler import (
    handle_single_image_training,
    handle_train_all_data,
    handle_continue_training,
)


def main():
    args = parse_args()
    args.device = auto_select_device(args.device)
    output_dir, verification_dir = create_output_dirs(args.output_base)

    if args.generate_augmented_dataset:
        handle_generate_augmented_dataset(args, output_dir, verification_dir)
        return

    best_weights = None
    if args.single_image_test:
        best_weights = handle_single_image_training(args, output_dir)
    elif args.all_data_from_scratch:
        best_weights = handle_train_all_data(args, output_dir)
    elif args.continue_training:
        best_weights = handle_continue_training(args, output_dir)

    if best_weights:
        print(f"Training complete. Best weights saved to: {best_weights}")


if __name__ == "__main__":
    main()
