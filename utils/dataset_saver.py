import os
import cv2
import random
from tqdm import tqdm

def save_augmented_dataset(augmented_data, output_dir):
    augmented_train_img_dir = os.path.join(output_dir, 'augmented_dataset', 'train', 'images')
    augmented_train_label_dir = os.path.join(output_dir, 'augmented_dataset', 'train', 'labels')
    augmented_val_img_dir = os.path.join(output_dir, 'augmented_dataset', 'val', 'images')
    augmented_val_label_dir = os.path.join(output_dir, 'augmented_dataset', 'val', 'labels')

    for dir_path in [augmented_train_img_dir, augmented_train_label_dir,
                     augmented_val_img_dir, augmented_val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)

    random.shuffle(augmented_data)
    split = int(0.8 * len(augmented_data))
    train_data, val_data = augmented_data[:split], augmented_data[split:]

    print(f"Training set: {len(train_data)} files")
    print(f"Validation set: {len(val_data)} files")

    for i, item in enumerate(tqdm(train_data, desc="Saving training data")):
        cv2.imwrite(os.path.join(augmented_train_img_dir, f"train_{i}.png"), item['bev_image'])
        with open(os.path.join(augmented_train_label_dir, f"train_{i}.txt"), 'w') as f:
            f.writelines(label + '\n' for label in item['yolo_labels'])

    for i, item in enumerate(tqdm(val_data, desc="Saving validation data")):
        cv2.imwrite(os.path.join(augmented_val_img_dir, f"val_{i}.png"), item['bev_image'])
        with open(os.path.join(augmented_val_label_dir, f"val_{i}.txt"), 'w') as f:
            f.writelines(label + '\n' for label in item['yolo_labels'])

    print(f"Augmented dataset saved to {os.path.join(output_dir, 'augmented_dataset')}")
