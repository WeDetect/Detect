from ultralytics import YOLO
import torch

# Load pretrained YOLOv5 model
model = YOLO("yolov5s.pt")  # You can change to yolov5m.pt or yolov5n.pt

# Update number of classes to your specific number
NUM_CLASSES = 4  # Car, Pedestrian, Cyclist, Truck (removed DontCare)
model.model.model[-1].nc = NUM_CLASSES
model.model.model[-1].detect = True

# Unfreeze all layers for better transfer learning
for name, param in model.model.named_parameters():
    param.requires_grad = True

# Print which parameters will be trained
print("Trainable parameters:")
for name, param in model.model.named_parameters():
    if param.requires_grad:
        print(f"- {name}")

# Train the model on your custom dataset
data_yaml_path = "config/data.yaml"  # Define your dataset YAML here
model.train(
    data=data_yaml_path, 
    epochs=50, 
    imgsz=608, 
    batch=8, 
    project="runs/train", 
    name="yolo5-bev-transfer",
    single_cls=False,
    rect=True)

# Save the final model
model.save("best_yolov5_transfer.pt")