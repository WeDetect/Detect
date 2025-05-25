from ultralytics import YOLO
import numpy as np

def create_full_trainable_model(pretrained_model="yolov5s.pt", num_classes=4):
    """
    Create a YOLO model with all layers unfrozen for training from scratch
    
    Args:
        pretrained_model: Path to pretrained model or model name
        num_classes: Number of classes to detect
        
    Returns:
        YOLO model with all layers unfrozen
    """
    # Load pretrained YOLOv5 model
    model = YOLO(pretrained_model)
    
    # Update number of classes to your specific number
    model.model.model[-1].nc = num_classes
    model.model.model[-1].detect = True
    
    # Unfreeze all layers for full training
    for name, param in model.model.named_parameters():
        param.requires_grad = True
        
    # Log trainable parameters
    print("Trainable parameters (all layers):")
    trainable_params = 0
    total_params = 0
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"- {name}")
        total_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} of {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model

def create_transfer_learning_model(pretrained_model="yolov5s.pt", num_classes=4, unfreeze_layers=20):
    """
    Create a YOLO model with only the last N layers unfrozen for transfer learning
    
    Args:
        pretrained_model: Path to pretrained model or model name
        num_classes: Number of classes to detect
        unfreeze_layers: Number of layers to unfreeze from the end
        
    Returns:
        YOLO model with specified layers unfrozen
    """
    # Load pretrained YOLOv5 model
    model = YOLO(pretrained_model)
    
    # Update number of classes to your specific number
    model.model.model[-1].nc = num_classes
    model.model.model[-1].detect = True
    
    # First freeze all layers
    for name, param in model.model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze only the last N layers
    layers_to_unfreeze = min(unfreeze_layers, len(model.model.model))
    print(f"Unfreezing last {layers_to_unfreeze} layers:")
    
    for i in range(layers_to_unfreeze):
        layer_idx = -1 - i  # Start from the last layer and move backwards
        if abs(layer_idx) <= len(model.model.model):
            for name, param in model.model.model[layer_idx].named_parameters():
                param.requires_grad = True
                print(f"- Unfreezing {name} in layer {layer_idx}")
    
    # Log trainable parameters
    trainable_params = 0
    total_params = 0
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        total_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} of {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model

if __name__ == "__main__":
    # Test both models
    print("Creating fully trainable model...")
    full_model = create_full_trainable_model()
    
    print("\nCreating transfer learning model...")
    transfer_model = create_transfer_learning_model(unfreeze_layers=20)
    
    # Note: YOLO handles the conversion to grayscale or appropriate format internally
    # The model will work with our 608x608x3 input images
    print("\nBoth models are ready for training")