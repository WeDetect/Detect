import os
import sys
import argparse
import torch
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import glob

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_processing.preprocessing import read_bin_file, read_label_file, create_bev_image, load_config, draw_labels_on_bev
from models.yolo_bev import YOLOBEV
from models.loss import bbox_iou  # Import bbox_iou from loss.py

def bbox_iou(box1, box2, x1y1x2y2=True):
    """Calculate IoU between boxes"""
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
    
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
    return iou

def non_max_suppression(predictions, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    
    Args:
        predictions: List of outputs from model, each with shape (B, num_anchors*grid_size*grid_size, 5+num_classes)
        conf_thres: Confidence threshold
        nms_thres: NMS threshold
        
    Returns:
        List of detections with each detection in the format: [x1, y1, x2, y2, conf, cls_conf, cls_pred]
    """
    # Process all scale outputs and combine them
    batch_detections = [None]  # Initialize with None for each batch
    
    try:
        for prediction in predictions:
            print(f"Processing prediction with shape: {prediction.shape}")
            
            # From (center x, center y, width, height) to (x1, y1, x2, y2)
            box_corner = prediction.new(prediction.shape)
            box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
            box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
            box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
            box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
            prediction[:, :, :4] = box_corner[:, :, :4]
            
            # Process each batch
            for image_i, image_pred in enumerate(prediction):
                print(f"  Processing batch {image_i}, image_pred shape: {image_pred.shape}")
                
                # Filter out confidence scores below threshold
                conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
                image_pred = image_pred[conf_mask]
                
                print(f"  After confidence filtering: {image_pred.shape[0]} detections")
                
                # If none remain, process next image
                if not image_pred.size(0):
                    continue
                    
                # Get score and class with highest confidence
                class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
                
                # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
                detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
                
                print(f"  Before NMS: {detections.shape[0]} detections")
                
                # Perform non-maximum suppression
                try:
                    # Simple approach: sort by confidence and keep top boxes
                    _, conf_sort_index = torch.sort(detections[:, 4], descending=True)
                    detections = detections[conf_sort_index]
                    
                    # Keep track of which detections to keep
                    keep = torch.ones(detections.size(0)).type(torch.bool)
                    
                    # Go through each detection and compare with others
                    for i in range(detections.size(0)):
                        if keep[i]:
                            # Get IoU with all remaining detections
                            ious = bbox_iou(detections[i:i+1, :4], detections[i+1:, :4])
                            
                            # Find detections with same class and IoU > threshold
                            same_class = detections[i, -1] == detections[i+1:, -1]
                            remove = same_class & (ious > nms_thres)
                            
                            # Update keep mask
                            keep[i+1:][remove] = 0
                    
                    # Keep only selected detections
                    detections = detections[keep]
                    
                    print(f"  After NMS: {detections.shape[0]} detections")
                    
                    # Update batch detections
                    if batch_detections[image_i] is None:
                        batch_detections[image_i] = detections
                    else:
                        batch_detections[image_i] = torch.cat((batch_detections[image_i], detections))
                
                except Exception as e:
                    print(f"  Error in NMS: {e}")
                    # If NMS fails, just return the detections sorted by confidence
                    if batch_detections[image_i] is None:
                        batch_detections[image_i] = detections[:10]  # Limit to top 10 to avoid issues
                    else:
                        batch_detections[image_i] = torch.cat((batch_detections[image_i], detections[:10]))
        
        return batch_detections
        
    except Exception as e:
        print(f"Error in non_max_suppression: {e}")
        return [None]

def evaluate(args):
    """Evaluate YOLO BEV model on test data"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Define ANSI color codes for terminal output
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    # Load class names
    with open(args.classes_json, 'r') as f:
        classes_data = json.load(f)
        print(f"Classes data: {classes_data}")
        class_names = [c['name'] for c in classes_data['classes']]
        class_colors = {c['name']: tuple(int(c['color'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) 
                        for c in classes_data['classes']}
        # Create a mapping from class ID to name for debugging
        class_id_to_name = {c['id']: c['name'] for c in classes_data['classes']}
        print(f"Class ID to name mapping: {class_id_to_name}")
    
    # Initialize model
    model = YOLOBEV(num_classes=len(class_names), img_size=args.img_size)
    
    # Load model weights
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Warning: Weights file not found at {args.weights}")
        return
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config = load_config(args.config_path)
    
    # Find all bin files
    bin_files = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
    if not bin_files:
        print(f"Error: No bin files found in {args.data_dir}")
        return
    
    # Process each bin file
    for bin_file in bin_files:
        bin_file = Path(bin_file)
        print(f"\nProcessing {bin_file.name}...")
        
        # Find corresponding label file
        label_file = os.path.join(args.label_dir, f"{bin_file.stem}.txt")
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {bin_file.name}")
            continue
        
        # Load point cloud and labels
        points = read_bin_file(bin_file)
        gt_labels = read_label_file(label_file)
        
        # Print ground truth labels
        print(f"\n{GREEN}Ground Truth Labels:{RESET}")
        for i, label in enumerate(gt_labels):
            if label['type'] == 'DontCare':
                continue
                
            x = label['location'][0]  # x
            y = label['location'][1]  # y
            
            # Convert to BEV pixel coordinates
            x_bev = int(x / config['DISCRETIZATION'])
            y_bev = int(y / config['DISCRETIZATION'] + config['BEV_WIDTH'] / 2)
            
            print(f"{GREEN}GT #{i+1}: Class={label['type']}, Position=({x_bev}, {y_bev}), 3D Location=({x:.2f}, {y:.2f}, {label['location'][2]:.2f}){RESET}")
        
        # Create BEV image using the same function as in preprocessing.py
        bev_image, white_dots_positions = create_bev_image(points, config, gt_labels)
        
        # Create a copy for the original image with ground truth
        bev_original = bev_image.copy()
        
        # Draw ground truth boxes on original image using draw_labels_on_bev from preprocessing.py
        bev_original = draw_labels_on_bev(bev_original, gt_labels, config, white_dots_positions)
        
        # Create a copy for model predictions
        bev_model = bev_image.copy()
        
        # Resize if needed for the model
        model_input = bev_image.copy()
        if model_input.shape[0] != args.img_size or model_input.shape[1] != args.img_size:
            model_input = cv2.resize(model_input, (args.img_size, args.img_size))

        # Convert to grayscale (simulate YOLO behavior)
        model_input_gray = cv2.cvtColor(model_input, cv2.COLOR_RGB2GRAY)
        model_input_gray = np.expand_dims(model_input_gray, axis=2)  # HWC → HWC with 1 channel

        # Convert to tensor - make sure we're using the same preprocessing as in training
        img = torch.from_numpy(model_input_gray.transpose(2, 0, 1)).float() / 255.0
        img = img.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img)
            
            # Process each scale output
            processed_outputs = []
            for i, output in enumerate(outputs):
                # Get grid size
                batch_size, num_anchors, grid_size, grid_size, box_attrs = output.shape
                
                # Get stride (how many pixels each grid cell represents)
                stride = args.img_size / grid_size
                
                # Get anchor set for this scale
                if i == 0:  # Small objects
                    anchors = torch.tensor(model.anchors[0], device=device)
                elif i == 1:  # Medium objects
                    anchors = torch.tensor(model.anchors[1], device=device)
                else:  # Large objects
                    anchors = torch.tensor(model.anchors[2], device=device)
                
                # Scale anchors to current grid size
                anchors = anchors / stride
                
                # Create grid
                grid_y, grid_x = torch.meshgrid(torch.arange(grid_size, device=device), 
                                                torch.arange(grid_size, device=device), 
                                                indexing='ij')
                
                # Reshape for broadcasting
                anchors = anchors.view(1, -1, 1, 1, 2)
                grid_x = grid_x.view(1, 1, grid_size, grid_size)
                grid_y = grid_y.view(1, 1, grid_size, grid_size)
                
                # Apply sigmoid to x, y predictions and add grid offsets
                output[..., 0:2] = torch.sigmoid(output[..., 0:2])
                output[..., 0] += grid_x
                output[..., 1] += grid_y
                
                # Apply exponential to width, height predictions and multiply by anchors
                output[..., 2:4] = torch.exp(output[..., 2:4]) * anchors
                
                # Scale x, y, w, h by stride to get pixel coordinates
                output[..., :4] *= stride
                
                # Apply sigmoid to objectness and class predictions
                output[..., 4:] = torch.sigmoid(output[..., 4:])
                
                # Reshape to [batch, num_anchors*grid_size*grid_size, 5+num_classes]
                output_reshaped = output.view(batch_size, -1, box_attrs)
                
                # Debug raw outputs
                print(f"\nProcessed model output stats for scale {i}:")
                print(f"  Shape: {output_reshaped.shape}")
                print(f"  Confidence range: {output_reshaped[..., 4].min().item():.4f} to {output_reshaped[..., 4].max().item():.4f}")
                print(f"  Class probs range: {output_reshaped[..., 5:].min().item():.4f} to {output_reshaped[..., 5:].max().item():.4f}")
                
                processed_outputs.append(output_reshaped)
            
            # Process detections with NMS across all scales
            detections = non_max_suppression(processed_outputs, conf_thres=args.conf_thres, nms_thres=args.nms_thres)
            if detections[0] is not None:
                print(f"\n{YELLOW}Model Predictions:{RESET}")
                print(f"Found {len(detections[0])} detections with confidence >= {args.conf_thres}")
                
                # Calculate IoU between ground truth and predictions
                print(f"\n{GREEN}IoU Calculations:{RESET}")
                for i, gt in enumerate(gt_labels):
                    if gt['type'] == 'DontCare':
                        continue
                        
                    # Convert GT to format expected by bbox_iou
                    x_bev = gt['location'][0] / config['DISCRETIZATION']
                    y_bev = gt['location'][1] / config['DISCRETIZATION'] + config['BEV_WIDTH'] / 2
                    w_bev = gt['dimensions'][1] / config['DISCRETIZATION']  # width in BEV
                    h_bev = gt['dimensions'][2] / config['DISCRETIZATION']  # length in BEV
                    
                    gt_box = torch.tensor([[x_bev, y_bev, w_bev, h_bev]], device=device)
                    
                    for j, pred in enumerate(detections[0]):
                        # Convert prediction from [x1, y1, x2, y2] to [x_center, y_center, width, height]
                        x1, y1, x2, y2 = pred[0].item(), pred[1].item(), pred[2].item(), pred[3].item()
                        pred_center_x = (x1 + x2) / 2
                        pred_center_y = (y1 + y2) / 2
                        pred_width = x2 - x1
                        pred_height = y2 - y1
                        
                        # Scale back if needed
                        if args.img_size != bev_model.shape[0]:
                            scale_factor = bev_model.shape[0] / args.img_size
                            pred_center_x *= scale_factor
                            pred_center_y *= scale_factor
                            pred_width *= scale_factor
                            pred_height *= scale_factor
                        
                        pred_box = torch.tensor([[pred_center_x, pred_center_y, pred_width, pred_height]], device=device)
                        
                        # Calculate IoU
                        iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                        print(f"GT #{i+1} ({gt['type']}) - Pred #{j+1} IoU: {iou.item():.4f}")
                
                # Limit number of displayed detections
                max_display = min(10, len(detections[0]))
                
                # Create a copy for visualization
                bev_with_detections = bev_model.copy()
                
                try:
                    # Draw detections on the image
                    for i, detection in enumerate(detections[0][:max_display]):
                        # Extract detection info
                        x1, y1, x2, y2, conf, cls_conf, cls_pred = detection.cpu().numpy()
                        
                        # Convert back to original image size if needed
                        if args.img_size != bev_with_detections.shape[0]:
                            scale_factor = bev_with_detections.shape[0] / args.img_size
                            x1 *= scale_factor
                            y1 *= scale_factor
                            x2 *= scale_factor
                            y2 *= scale_factor
                        
                        # Get class name and color
                        cls_id = int(cls_pred.item())
                        cls_name = class_id_to_name.get(cls_id, f"Unknown-{cls_id}")
                        color = class_colors.get(cls_name, (255, 255, 255))
                        
                        # Print detection info
                        print(f"{YELLOW}Detection #{i+1}: Class={cls_name}, Confidence={conf:.4f}, Class Confidence={cls_conf:.4f}, Position=({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f}){RESET}")
                        
                        # Draw bounding box
                        cv2.rectangle(bev_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{cls_name}: {conf:.2f}"
                        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
                        cv2.rectangle(bev_with_detections, (int(x1), int(y1) - t_size[1] - 3), c2, color, -1)
                        cv2.putText(bev_with_detections, label, (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    print("Successfully drew detections on the image")
                except Exception as e:
                    print(f"Error drawing detections: {e}")
                    # Continue without drawing
            else:
                print(f"{YELLOW}No detections found with confidence >= {args.conf_thres}{RESET}")
                bev_with_detections = bev_model.copy()
            
            # Save images
            try:
                output_base = os.path.join(args.output_dir, bin_file.stem)
                
                # Save original BEV with ground truth
                cv2.imwrite(f"{output_base}_gt.png", bev_original)
                print(f"Saved ground truth image to {output_base}_gt.png")
                
                # Save BEV with model detections
                cv2.imwrite(f"{output_base}_pred.png", bev_with_detections)
                print(f"Saved prediction image to {output_base}_pred.png")
                
                # Save combined visualization
                combined = np.hstack((bev_original, bev_with_detections))
                cv2.imwrite(f"{output_base}_combined.png", combined)
                print(f"Saved combined image to {output_base}_combined.png")
            except Exception as e:
                print(f"Error saving images: {e}")
        
        # Display only if requested
        if args.display:
            # Show original image
            cv2.imshow("Original Points and Ground Truth", bev_original)
            print("Showing original image. Press any key to see model predictions...")
            cv2.waitKey(0)
            cv2.destroyWindow("Original Points and Ground Truth")
            
            # Show model predictions
            cv2.imshow("Model Predictions", bev_with_detections)
            print("Showing model predictions. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyWindow("Model Predictions")
    
    cv2.destroyAllWindows()
    print("Evaluation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data", help="Directory containing bin files")
    parser.add_argument("--label_dir", type=str, default="../../labels", help="Directory containing label files")
    parser.add_argument("--config_path", type=str, default="../config/preprocessing_config.yaml", help="Path to config file")
    parser.add_argument("--classes_json", type=str, default="../../labels/_classes.json", help="Path to classes JSON file")
    parser.add_argument("--weights", type=str, default="../output/yolo_bev_final.pth", help="Path to model weights")
    parser.add_argument("--output_dir", type=str, default="../output/eval", help="Directory to save evaluation results")
    parser.add_argument("--img_size", type=int, default=608, help="Size of input images")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--display", action="store_true", help="Display detection results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    
    args = parser.parse_args()
    evaluate(args)