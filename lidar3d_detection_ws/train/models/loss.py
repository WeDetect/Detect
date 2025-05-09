import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between box1 and box2, with options for GIoU, DIoU, and CIoU
    
    Args:
        box1: First box
        box2: Second box
        x1y1x2y2: Whether boxes are in (x1, y1, x2, y2) format
        GIoU: Whether to calculate GIoU
        DIoU: Whether to calculate DIoU
        CIoU: Whether to calculate CIoU
        eps: Small constant to avoid division by zero
    
    Returns:
        IoU (or GIoU/DIoU/CIoU) between box1 and box2
    """
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
    union = b1_area + b2_area - inter_area + eps
    
    # IoU
    iou = inter_area / union
    
    if GIoU or DIoU or CIoU:
        # Enclosing box
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + eps  # Enclosing area
        
        if GIoU:
            # GIoU = IoU - (C - Union) / C
            giou = iou - (c_area - union) / c_area
            return giou
        
        if DIoU or CIoU:
            # Center distance
            c1_x = (b1_x1 + b1_x2) / 2
            c1_y = (b1_y1 + b1_y2) / 2
            c2_x = (b2_x1 + b2_x2) / 2
            c2_y = (b2_y1 + b2_y2) / 2
            center_dist = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)
            
            # Diagonal distance of enclosing box
            c_diag = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps
            
            if DIoU:
                # DIoU = IoU - center_distance^2 / diagonal_distance^2
                diou = iou - center_dist / c_diag
                return diou
            
            if CIoU:
                # Aspect ratio consistency
                w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
                w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
                v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v + eps)
                
                # CIoU = IoU - (center_distance^2 / diagonal_distance^2 + alpha * v)
                ciou = iou - (center_dist / c_diag + alpha * v)
                return ciou
    
    return iou

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Split anchors for each scale
        self.anchors_small = torch.tensor(anchors[0])
        self.anchors_medium = torch.tensor(anchors[1])
        self.anchors_large = torch.tensor(anchors[2])
        
        # Number of anchors per scale
        self.num_anchors = 3  # Each scale has 3 anchors
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        
        # Constants - adjusted for better balance
        self.lambda_coord = 5.0
        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5
        self.lambda_class = 1.0
        self.lambda_ciou = 1.0  # Weight for CIoU loss
        
    def forward(self, predictions, targets):
        """
        Calculate loss for YOLO predictions
        
        Args:
            predictions: List of predictions from each scale [small, medium, large]
            targets: Target boxes in format [batch_idx, class, x, y, w, h] or [batch_idx, x, y, w, h]
            
        Returns:
            Total loss
        """
        # Debug print to understand target format
        print(f"Target shape in loss: {targets.shape}")
        if len(targets) > 0:
            print(f"First target: {targets[0]}")
        
        # Determine if targets include class information
        has_class = targets.shape[1] > 5
        
        loss = 0
        
        # Process each scale
        for i, prediction in enumerate(predictions):
            # Get anchors for this scale
            if i == 0:  # Small objects
                scale_anchors = self.anchors_small
            elif i == 1:  # Medium objects
                scale_anchors = self.anchors_medium
            else:  # Large objects
                scale_anchors = self.anchors_large
            
            # Compute loss for this scale
            scale_loss = self.compute_scale_loss(prediction, targets, scale_anchors, has_class)
            
            # Add to total loss
            loss += scale_loss
        
        return loss
        
    def compute_scale_loss(self, prediction, targets, scale_anchors, has_class=True):
        """Compute loss for a single scale prediction"""
        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride = self.img_size / grid_size
        
        # Initialize losses
        loss_x = 0
        loss_y = 0
        loss_w = 0
        loss_h = 0
        loss_ciou = 0
        loss_obj = 0
        loss_noobj = 0
        loss_class = 0
        
        # Reshape prediction for easier access
        # [batch, num_anchors, grid, grid, 5+num_classes]
        pred_boxes = prediction[..., :4].clone()  # [x, y, w, h]
        pred_conf = prediction[..., 4]  # objectness
        pred_cls = prediction[..., 5:]  # class predictions
        
        # Create masks for objects and no objects
        obj_mask = torch.zeros_like(pred_conf, dtype=torch.bool)
        noobj_mask = torch.ones_like(pred_conf, dtype=torch.bool)
        
        # No targets, only no object loss
        if len(targets) == 0:
            loss_noobj = self.lambda_noobj * self.bce_loss(
                torch.sigmoid(pred_conf), 
                torch.zeros_like(pred_conf)
            )
            return loss_noobj / batch_size
        
        # Process targets
        batch_idx = targets[:, 0].long()
        if has_class:
            target_classes = targets[:, 1].long()
            target_boxes = targets[:, 2:6]  # [x, y, w, h]
        else:
            target_classes = torch.zeros(len(targets), dtype=torch.long, device=prediction.device)
            target_boxes = targets[:, 1:5]  # [x, y, w, h]
        
        # Convert target coordinates to grid coordinates
        gx = target_boxes[:, 0] * grid_size  # center x
        gy = target_boxes[:, 1] * grid_size  # center y
        gw = target_boxes[:, 2] * grid_size  # width
        gh = target_boxes[:, 3] * grid_size  # height
        
        # Get grid cell indices
        gi = gx.long()
        gj = gy.long()
        
        # Constrain to grid
        gi = torch.clamp(gi, 0, grid_size - 1)
        gj = torch.clamp(gj, 0, grid_size - 1)
        
        # Calculate IoU between targets and anchors
        # Convert anchors to grid scale
        scaled_anchors = scale_anchors.to(prediction.device) / stride
        
        # Calculate IoU between targets and anchors
        anchor_ious = []
        
        for anchor_idx in range(self.num_anchors):
            anchor_w = scaled_anchors[anchor_idx, 0]
            anchor_h = scaled_anchors[anchor_idx, 1]
            
            # Calculate IoU
            anchor_area = anchor_w * anchor_h
            target_area = gw * gh
            inter_w = torch.min(anchor_w, gw)
            inter_h = torch.min(anchor_h, gh)
            inter_area = inter_w * inter_h
            union_area = anchor_area + target_area - inter_area
            iou = inter_area / (union_area + 1e-16)
            
            anchor_ious.append(iou)
        
        anchor_ious = torch.stack(anchor_ious, dim=1)
        
        # Find best anchor for each target
        best_anchor_idx = torch.argmax(anchor_ious, dim=1)
        
        # Create ignore mask for anchors that have IoU > threshold with any target
        ignore_threshold = 0.5
        for batch_i in range(batch_size):
            for anchor_i in range(self.num_anchors):
                for grid_j in range(grid_size):
                    for grid_i in range(grid_size):
                        # Skip if this is the best anchor for a target
                        if obj_mask[batch_i, anchor_i, grid_j, grid_i]:
                            continue
                        
                        # Create anchor box
                        anchor_box = torch.zeros(4, device=prediction.device)
                        anchor_box[0] = (grid_i + 0.5) / grid_size  # center x
                        anchor_box[1] = (grid_j + 0.5) / grid_size  # center y
                        anchor_box[2] = scaled_anchors[anchor_i, 0] / grid_size  # width
                        anchor_box[3] = scaled_anchors[anchor_i, 1] / grid_size  # height
                        
                        # Calculate IoU with targets
                        batch_targets = targets[batch_idx == batch_i]
                        if len(batch_targets) > 0:
                            if has_class:
                                target_boxes_batch = batch_targets[:, 2:6]
                            else:
                                target_boxes_batch = batch_targets[:, 1:5]
                            
                            ious = bbox_iou(anchor_box.unsqueeze(0), target_boxes_batch, x1y1x2y2=False)
                            if ious.max() > ignore_threshold:
                                noobj_mask[batch_i, anchor_i, grid_j, grid_i] = 0
        
        # Set object mask and compute losses
        for i in range(len(targets)):
            # Skip if target is from a different batch
            if batch_idx[i] >= batch_size:
                continue
            
            # Set object mask
            obj_mask[batch_idx[i], best_anchor_idx[i], gj[i], gi[i]] = 1
            noobj_mask[batch_idx[i], best_anchor_idx[i], gj[i], gi[i]] = 0
            
            # Calculate cell-relative coordinates (tx, ty) for BCE loss
            tx = gx[i] - gi[i]  # x offset within cell (0-1)
            ty = gy[i] - gj[i]  # y offset within cell (0-1)
            
            # Coordinate losses using BCE for x,y (cell-relative)
            loss_x += self.lambda_coord * self.bce_loss(
                torch.sigmoid(pred_boxes[batch_idx[i], best_anchor_idx[i], gj[i], gi[i], 0]).unsqueeze(0), 
                tx.unsqueeze(0)
            )
            loss_y += self.lambda_coord * self.bce_loss(
                torch.sigmoid(pred_boxes[batch_idx[i], best_anchor_idx[i], gj[i], gi[i], 1]).unsqueeze(0), 
                ty.unsqueeze(0)
            )
            
            # Prepare predicted box for CIoU calculation
            pred_box = torch.zeros(1, 4, device=prediction.device)
            pred_box[0, 0] = (gi[i] + torch.sigmoid(pred_boxes[batch_idx[i], best_anchor_idx[i], gj[i], gi[i], 0])) * stride  # x center
            pred_box[0, 1] = (gj[i] + torch.sigmoid(pred_boxes[batch_idx[i], best_anchor_idx[i], gj[i], gi[i], 1])) * stride  # y center
            pred_box[0, 2] = torch.exp(pred_boxes[batch_idx[i], best_anchor_idx[i], gj[i], gi[i], 2]) * scaled_anchors[best_anchor_idx[i], 0] * stride  # width
            pred_box[0, 3] = torch.exp(pred_boxes[batch_idx[i], best_anchor_idx[i], gj[i], gi[i], 3]) * scaled_anchors[best_anchor_idx[i], 1] * stride  # height
            
            # Target box
            target_box = torch.zeros(1, 4, device=prediction.device)
            target_box[0, 0] = gx[i] * stride  # x center
            target_box[0, 1] = gy[i] * stride  # y center
            target_box[0, 2] = gw[i] * stride  # width
            target_box[0, 3] = gh[i] * stride  # height
            
            # Calculate CIoU
            ciou = bbox_iou(pred_box, target_box, x1y1x2y2=False, CIoU=True)
            loss_ciou += self.lambda_ciou * (1.0 - ciou)
            
            # Objectness loss - use IoU as target value for better convergence
            target_conf = torch.clamp(ciou.detach(), 0.0, 1.0)  # Use IoU as target confidence
            loss_obj += self.lambda_obj * self.bce_loss(
                torch.sigmoid(pred_conf[batch_idx[i], best_anchor_idx[i], gj[i], gi[i]]).unsqueeze(0), 
                target_conf
            )
            
            # Class loss
            if self.num_classes > 1:  # Skip if only one class
                target_cls = torch.zeros(self.num_classes, device=prediction.device)
                target_cls[target_classes[i]] = 1
                loss_class += self.lambda_class * self.bce_loss(
                    torch.sigmoid(pred_cls[batch_idx[i], best_anchor_idx[i], gj[i], gi[i]]), 
                    target_cls
                )
        
        # No object loss - only apply to anchors not ignored
        # Increase lambda_noobj for better false positive suppression
        self.lambda_noobj = 2.0  # Increased from 0.5 to 2.0
        if noobj_mask.sum() > 0:  # Only compute if there are any noobj elements
            loss_noobj += self.lambda_noobj * self.bce_loss(
                torch.sigmoid(pred_conf[noobj_mask]), 
                torch.zeros_like(pred_conf[noobj_mask])
            )
        
        # Total loss for this scale - prioritize CIoU loss
        scale_loss = loss_ciou + loss_obj + loss_noobj + loss_class + loss_x + loss_y
        
        # Normalize by number of targets
        num_targets = max(1, obj_mask.sum().item())
        scale_loss = scale_loss / num_targets
        
        return scale_loss