import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ConvBlock, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """Residual block with dropout for regularization"""
    def __init__(self, channels, dropout_rate=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out += residual
        return out

class DownsampleBlock(nn.Module):
    """Downsample block: Conv with stride 2 + Conv"""
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=2)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class SPPBlock(nn.Module):
    """Spatial Pyramid Pooling block"""
    def __init__(self, in_channels):
        super(SPPBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(in_channels * 2, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        
        # Apply different max pooling operations
        p1 = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        p2 = F.max_pool2d(x, kernel_size=9, stride=1, padding=4)
        p3 = F.max_pool2d(x, kernel_size=13, stride=1, padding=6)
        
        # Concatenate pooling results
        cat = torch.cat([x, p1, p2, p3], dim=1)
        
        # Final 1x1 conv
        out = self.conv2(cat)
        return out

class UpsampleBlock(nn.Module):
    """Upsample block with interpolation + Conv"""
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class DetectionHead(nn.Module):
    """Detection head for YOLO"""
    def __init__(self, in_channels, num_anchors, num_classes):
        super(DetectionHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Final layer to predict boxes, objectness and class probabilities
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), kernel_size=1)
        
    def forward(self, x):
        # Need to reshape the output to the proper format
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # Apply conv and reshape
        pred = self.conv(x)
        pred = pred.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        return pred

class YOLOBEV(nn.Module):
    """YOLO model for Bird's Eye View object detection"""
    def __init__(self, num_classes=4, img_size=608, dropout_rate=0.1):
        super(YOLOBEV, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout_rate = dropout_rate
        
        # Define anchors for different scales - MODIFIED FOR BEV LIDAR DATA
        # These should be calculated using K-means on your dataset
        # Format is (width, height) in pixels relative to 608x608 image
        self.anchors = [
            # Small objects (76x76 grid)
            [(8, 8), (12, 12), (16, 16)],
            # Medium objects (38x38 grid)
            [(20, 20), (28, 28), (36, 36)],
            # Large objects (19x19 grid)
            [(45, 45), (60, 60), (80, 80)]
        ]
        
        # Input layer - handle grayscale input
        self.conv1 = ConvBlock(1, 32, kernel_size=3)
        
        # Backbone - Darknet-like
        self.down1 = DownsampleBlock(32, 64)
        self.res1 = ResBlock(64, dropout_rate)
        
        self.down2 = DownsampleBlock(64, 128)
        self.res2_1 = ResBlock(128, dropout_rate)
        self.res2_2 = ResBlock(128, dropout_rate)
        
        self.down3 = DownsampleBlock(128, 256)
        self.res3_1 = ResBlock(256, dropout_rate)
        self.res3_2 = ResBlock(256, dropout_rate)
        self.res3_3 = ResBlock(256, dropout_rate)
        self.res3_4 = ResBlock(256, dropout_rate)
        self.res3_5 = ResBlock(256, dropout_rate)
        self.res3_6 = ResBlock(256, dropout_rate)
        self.res3_7 = ResBlock(256, dropout_rate)
        self.res3_8 = ResBlock(256, dropout_rate)
        
        # Save small object features
        self.conv_s = ConvBlock(256, 128, kernel_size=1, padding=0)
        
        self.down4 = DownsampleBlock(256, 512)
        self.res4_1 = ResBlock(512, dropout_rate)
        self.res4_2 = ResBlock(512, dropout_rate)
        self.res4_3 = ResBlock(512, dropout_rate)
        self.res4_4 = ResBlock(512, dropout_rate)
        self.res4_5 = ResBlock(512, dropout_rate)
        self.res4_6 = ResBlock(512, dropout_rate)
        self.res4_7 = ResBlock(512, dropout_rate)
        self.res4_8 = ResBlock(512, dropout_rate)
        
        # Save medium object features
        self.conv_m = ConvBlock(512, 256, kernel_size=1, padding=0)
        
        self.down5 = DownsampleBlock(512, 1024)
        self.res5_1 = ResBlock(1024, dropout_rate)
        self.res5_2 = ResBlock(1024, dropout_rate)
        self.res5_3 = ResBlock(1024, dropout_rate)
        self.res5_4 = ResBlock(1024, dropout_rate)
        
        # SPP block
        self.spp = SPPBlock(1024)
        
        # Neck - Feature fusion
        self.conv_neck1 = ConvBlock(1024, 512, kernel_size=1, padding=0)
        self.conv_neck2 = ConvBlock(512, 1024, kernel_size=3)
        
        # Large object branch
        self.conv_large = ConvBlock(1024, 512, kernel_size=1, padding=0)
        self.head_large = DetectionHead(512, len(self.anchors[2]), num_classes)
        
        # Upsample for medium objects
        self.up_m = UpsampleBlock(512, 256)
        self.conv_m_fusion = ConvBlock(512, 256)
        self.conv_medium = ConvBlock(256, 256, kernel_size=1, padding=0)
        self.head_medium = DetectionHead(256, len(self.anchors[1]), num_classes)
        
        # Upsample for small objects
        self.up_s = UpsampleBlock(256, 128)
        self.conv_s_fusion = ConvBlock(256, 128)
        self.conv_small = ConvBlock(128, 128, kernel_size=1, padding=0)
        self.head_small = DetectionHead(128, len(self.anchors[0]), num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # Handle both 3-channel and 1-channel inputs
        if x.size(1) == 3:
            # Convert RGB to grayscale if needed
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Backbone
        x = self.conv1(x)
        x = self.down1(x)
        x = self.res1(x)
        
        x = self.down2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        
        x = self.down3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res3_3(x)
        x = self.res3_4(x)
        x = self.res3_5(x)
        x = self.res3_6(x)
        x = self.res3_7(x)
        x = self.res3_8(x)
        
        # Save small object features
        small_features = self.conv_s(x)
        
        x = self.down4(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.res4_4(x)
        x = self.res4_5(x)
        x = self.res4_6(x)
        x = self.res4_7(x)
        x = self.res4_8(x)
        
        # Save medium object features
        medium_features = self.conv_m(x)
        
        x = self.down5(x)
        x = self.res5_1(x)
        x = self.res5_2(x)
        x = self.res5_3(x)
        x = self.res5_4(x)
        
        # Apply dropout during training
        if self.training:
            x = self.dropout(x)
        
        # SPP block
        x = self.spp(x)
        
        # Neck - Feature fusion
        x = self.conv_neck1(x)
        x = self.conv_neck2(x)
        
        # Large object detection
        large_branch = self.conv_large(x)
        large_output = self.head_large(large_branch)
        
        # Upsample for medium objects
        x_up_m = self.up_m(large_branch)
        x_m = torch.cat([x_up_m, medium_features], dim=1)
        x_m = self.conv_m_fusion(x_m)
        
        # Apply dropout during training
        if self.training:
            x_m = self.dropout(x_m)
        
        # Medium object detection
        medium_branch = self.conv_medium(x_m)
        medium_output = self.head_medium(medium_branch)
        
        # Upsample for small objects
        x_up_s = self.up_s(medium_branch)
        x_s = torch.cat([x_up_s, small_features], dim=1)
        x_s = self.conv_s_fusion(x_s)
        
        # Apply dropout during training
        if self.training:
            x_s = self.dropout(x_s)
        
        # Small object detection
        small_branch = self.conv_small(x_s)
        small_output = self.head_small(small_branch)
        
        # Return outputs from all three scales
        return [small_output, medium_output, large_output]

    def load_darknet_weights(self, weights_path):
        """
        Load pre-trained weights from Darknet format
        This is a placeholder - implementation would depend on the specific weight format
        """
        # This would need to be implemented based on the specific weight format
        pass

    def process_detections(self, detections, img_size):
        """
        Process raw detections to get bounding boxes and class predictions
        """
        processed_detections = []
        
        for i, detection in enumerate(detections):
            # Get grid size
            grid_size = detection.size(2)
            
            # Get stride (how many pixels each grid cell represents)
            stride = img_size / grid_size
            
            # Get anchor set for this scale
            if i == 0:  # Small objects
                anchors = torch.tensor(self.anchors[0])
            elif i == 1:  # Medium objects
                anchors = torch.tensor(self.anchors[1])
            else:  # Large objects
                anchors = torch.tensor(self.anchors[2])
            
            # Scale anchors to current grid size
            anchors = anchors / stride
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
            
            if detection.is_cuda:
                grid_x = grid_x.cuda()
                grid_y = grid_y.cuda()
                anchors = anchors.cuda()
            
            # Reshape for broadcasting
            anchors = anchors.view(1, -1, 1, 1, 2)
            grid_x = grid_x.view(1, 1, grid_size, grid_size)
            grid_y = grid_y.view(1, 1, grid_size, grid_size)
            
            # Apply sigmoid to x, y predictions and add grid offsets
            detection[..., 0:2] = torch.sigmoid(detection[..., 0:2])
            detection[..., 0] += grid_x
            detection[..., 1] += grid_y
            
            # Apply exponential to width, height predictions and multiply by anchors
            detection[..., 2:4] = torch.exp(detection[..., 2:4]) * anchors
            
            # Scale x, y, w, h by stride to get pixel coordinates
            detection[..., :4] *= stride
            
            # Apply sigmoid only during inference for confidence and class predictions
            detection[..., 4:] = torch.sigmoid(detection[..., 4:])
            
            # Reshape to [batch, num_anchors*grid_size*grid_size, 5+num_classes]
            batch_size = detection.size(0)
            detection = detection.view(batch_size, -1, 5 + self.num_classes)
            
            processed_detections.append(detection)
        
        return processed_detections

if __name__ == "__main__":
    # Test the model
    model = YOLOBEV(num_classes=5, img_size=608)
    x = torch.randn(1, 1, 608, 608)  # Single channel input
    outputs = model(x)
    
    # Print output shapes
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")