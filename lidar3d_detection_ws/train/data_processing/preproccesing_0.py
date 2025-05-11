import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import yaml

class PointCloudProcessor:
    """Class for processing point cloud data into bird's eye view (BEV) images"""
    
    def __init__(self, config_path=None, data_config_path=None):
        """
        Initialize the point cloud processor with configuration parameters
        
        Parameters:
        -----------
        config_path : str
            Path to preprocessing configuration YAML file
        data_config_path : str
            Path to data configuration YAML file
        """
        # Default values that will be overridden by config file
        self.bev_height = 608
        self.bev_width = 608
        self.resolution = 0.05
        self.side_range = (-15., 15.)
        self.fwd_range = (0., 30.)
        self.height_range = (-2., 2.)
        self.z_resolution = 0.2
        
        # Default colors for classes (BGR format)
        self.colors = {
            'Car': (0, 0, 255),      # Red
            'Pedestrian': (0, 255, 0),  # Green
            'Cyclist': (255, 0, 0),   # Blue
            'Truck': (255, 255, 0)    # Cyan
        }
        
        # Load configs if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            print("\n===== Configuration Parameters =====")
            
            # Get BEV dimensions from config
            if 'BEV_HEIGHT' in config and 'BEV_WIDTH' in config:
                self.bev_height = int(config['BEV_HEIGHT'])
                self.bev_width = int(config['BEV_WIDTH'])
                print(f"BEV image dimensions: {self.bev_width}x{self.bev_height}")
            
            # Get parameters from config    
            if 'DISCRETIZATION' in config:
                self.resolution = float(config['DISCRETIZATION'])
                print(f"Resolution: {self.resolution} m/pixel")
                
            if 'Z_RESOLUTION' in config:
                self.z_resolution = float(config['Z_RESOLUTION'])
                print(f"Z Resolution: {self.z_resolution} m")
                
            if 'boundary' in config:
                boundary = config['boundary']
                self.fwd_range = (float(boundary['minX']), float(boundary['maxX']))
                self.side_range = (float(boundary['minY']), float(boundary['maxY']))
                self.height_range = (float(boundary['minZ']), float(boundary['maxZ']))
                
                print(f"Forward range: {self.fwd_range[0]} to {self.fwd_range[1]} m")
                print(f"Side range: {self.side_range[0]} to {self.side_range[1]} m")
                print(f"Height range: {self.height_range[0]} to {self.height_range[1]} m")
                
            # Get colors if available
            if 'colors' in config:
                self.colors = config['colors']
                print("Colors loaded for classes:", list(self.colors.keys()))
                
            print("=====================================\n")
                
        # Load class names from data config
        self.class_names = ['Car', 'Pedestrian', 'Cyclist', 'Truck']
        if data_config_path and os.path.exists(data_config_path):
            with open(data_config_path, 'r') as f:
                data_config = yaml.safe_load(f)
                if 'names' in data_config:
                    self.class_names = data_config['names']
                    print("Classes loaded:", self.class_names)
        
        # Calculate image dimensions
        self.x_max = int((self.fwd_range[1] - self.fwd_range[0]) / self.resolution)
        self.y_max = int((self.side_range[1] - self.side_range[0]) / self.resolution)
        self.z_max = int((self.height_range[1] - self.height_range[0]) / self.z_resolution)
        
        # Class ID mapping based on class names
        self.class_map = {}
        for i, name in enumerate(self.class_names):
            self.class_map[name] = i
        
    def load_point_cloud(self, file_path):
        """
        Load point cloud data from binary file
        
        Parameters:
        -----------
        file_path : str
            Path to the point cloud binary file
            
        Returns:
        --------
        numpy.ndarray
            Array of points with shape (N, 4) - x, y, z, intensity
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
            
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_labels(self, label_path):
        """
        Load labels from file
        
        Parameters:
        -----------
        label_path : str
            Path to the label file
            
        Returns:
        --------
        list
            List of Object3d instances
        """
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
            
        with open(label_path, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
            
        objects = []
        for line in lines:
            parts = line.split(' ')
            if len(parts) < 15:  # Basic validation for label format
                continue
                
            obj = {
                'type': parts[0],
                'truncation': float(parts[1]),
                'occlusion': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                'rotation_y': float(parts[14])
            }
            objects.append(obj)
            
        return objects
    
    def create_bev_image(self, points):
        """
        Create a bird's eye view image from point cloud data
        
        Parameters:
        -----------
        points : numpy.ndarray
            Array of points (N, 4) with x, y, z, intensity values
            
        Returns:
        --------
        numpy.ndarray
            BEV image as a multi-channel array
        """
        # Initialize BEV image array
        bev_image = np.zeros((self.y_max + 1, self.x_max + 1, self.z_max + 2), dtype=np.float32)
        
        # Extract point coordinates
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        intensity = points[:, 3]
        
        # Filter points that are within the specified ranges
        f_filter = np.logical_and((x_points > self.fwd_range[0]), (x_points < self.fwd_range[1]))
        s_filter = np.logical_and((y_points > -self.side_range[1]), (y_points < -self.side_range[0]))
        filt = np.logical_and(f_filter, s_filter)
        z_filter = np.logical_and((z_points > self.height_range[0]), (z_points < self.height_range[1]))
        filt = np.logical_and(filt, z_filter)
        
        # Extract filtered points
        indices = np.where(filt)[0]
        x_filt = x_points[indices]
        y_filt = y_points[indices]
        z_filt = z_points[indices]
        intensity_filt = intensity[indices]
        
        # Convert coordinates to pixel positions
        x_img = (-y_filt / self.resolution).astype(np.int32)
        y_img = (-x_filt / self.resolution).astype(np.int32)
        
        # Shift coordinates to image space
        x_img -= int(np.floor(self.side_range[0] / self.resolution))
        y_img += int(np.floor(self.fwd_range[1] / self.resolution))
        
        # Height slices
        for i, height in enumerate(np.arange(self.height_range[0], self.height_range[1], self.z_resolution)):
            z_slice = np.logical_and((z_filt >= height), (z_filt < height + self.z_resolution))
            if np.any(z_slice):
                z_indices = np.where(z_slice)[0]
                bev_image[y_img[z_indices], x_img[z_indices], i] = 1
        
        # Add intensity channel
        bev_image[y_img, x_img, -1] = intensity_filt / 255.0
        
        # Create a colored height map for better visualization
        height_map = np.zeros((self.y_max + 1, self.x_max + 1, 3), dtype=np.uint8)
        
        # Assign different colors based on height
        for i in range(self.z_max):
            # Create a color gradient from blue (lower) to red (higher)
            b = max(0, 255 - (i * 255 // self.z_max))
            r = min(255, i * 255 // self.z_max)
            g = min(100, i * 100 // self.z_max)
            
            mask = bev_image[:, :, i] > 0
            height_map[mask, 0] = r
            height_map[mask, 1] = g
            height_map[mask, 2] = b
            
        # Use intensity for empty cells
        empty = np.sum(bev_image[:, :, :-1], axis=2) == 0
        intensity_map = (bev_image[:, :, -1] * 255).astype(np.uint8)
        height_map[empty, 0] = intensity_map[empty]
        height_map[empty, 1] = intensity_map[empty]
        height_map[empty, 2] = intensity_map[empty]
        
        return height_map
    
    # 3D bounding box to BEV image coordinates
    def transform_3d_box_to_bev(self, dimensions, location, rotation_y):
        """
        Transform a 3D bounding box to BEV image coordinates
        
        Parameters:
        -----------
        dimensions : list
            [height, width, length] of the 3D box
        location : list
            [x, y, z] location of the box center in camera coordinates
        rotation_y : float
            Rotation around the y-axis
            
        Returns:
        --------
        tuple
            (corners_bev, center_bev) where corners_bev is a list of corner coordinates
            and center_bev is the center point in BEV image coordinates
        """
        # Extract dimensions
        h, w, l = dimensions
        x, y, z = location
        
        # Calculate 3D box corners
        # 3D bounding box corners
        corners_3d = np.array([
            [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],
            [0, 0, 0, 0, -h, -h, -h, -h],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(rotation_y), 0, np.sin(rotation_y)],
            [0, 1, 0],
            [-np.sin(rotation_y), 0, np.cos(rotation_y)]
        ])
        
        # Apply rotation and translation
        corners_3d = np.dot(R, corners_3d)
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        
        # Get only the base of the box for BEV
        base_corners_3d = corners_3d[:, :4]
        
        # Transform to BEV image coordinates
        x_img = (-base_corners_3d[1, :] / self.resolution).astype(np.int32)
        y_img = (-base_corners_3d[0, :] / self.resolution).astype(np.int32)
        
        # Shift to image coordinates
        x_img -= int(np.floor(self.side_range[0] / self.resolution))
        y_img += int(np.floor(self.fwd_range[1] / self.resolution))
        
        # Center point in BEV
        center_x = (-y / self.resolution) - int(np.floor(self.side_range[0] / self.resolution))
        center_y = (-x / self.resolution) + int(np.floor(self.fwd_range[1] / self.resolution))
        
        corners_bev = list(zip(x_img, y_img))
        center_bev = (int(center_x), int(center_y))
        
        return corners_bev, center_bev
    
    # Draw a bounding box on the BEV image
    def draw_box_on_bev(self, bev_image, corners_bev, center_bev, obj_type):
        """
        Draw a bounding box on the BEV image
        
        Parameters:
        -----------
        bev_image : numpy.ndarray
            BEV image to draw on
        corners_bev : list
            List of corner coordinates [(x1,y1), (x2,y2), ...]
        center_bev : tuple
            (x, y) center point in BEV image coordinates
        obj_type : str
            Object type (e.g., 'Car', 'Pedestrian', etc.)
            
        Returns:
        --------
        numpy.ndarray
            BEV image with drawn bounding box
        """
        # Create a copy of the image to draw on
        bev_with_box = bev_image.copy()
        
        # Color mapping for different object types
        color_map = {
            'Car': (0, 0, 255),        # Red
            'Pedestrian': (0, 255, 0), # Green
            'Cyclist': (255, 0, 0),    # Blue
            'Van': (255, 0, 255),      # Magenta
            'Truck': (255, 255, 0),    # Cyan
            'Person': (0, 255, 255),   # Yellow
            'Tram': (128, 128, 255),   # Pink
            'Misc': (128, 128, 128)    # Gray
        }
        
        color = color_map.get(obj_type, (255, 255, 255))  # Default: white
        
        # Connect corners with lines
        for i in range(4):
            cv2.line(bev_with_box, 
                     corners_bev[i], 
                     corners_bev[(i+1)%4], 
                     color, 
                     2)
        
        # Draw center point
        cv2.circle(bev_with_box, center_bev, 3, color, -1)
        
        # Add label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bev_with_box, obj_type, 
                    (center_bev[0], center_bev[1] - 10), 
                    font, 0.5, color, 2)
        
        return bev_with_box
    
    # Create YOLO format label from BEV box corners
    def create_yolo_label(self, corners_bev, obj_type, img_shape):
        """
        Create YOLO format label from BEV box corners
        
        Parameters:
        -----------
        corners_bev : list
            List of corner coordinates [(x1,y1), (x2,y2), ...]
        obj_type : str
            Object type (e.g., 'Car', 'Pedestrian', etc.)
        img_shape : tuple
            (height, width) of the BEV image
            
        Returns:
        --------
        str
            YOLO format label string
        """
        # Class ID mapping
        class_id = self.class_map.get(obj_type, 0)
        
        # Extract corner coordinates
        x_coords = [x for x, y in corners_bev]
        y_coords = [y for x, y in corners_bev]
        
        # Calculate bounding box parameters
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalize by image dimensions
        img_height, img_width = img_shape
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        # Format as YOLO label
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def draw_filled_box_on_bev(self, bev_image, corners_bev, center_bev, obj_type):
        """
        Draw a filled bounding box on the BEV image
        
        Parameters:
        -----------
        bev_image : numpy.ndarray
            BEV image to draw on
        corners_bev : list
            List of corner coordinates [(x1,y1), (x2,y2), ...]
        center_bev : tuple
            (x, y) center point in BEV image coordinates
        obj_type : str
            Object type (e.g., 'Car', 'Pedestrian', etc.)
            
        Returns:
        --------
        numpy.ndarray
            BEV image with drawn filled bounding box
        """
        # Create a copy of the image to draw on
        bev_with_box = bev_image.copy()
        
        # Color mapping for different object types
        color_map = {
            'Car': (0, 0, 255),        # Red
            'Pedestrian': (0, 255, 0), # Green
            'Cyclist': (255, 0, 0),    # Blue
            'Van': (255, 0, 255),      # Magenta
            'Truck': (255, 255, 0),    # Cyan
            'Person': (0, 255, 255),   # Yellow
            'Tram': (128, 128, 255),   # Pink
            'Misc': (128, 128, 128)    # Gray
        }
        
        color = color_map.get(obj_type, (255, 255, 255))  # Default: white
        
        # Create a filled polygon with solid color (no transparency)
        points = np.array([corners_bev], dtype=np.int32)
        
        # Create a separate overlay image for the filled polygon
        overlay = bev_image.copy()
        cv2.fillPoly(overlay, points, color)
        
        # Blend the overlay with the original image (with custom alpha for better visibility)
        alpha = 0.6  # Higher alpha means more solid color (0.0-1.0)
        cv2.addWeighted(overlay, alpha, bev_with_box, 1 - alpha, 0, bev_with_box)
        
        # Optional: Add a small class indicator in the center without drawing a point
        # You can comment these lines if you want absolutely no center indication
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(obj_type[0], font, 0.4, 1)[0]
        text_x = center_bev[0] - text_size[0] // 2
        text_y = center_bev[1] + text_size[1] // 2
        cv2.putText(bev_with_box, obj_type[0], (text_x, text_y), font, 0.4, (255, 255, 255), 1)
        
        return bev_with_box

    def process_point_cloud(self, pc_file, label_file, output_img=None, output_label=None):
        """
        Process a point cloud file and corresponding labels
        
        Parameters:
        -----------
        pc_file : str
            Path to point cloud file
        label_file : str
            Path to label file
        output_img : str or None
            Path to save the output BEV image, if None, the image is not saved
        output_label : str or None
            Path to save the output YOLO labels, if None, the labels are not saved
            
        Returns:
        --------
        tuple
            (bev_image, yolo_labels) - the BEV image and list of YOLO labels
        """
        # Load point cloud data
        points = self.load_point_cloud(pc_file)
        
        # Load label data
        objects = self.load_labels(label_file)
        
        # Create BEV image
        bev_image = self.create_bev_image(points)
        
        # For visualization with all boxes
        bev_with_boxes = bev_image.copy()
        
        # Process objects and create YOLO labels
        yolo_labels = []
        
        for obj in objects:
            # Transform 3D box to BEV
            corners_bev, center_bev = self.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            
            # Draw box on the visualization image
            bev_with_boxes = self.draw_box_on_bev(
                bev_with_boxes, corners_bev, center_bev, obj['type']
            )
            
            # Create YOLO label
            yolo_label = self.create_yolo_label(
                corners_bev, obj['type'], bev_image.shape[:2]
            )
            yolo_labels.append(yolo_label)
        
        # Save outputs if paths are provided
        if output_img:
            cv2.imwrite(output_img, bev_with_boxes)
            print(f"BEV image saved to: {output_img}")
            
        if output_label:
            with open(output_label, 'w') as f:
                f.write('\n'.join(yolo_labels))
            print(f"YOLO labels saved to: {output_label}")
            
        return bev_with_boxes, yolo_labels

    def process_point_cloud_with_filled_boxes(self, pc_file, label_file, output_img=None, output_label=None):
        """
        Process a point cloud file and corresponding labels with filled boxes
        
        Parameters:
        -----------
        pc_file : str
            Path to point cloud file
        label_file : str
            Path to label file
        output_img : str or None
            Path to save the output BEV image, if None, the image is not saved
        output_label : str or None
            Path to save the output YOLO labels, if None, the labels are not saved
            
        Returns:
        --------
        tuple
            (bev_image, yolo_labels) - the BEV image and list of YOLO labels
        """
        # Load point cloud data
        points = self.load_point_cloud(pc_file)
        
        # Load label data
        objects = self.load_labels(label_file)
        
        # Create BEV image
        bev_image = self.create_bev_image(points)
        
        # For visualization with all boxes
        bev_with_boxes = bev_image.copy()
        
        # Process objects and create YOLO labels
        yolo_labels = []
        
        for obj in objects:
            # Transform 3D box to BEV
            corners_bev, center_bev = self.transform_3d_box_to_bev(
                obj['dimensions'], obj['location'], obj['rotation_y']
            )
            
            # Draw filled box on the visualization image
            bev_with_boxes = self.draw_filled_box_on_bev(
                bev_with_boxes, corners_bev, center_bev, obj['type']
            )
            
            # Create YOLO label
            yolo_label = self.create_yolo_label(
                corners_bev, obj['type'], bev_image.shape[:2]
            )
            yolo_labels.append(yolo_label)
        
        # Save outputs if paths are provided
        if output_img:
            cv2.imwrite(output_img, bev_with_boxes)
            print(f"BEV image saved to: {output_img}")
            
        if output_label:
            with open(output_label, 'w') as f:
                f.write('\n'.join(yolo_labels))
            print(f"YOLO labels saved to: {output_label}")
            
        return bev_with_boxes, yolo_labels

def main():
    # Example usage
    processor = PointCloudProcessor(
        config_path="/path/to/config.yaml",
        data_config_path="/path/to/data_config.yaml"
    )
    
    # Paths
    pc_file = "/lidar3d_detection_ws/data/innoviz/innoviz_00010.bin"
    label_file = "/lidar3d_detection_ws/data/labels/innoviz_00010.txt"
    
    # Create output directory if it doesn't exist
    output_dir = "/lidar3d_detection_ws/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output paths
    output_img = os.path.join(output_dir, "innoviz_00010_bev.png")
    output_label = os.path.join(output_dir, "innoviz_00010_yolo.txt")
    
    # Process the point cloud
    bev_image, yolo_labels = processor.process_point_cloud(
        pc_file, label_file, output_img, output_label
    )
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Bird's Eye View with Bounding Boxes")
    plt.show()

if __name__ == "__main__":
    main()
