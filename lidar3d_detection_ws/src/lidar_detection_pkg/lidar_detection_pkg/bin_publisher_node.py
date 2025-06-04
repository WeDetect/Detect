#!/usr/bin/env python3
# bin_publisher_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import os
import time
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class BinPublisherNode(Node):
    def __init__(self):
        super().__init__('bin_publisher_node')
        
        # Declare parameters
        self.declare_parameter('bin_directory', '/lidar3d_detection_ws/data/kitti_velodyne_bins_normal')
        self.declare_parameter('label_directory', '/lidar3d_detection_ws/data/kitti_velodune_labels_normal')
        self.declare_parameter('output_topic', '/lidar_streamer_first_reflection')
        self.declare_parameter('markers_topic', '/ground_truth_objects')
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('loop', True)
        
        # Get parameters
        self.bin_directory = self.get_parameter('bin_directory').value
        self.label_directory = self.get_parameter('label_directory').value
        self.output_topic = self.get_parameter('output_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.loop = self.get_parameter('loop').value
        
        self.publisher = self.create_publisher(PointCloud2, self.output_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        
        self.class_colors = {
            'Car': ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            'Pedestrian': ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0),
            'Cyclist': ColorRGBA(r=0.6, g=0.0, b=0.8, a=1.0),
            'Bus': ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0),
            'Truck': ColorRGBA(r=0.0, g=0.8, b=0.0, a=1.0),
        }

        self.bin_files = sorted([f for f in os.listdir(self.bin_directory) if f.endswith('.bin')])
        self.current_bin_index = 0

        if not self.bin_files:
            self.get_logger().error(f"No .bin files found in {self.bin_directory}")
            return

        self.get_logger().info(f"Found {len(self.bin_files)} .bin files in {self.bin_directory}")
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_next_bin)

    def load_label_file(self, label_path):
        if not os.path.exists(label_path):
            return []

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 14:
                    continue

                try:
                    # קריאת הפורמט: Car 0 0 0 0 0 0 0 height width length x y z yaw
                    obj_type = parts[0]  # Car
                    h = float(parts[8])   # 1.57 (גובה)
                    w = float(parts[9])   # 1.65 (רוחב)  
                    l = float(parts[10])   # 3.35 (אורך)
                    x = float(parts[11])  # 5.20 (X - עומק קדימה)
                    y = float(parts[12])  # 4.43 (Y - צדדים)
                    z = float(parts[13])  # 1.65 (Z - גובה)
                    yaw = float(parts[14]) # -1.42 (YAW)

                    self.get_logger().info(f"Loaded {obj_type}: pos=({x:.2f},{y:.2f},{z:.2f}), dim=({h:.2f},{w:.2f},{l:.2f}), yaw={yaw:.2f}")
                    
                    label = {
                        'type': obj_type,
                        'dimensions': [h, w, l],  # [height, width, length]
                        'location': [x, y, z],    # [X=עומק, Y=צדדים, Z=גובה]
                        'rotation_y': yaw
                    }
                    labels.append(label)
                except ValueError as e:
                    self.get_logger().warn(f"Could not parse label line: {line}, error: {e}")
                    continue

        return labels

    def create_ground_truth_markers(self, labels, header):
        markers = MarkerArray()
        if not labels:
            return markers

        world_header = Header()
        world_header.stamp = header.stamp
        world_header.frame_id = self.frame_id

        for i, label in enumerate(labels):
            obj_type = label['type']
            if obj_type not in self.class_colors:
                continue

            x, y, z = label['location']  # x=עומק, y=צדדים, z=גובה
            h, w, l = label['dimensions']  # h=גובה, w=רוחב, l=אורך
            yaw = label['rotation_y']
            color = self.class_colors[obj_type]

            # תיבת 3D - Marker מסוג LINE_LIST
            marker = Marker()
            marker.header = world_header
            marker.ns = "ground_truth"
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.08
            marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()

            # חישוב פינות התיבה עם סיבוב נכון
            # מרכז התיבה צריך להיות בנקודה (x, y, z)
            half_l = l / 2.0  # חצי אורך
            half_w = w / 2.0  # חצי רוחב
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            # יצירת 8 פינות של התיבה
            corners = []
            # רשימת הפינות: קדמי/אחורי × שמאל/ימין × למטה/למעלה
            for dx in [-half_l, half_l]:  # אורך (קדימה/אחורה)
                for dy in [-half_w, half_w]:  # רוחב (שמאל/ימין)
                    for dz in [-h/2, h/2]:  # גובה (למטה/למעלה מהמרכז)
                        # סיבוב סביב ציר Z (yaw)
                        rx = dx * cos_yaw - dy * sin_yaw
                        ry = dx * sin_yaw + dy * cos_yaw
                        rz = dz
                        
                        # הוספת המיקום הגלובלי
                        corners.append(Point(x=x + rx, y=y + ry, z=z + rz))

            # חיבור פינות ליצירת תיבה
            # סדר הפינות: [back-left-bottom, back-right-bottom, front-left-bottom, front-right-bottom,
            #              back-left-top, back-right-top, front-left-top, front-right-top]
            edges = [
                # תחתית
                (0, 1), (1, 3), (3, 2), (2, 0),  
                # עליון
                (4, 5), (5, 7), (7, 6), (6, 4),  
                # חיבורים אנכיים
                (0, 4), (1, 5), (2, 6), (3, 7)   
            ]
            
            for start, end in edges:
                marker.points.append(corners[start])
                marker.points.append(corners[end])
                marker.colors.append(color)
                marker.colors.append(color)

            markers.markers.append(marker)

            # Marker של טקסט
            text_marker = Marker()
            text_marker.header = world_header
            text_marker.ns = "ground_truth_text"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = z + h/2 + 0.5  # מעל התיבה
            text_marker.scale.z = 1.0
            text_marker.text = f"{obj_type}"
            text_marker.color = color
            text_marker.color.a = 1.0
            text_marker.lifetime = rclpy.duration.Duration(seconds=2.0).to_msg()

            markers.markers.append(text_marker)

        return markers

    def publish_next_bin(self):
        if self.current_bin_index >= len(self.bin_files):
            if self.loop:
                self.current_bin_index = 0
                self.get_logger().info("Looping back to first bin file")
            else:
                self.get_logger().info("Finished publishing all bin files")
                self.timer.cancel()
                return

        bin_file = os.path.join(self.bin_directory, self.bin_files[self.current_bin_index])
        self.get_logger().info(f"Publishing bin file: {bin_file}")

        file_name = os.path.splitext(self.bin_files[self.current_bin_index])[0]
        label_file = os.path.join(self.label_directory, file_name + '.txt')

        try:
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
            ]

            cloud_msg = pc2.create_cloud(header, fields, points)
            self.publisher.publish(cloud_msg)
            self.get_logger().info(f"Published point cloud with {points.shape[0]} points")

            labels = self.load_label_file(label_file)
            if labels:
                markers = self.create_ground_truth_markers(labels, header)
                self.marker_pub.publish(markers)
                self.get_logger().info(f"Published {len(labels)} ground truth markers")
            else:
                empty_markers = MarkerArray()
                delete_marker = Marker()
                delete_marker.action = Marker.DELETEALL
                delete_marker.header = header
                empty_markers.markers.append(delete_marker)
                self.marker_pub.publish(empty_markers)
                self.get_logger().info("No labels found, published empty markers")

            self.current_bin_index += 1

        except Exception as e:
            self.get_logger().error(f"Error publishing bin file: {e}")
            self.current_bin_index += 1

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BinPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("🔴 Shutdown requested by user.")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        print("✅ ROS2 Node shut down successfully.")

if __name__ == '__main__':
    main()
