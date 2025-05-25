#function to preprocess the point cloud from innoviz to velodyne format before feeding it to the SFA3D network
def preprocess_pointcloud(self, pc_array):
    """Transform Innoviz point cloud to match Velodyne format"""
    if pc_array.shape[0] == 0:
        self.get_logger().warn("Received empty point cloud!")
        return np.zeros((0, 4), dtype=np.float32)
    
    processed_pc = pc_array.copy()
    print(f"\nInitial point cloud statistics:")
    print(f"Points: {processed_pc.shape[0]}")
    print(f"X range: [{processed_pc[:,0].min():.2f}, {processed_pc[:,0].max():.2f}]")
    print(f"Y range: [{processed_pc[:,1].min():.2f}, {processed_pc[:,1].max():.2f}]")
    print(f"Z range: [{processed_pc[:,2].min():.2f}, {processed_pc[:,2].max():.2f}]")
    
    # 1. Apply transformation from innoviz to velodyne
    t = np.array([-5.5, 0, 8])
    pitch = 0.3
    R = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Apply transformation
    points = processed_pc[:, :3]
    points = (R @ points.T).T  # Apply rotation first
    points = points - t  # Then translation
    
    # Adjust Z coordinate to match KITTI range
    z_offset = abs(points[:, 2].min()) + 2  # Ensure minimum Z is above -2.73
    points[:, 2] += z_offset
    
    processed_pc[:, :3] = points
    
    print(f"\nAfter transformation and Z adjustment:")
    print(f"X range: [{processed_pc[:,0].min():.2f}, {processed_pc[:,0].max():.2f}]")
    print(f"Y range: [{processed_pc[:,1].min():.2f}, {processed_pc[:,1].max():.2f}]")
    print(f"Z range: [{processed_pc[:,2].min():.2f}, {processed_pc[:,2].max():.2f}]")
    
    # 2. Ensure XYZR format
    if processed_pc.shape[1] != 4:
        intensity = np.ones((processed_pc.shape[0], 1)) * 0.5
        processed_pc = np.hstack((processed_pc[:, :3], intensity))
    
    # 3. Apply KITTI-like range filtering
    mask = (processed_pc[:, 0] >= 0) & (processed_pc[:, 0] <= 50) & \
           (processed_pc[:, 1] >= -25) & (processed_pc[:, 1] <= 25) & \
           (processed_pc[:, 2] >= -2.73) & (processed_pc[:, 2] <= 1.27)
    processed_pc = processed_pc[mask]
    
    if processed_pc.shape[0] > 0:
        print(f"\nAfter filtering:")
        print(f"Remaining points: {processed_pc.shape[0]}")
        print(f"X range: [{processed_pc[:,0].min():.2f}, {processed_pc[:,0].max():.2f}]")
        print(f"Y range: [{processed_pc[:,1].min():.2f}, {processed_pc[:,1].max():.2f}]")
        print(f"Z range: [{processed_pc[:,2].min():.2f}, {processed_pc[:,2].max():.2f}]")
    
    return processed_pc










/// ros2 run tf2_ros static_transform_publisher -5.5 0 8 0 0.3 0 velodyne innoviz
