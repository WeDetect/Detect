# -*- coding: utf-8 -*-
import numpy as np
from od_detection import kitti_config as cnf
import cv2

def makeBEVMap(PointCloud_, boundary):
    Height = cnf.BEV_HEIGHT
    Width = cnf.BEV_WIDTH

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

    # Ensure indices are within bounds
    PointCloud[:, 0] = np.clip(PointCloud[:, 0], 0, Height - 1)
    PointCloud[:, 1] = np.clip(PointCloud[:, 1], 0, Width - 1)

    # Sort and filter points
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map & Density Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # Image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    valid_indices = (PointCloud_top[:, 0] < Height) & (PointCloud_top[:, 1] < Width)
    heightMap[np.int_(PointCloud_top[valid_indices, 0]), np.int_(PointCloud_top[valid_indices, 1])] = PointCloud_top[valid_indices, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[valid_indices, 0]), np.int_(PointCloud_top[valid_indices, 1])] = PointCloud_top[valid_indices, 3]
    densityMap[np.int_(PointCloud_top[valid_indices, 0]), np.int_(PointCloud_top[valid_indices, 1])] = normalizedCounts[valid_indices]

    RGB_Map = np.zeros((3, Height, Width))
    RGB_Map[2, :, :] = densityMap  # r_map
    RGB_Map[1, :, :] = heightMap  # g_map
    RGB_Map[0, :, :] = intensityMap  # b_map

    # Rotate the map by 180 degrees
    RGB_Map = np.rot90(RGB_Map, 2, axes=(1, 2))

    return RGB_Map

def get_filtered_lidar(pc_array, boundary):
    """
    Filter point cloud array based on boundary conditions
    This version ignores Z height filtering.
    """
    mask = (
        (pc_array[:, 0] >= boundary['minX']) & (pc_array[:, 0] <= boundary['maxX']) &
        (pc_array[:, 1] >= boundary['minY']) & (pc_array[:, 1] <= boundary['maxY'])
        # Removed Z filtering to allow all heights
    )
    filtered_pc = pc_array[mask]
    return filtered_pc

def makeBEVMapUpgrade(PointCloud_, boundary, point_colors=None):
    Height = cnf.BEV_HEIGHT
    Width = cnf.BEV_WIDTH

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)

    # Ensure indices are within bounds
    PointCloud[:, 0] = np.clip(PointCloud[:, 0], 0, Height - 1)
    PointCloud[:, 1] = np.clip(PointCloud[:, 1], 0, Width - 1)

    # Sort and filter points
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Distance Map, Intensity Map & Density Map
    distanceMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # Calculate distance from LiDAR
    distances = np.sqrt(PointCloud_top[:, 0]**2 + PointCloud_top[:, 1]**2)
    max_distance = np.max(distances)
    valid_indices = (PointCloud_top[:, 0] < Height) & (PointCloud_top[:, 1] < Width)
    distanceMap[np.int_(PointCloud_top[valid_indices, 0]), np.int_(PointCloud_top[valid_indices, 1])] = distances[valid_indices] / max_distance

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[valid_indices, 0]), np.int_(PointCloud_top[valid_indices, 1])] = PointCloud_top[valid_indices, 3]
    densityMap[np.int_(PointCloud_top[valid_indices, 0]), np.int_(PointCloud_top[valid_indices, 1])] = normalizedCounts[valid_indices]

    RGB_Map = np.zeros((3, Height, Width))
    RGB_Map[2, :, :] = densityMap  # r_map
    RGB_Map[1, :, :] = distanceMap  # g_map
    RGB_Map[0, :, :] = intensityMap  # b_map

    # הוספת הנקודות הלבנות במיקום המדויק
    if point_colors is not None:
        point_colors = point_colors[sorted_indices][unique_indices]
        detected_points = point_colors[valid_indices].sum(axis=1) > 0
        detected_x = np.int_(PointCloud_top[valid_indices, 0][detected_points])
        detected_y = np.int_(PointCloud_top[valid_indices, 1][detected_points])
        RGB_Map[:, detected_x, detected_y] = 1.0  # צביעה בלבן

    # Rotate the map by 180 degrees
    RGB_Map = np.rot90(RGB_Map, 2, axes=(1, 2))

    return RGB_Map

def makeHorizontalView(PointCloud_ , bounderis , point_colors=None):
    # Define the dimensions for the horizontal view
    Height = cnf.BEV_HEIGHT
    Width = cnf.BEV_WIDTH

    # Increase the discretization factor to reduce point density
    discretization_factor = 2

    # Initialize a dictionary to store the shortest distance for each azimuth-elevation pair
    azimuth_elevation_map = {}

    for point in PointCloud_:
        x, y, z = point[:3]
        
        # Calculate spherical coordinates
        distance = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)  # Angle to the side
        elevation = np.arctan2(z, np.sqrt(x**2 + y**2))  # Angle up/down

        # Discretize azimuth and elevation with increased factor
        azimuth_idx = int((azimuth + np.pi) / (2 * np.pi) * Width / discretization_factor)
        elevation_idx = int((elevation + np.pi / 2) / np.pi * Height / discretization_factor)

        # Update the map with the shortest distance
        if (azimuth_idx, elevation_idx) not in azimuth_elevation_map:
            azimuth_elevation_map[(azimuth_idx, elevation_idx)] = distance
        else:
            azimuth_elevation_map[(azimuth_idx, elevation_idx)] = min(
                azimuth_elevation_map[(azimuth_idx, elevation_idx)], distance
            )

    # Create the RGB map with black background
    RGB_Map = np.zeros((3, Height // discretization_factor, Width // discretization_factor))

    # Set selected points to white based on the closest distance
    for (azimuth_idx, elevation_idx), distance in azimuth_elevation_map.items():
        RGB_Map[:, elevation_idx, azimuth_idx] = 1  # Set to white

    return RGB_Map
