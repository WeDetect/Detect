# -*- coding: utf-8 -*-
import numpy as np

# Define KITTI boundary settings
boundary = {
    'minX': -5,
    'maxX': 50,
    'minY': -50,
    'maxY': 50,
    'minZ': -10,
    'maxZ': 10
}

# BEV dimensions
BEV_HEIGHT = 512
BEV_WIDTH = 512
DISCRETIZATION = 0.1  # Meters
CLASS_NAME_TO_ID = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2
}
