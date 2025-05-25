VERIFICATION_ORIGINAL_LIMIT = 20  # Number of original images to save for verification
VERIFICATION_AUGMENTATION_LIMIT = 5 # Number of augmented images to save for verification

# Augmentation specific
ROTATION_ANGLES = [-15, 15]
SCALE_DISTANCES = [-2.5, 2.5]
LATERAL_SHIFTS = [-2.5, 2.5]
HEIGHT_SHIFTS_CM = [-20, -10, -5, 5, 10, 20] # Shift in cm, will be converted to meters

FIXED_ZOOM_REGIONS = [
    {"x_min": 0, "x_max": 10, "y_min": -5, "y_max": 5, "name": "front_center"},
    {"x_min": 10, "x_max": 20, "y_min": -5, "y_max": 5, "name": "mid_center"}
]

# Dataset Configuration
NUM_CLASSES = 5
CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist', 'Bus', 'Truck']
TRAIN_VAL_SPLIT_RATIO = 0.8