import numpy as np
import yaml
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def read_bin_file(bin_path):
    """Read point cloud from binary file"""
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points

def read_label_file(label_path):
    """Read KITTI format label file"""
    labels = []
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                labels.append({
                    'type': values[0],
                    'truncated': float(values[1]),
                    'occluded': int(values[2]),
                    'alpha': float(values[3]),
                    'bbox': [float(x) for x in values[4:8]],
                    'dimensions': [float(x) for x in values[8:11]],
                    'location': [float(x) for x in values[11:14]],
                    'rotation_y': float(values[14])
                })
    return labels
