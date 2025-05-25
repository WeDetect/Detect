import os

def create_output_dirs(base_output):
    output_dir = os.path.join(base_output, 'output')
    verification_dir = os.path.join(output_dir, 'label_verification')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    return output_dir, verification_dir
