import torch

def auto_select_device(device_arg):
    if device_arg == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Auto-selected device: {device}")
        return device
    return device_arg
