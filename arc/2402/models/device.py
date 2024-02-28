import torch


def get_device():
    """Get device (cuda, mps, or cpu)"""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "0"
    else:
        device = "cpu"
    return device
