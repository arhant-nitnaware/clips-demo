import torch

DEVICE = torch.device("cpu")

torch.set_num_threads(4)

def get_device():
    return DEVICE