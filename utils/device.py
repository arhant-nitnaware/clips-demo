import torch

#DEVICE = torch.device("cpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

torch.set_num_threads(4)

def get_device():
    return DEVICE