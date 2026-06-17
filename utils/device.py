import torch

#DEVICE = torch.device("cpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

print(f"Using device: {DEVICE}")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print("\nDetected GPUs:")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  GPU {i}: {props.name} "
            f"({props.total_memory / 1024**3:.2f} GB)"
        )

    current_gpu = (
        DEVICE.index
        if isinstance(DEVICE, torch.device)
        and DEVICE.type == "cuda"
        and DEVICE.index is not None
        else torch.cuda.current_device()
    )

    print(f"\nCode will use GPU {current_gpu}: {torch.cuda.get_device_name(current_gpu)}")
else:
    print("No GPU detected")

torch.set_num_threads(4)

def get_device():
    return DEVICE