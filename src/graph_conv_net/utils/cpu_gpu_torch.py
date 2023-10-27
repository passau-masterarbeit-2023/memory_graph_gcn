import torch

# -------------------- GPU -------------------- # 
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

def print_device_info():
    print(f"🔦 Is CUDA (for Torch) available? {torch.cuda.is_available()}")
    print(f"🔦 Device: {DEVICE} (WARN: Due to CUDA out of memory errors, running on CPU only!)")