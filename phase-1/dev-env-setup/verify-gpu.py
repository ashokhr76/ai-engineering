import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())


import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using:", device)
print(torch.backends.mps.is_macos13_or_newer())
print(torch.backends.mps.get_core_count())
x = torch.randn(3,3).to(device)
print(x)
