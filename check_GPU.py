import torch
print("CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())