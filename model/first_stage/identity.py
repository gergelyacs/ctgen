import torch
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x: torch.Tensor):
        return x

    def decode(self, z: torch.Tensor):
        return z
    
    def forward(self, x: torch.Tensor):
        return x
    
    def get_latent(self, x: torch.Tensor):
        return x
    
    