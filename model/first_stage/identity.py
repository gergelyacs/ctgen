import torch
from torch import nn

class Identity(nn.Module):
    def __init__(self, input_size: int):
        """
        Identity model that does not perform any transformation.
        Args:
            input_size (int): Size of the input.
        """
        super().__init__()
        self.latent_size = input_size

    def encode(self, x: torch.Tensor):
        return x

    def decode(self, z: torch.Tensor):
        return z
    
    def forward(self, x: torch.Tensor):
        return x
    
    def get_latent(self, x: torch.Tensor):
        return x
    
    