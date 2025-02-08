import torch
from torch import nn
from utils import downsample_time_series, upsample_time_series

class Sampling(nn.Module):
    def __init__(self, sampling_rate : int, input_size : int, shift=0):
        super().__init__()

        self.sampling_rate = sampling_rate     
        self.input_size = input_size  
        self.shift = shift
        self.latent_size = input_size // self.sampling_rate
       
    def encode(self, x: torch.Tensor):
        return downsample_time_series(x, self.sampling_rate, shift=self.shift)

    def decode(self, z: torch.Tensor):
        return torch.from_numpy(upsample_time_series(z, self.sampling_rate)).float()
    
    def forward(self, x: torch.Tensor):
        return (self.decode(self.encode(x)), ) # due to compatibility with other models
    
    def get_latent(self, x: torch.Tensor):
        return self.encode(x)
    
    