from torch import nn
import torch
from einops import reduce
from functools import partial
import torch.nn.functional as F
from utils import exists

class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResBlock(nn.Module):
    """
        Residual block with time conditioning.
    """
    def __init__(self, dim, dim_out, cond_emb_dim=None, norm_groups=32):
        super().__init__()

        if exists(cond_emb_dim):
            self.emb_func = nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(cond_emb_dim), dim_out * 2)
            ) 

        self.dim_out = dim_out
        self.cond_emb_dim = cond_emb_dim
        
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
     
    def forward(self, x, time_emb = None, cl_emb = None):
        if exists(self.cond_emb_dim) and (exists(time_emb) or exists(cl_emb)):
            cond_emb = tuple(filter(exists, (time_emb, cl_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            #cond_emb = self.emb_func(cond_emb)
            #cond_emb = rearrange(cond_emb, 'b c -> b c 1') is the same as:
            cond_emb = self.emb_func(cond_emb).view(x.shape[0], -1, 1)
            scale_shift = cond_emb.chunk(2, dim = 1)
        else :
            scale_shift = None

        y = self.block1(x, scale_shift = scale_shift)
        y = self.block2(y)
        return y + self.res_conv(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x