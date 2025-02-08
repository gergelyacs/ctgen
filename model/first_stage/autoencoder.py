import torch
import torch.nn as nn
from model.layers import Attention, LinearAttention, ResBlock as ResidualBlock, Upsample, Downsample, PreNorm, Residual
from utils import calc_conv_out_size
import logging

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels: list[int] = None, input_size: int = 4096):
        """
        Encoder (based on UNet1D). 
        The final latent resolution will be: img_size / 2^{len(channels)}.
        Having fewer layers (2-4) with smaller latent_dim (2-4) tend to perserve higher frequency information.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]
        """
        super().__init__()
        
        channel_mults = channels
        num_mults = len(channel_mults)
   
        inner_channel = 32
        norm_groups = 8
        prev_channel = inner_channel
        logger.info (f"=== Encoder, init conv: in_channel: {in_channels} out_channel: {prev_channel}")
        self.init_conv = nn.Conv1d(in_channels, prev_channel, kernel_size=3, padding='same', padding_mode='reflect')
        out_sizes = [calc_conv_out_size(input_size, 3, 1, 1, 1)]
       
        self.downs = nn.ModuleList([])
        # Downsampling stage of U-net
        prev_channels = []
        for ind in range(num_mults):
            channel_mult = inner_channel * channel_mults[ind]
            logger.info (f"{ind}. downsampling layer: in_channel: {prev_channel}, out_channel: {channel_mult}")
       
            self.downs.append(nn.ModuleList([
                # Resnet block 1
                ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups),
                # Resnet block 2
                ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups),   
                # Attention
                Residual(PreNorm(prev_channel, LinearAttention(prev_channel))),
                # Downsample 
                Downsample(prev_channel, channel_mult) 
            ]))
            out_sizes.append(calc_conv_out_size(out_sizes[-1], 4, 2, 1, 1))
            prev_channels.append(prev_channel)
            prev_channel = channel_mult
        
        logger.info ("Mid-layer dimension: %s", prev_channel)
        # Resnet mid-block 1
        self.mid_block1 = ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups)
        # Mid-Attention
        self.mid_attn = Residual(PreNorm(prev_channel, Attention(prev_channel)))
        # Resnet mid-block 2
        self.mid_block2 = ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups)

        self.out = nn.Conv1d(prev_channel, latent_dim, kernel_size=3, padding='same')
        out_sizes.append(calc_conv_out_size(out_sizes[-1], 3, 1, 1, 1))

        #self.out = nn.Conv1d(prev_channel, 1, kernel_size=3, padding='same')
        #out_sizes.append(calc_conv_out_size(out_sizes[-1], 3, 1, 1, 1))

        #self.out = nn.Conv1d(prev_channel, latent_dim, kernel_size=latent_dim, stride=latent_dim)
        #out_sizes.append(calc_conv_out_size(out_sizes[-1], latent_dim, latent_dim, 1, 1))
        logger.info ("Output sizes: %s", out_sizes)
        self.latent_size = out_sizes[-1]
        self.latent_dim = latent_dim

        logger.info (f"Encoder output dimension for input size {input_size}: (batch_size, {latent_dim}, {self.latent_size})")

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)
   
        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x) 

        x = self.out(x)

        #x = rearrange(x, 'b 1 (k n)  -> b k n', k=self.latent_dim)
     
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int,
                 channels: list[int] = None, input_size: int = 4096):
        """
        Decoder converting
        a latent representation back to a time-series.

        Args:
            in_channels: Number of input channels of the image
            latent_dim: Number of channels for the latent space
            channels: List of channels for the number of down/up steps, eg [16, 32, 64]. Note
                that for the decoder the channels list will be reversed
        """
        super().__init__()
        
        channel_mults = channels
        self.latent_dim = latent_dim
        num_mults = len(channel_mults)

        inner_channel = 32
        norm_groups = 8

        prev_channel = channel_mults[-1] * inner_channel
        logger.info (f"=== Decoder, init conv: in_channel: {latent_dim} out_channel: {prev_channel}")
        #self.init_conv = nn.ConvTranspose1d(latent_dim, prev_channel, kernel_size=latent_dim, stride=latent_dim)
      
        self.init_conv = nn.Conv1d(latent_dim, prev_channel, kernel_size=3, padding='same')
        #self.init_conv = nn.Conv1d(1, prev_channel, kernel_size=3, padding='same')


        # Resnet mid-block 1
        self.mid_block1 = ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups)
        # Mid-Attention
        self.mid_attn = Residual(PreNorm(prev_channel, Attention(prev_channel)))
        # Resnet mid-block 2
        self.mid_block2 = ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups)

        self.ups = nn.ModuleList([])
        # Upsampling stage of U-net
        for i, ind in enumerate(reversed([0] + list(range(num_mults - 1)))):
            channel_mult = inner_channel * channel_mults[ind]
            logger.info (f"{i}. upsampling layer: in_channel: {prev_channel}, out_channel: {channel_mult}")
          
            self.ups.append(nn.ModuleList([
                # Resnet block 1
                ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups),
                # Resnet block 2
                ResidualBlock(prev_channel, prev_channel, norm_groups=norm_groups),
                # Attention
                Residual(PreNorm(prev_channel, LinearAttention(prev_channel))),            
                # Upsample
                Upsample(prev_channel, channel_mult)
            ]))
            prev_channel = channel_mult
        logger.info (f"Final conv layer: in_channel: {prev_channel}, out_channel: {in_channels}")
        self.out = nn.Conv1d(prev_channel, in_channels, 1)

    def forward(self, x: torch.Tensor):
        # bottleneck
        #x = rearrange(x, 'b k n -> b 1 (k n)', k=self.latent_dim)
        x = self.init_conv(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x) 

        for block1, block2, attn, upsample in self.ups:
            x = block1(x)
            x = block2(x)
            x = attn(x)
            x = upsample(x)

        # data is supposed to be scaled between -1 and 1 
        x = torch.tanh(self.out(x))
        #x = self.out(x)              

        return x

   