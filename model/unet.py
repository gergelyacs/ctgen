"""
    U-net Architecture for multi-channel time-series data
    Time-series are 1D signals (as opposed to 2D images), therefore 1D convolutional layers are used instead of 2D.
    The input tensor has shape (batch_size, in_channel, ts_len), where ts_len is the length of the time-series,
    in_channel is the number of time_series (2 for CTG) plus the number of channels in the condition tensor (for super-resolution).
    out_channel is always the number of time-series (2 for CTG) to be generated.
"""

import torch

from model.layers import LinearAttention, PositionalEncoding, Attention, ResBlock, Upsample, Downsample, Residual, PreNorm
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class UNet1D(nn.Module):
    def __init__(self, in_channel, out_channel, inner_channel=32, norm_groups=32,
        channel_mults=[1, 2, 4, 8, 8]): 
        super().__init__()
        self.out_channel = out_channel

        cond_channel = inner_channel
        self.time_emb = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            nn.SiLU(), 
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
   
        prev_channel = inner_channel
        logger.info ("=====> Building U-net model with inner channel: %s", inner_channel)
        logger.info (f"init conv: in_channel: {in_channel} out_channel: {prev_channel}")
        self.init_conv = nn.Conv1d(in_channel, prev_channel, kernel_size=3, padding='same', padding_mode='reflect')
       
        self.downs = nn.ModuleList([])
        # Downsampling stage of U-net
        prev_channels = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            logger.info (f"{ind}. downsampling layer: in_channel: {prev_channel}, out_channel: {channel_mult}")
       
            self.downs.append(nn.ModuleList([
                # Resnet block 1
                ResBlock(prev_channel, prev_channel, cond_emb_dim=cond_channel, 
                    norm_groups=norm_groups),
                # Resnet block 2
                ResBlock(prev_channel, prev_channel, cond_emb_dim=cond_channel, 
                    norm_groups=norm_groups),   
                # Attention
                Residual(PreNorm(prev_channel, LinearAttention(prev_channel))),
                # Downsample 
                Downsample(prev_channel, channel_mult) if not is_last else nn.Conv1d(prev_channel, channel_mult, 3, padding = 'same')
            ]))
            prev_channels.append(prev_channel)
            prev_channel = channel_mult
        
        logger.info ("Mid-layer dimension: %s", prev_channel)
        # Resnet mid-block 1
        self.mid_block1 = ResBlock(prev_channel, prev_channel, cond_emb_dim=cond_channel, 
                            norm_groups=norm_groups)
        # Mid-Attention
        self.mid_attn = Residual(PreNorm(prev_channel, Attention(prev_channel)))

        # Resnet mid-block 2
        self.mid_block2 = ResBlock(prev_channel, prev_channel, cond_emb_dim=cond_channel, 
                        norm_groups=norm_groups)

        self.ups = nn.ModuleList([])
        # Upsampling stage of U-net
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
          
            logger.info (f"{ind}. upsampling layer: in_channel: {channel_mult}+{prev_channels[ind]}, out_channel: {channel_mult}")
          
            self.ups.append(nn.ModuleList([
                # Resnet block 1
                ResBlock(
                    channel_mult + prev_channels[ind], channel_mult, cond_emb_dim=cond_channel, 
                    norm_groups=norm_groups),
                # Resnet block 2
                ResBlock(
                    channel_mult + prev_channels[ind], channel_mult, cond_emb_dim=cond_channel, 
                    norm_groups=norm_groups),
                # Attention
                Residual(PreNorm(channel_mult, LinearAttention(channel_mult))),            
                # Upsample
                Upsample(channel_mult, prev_channels[ind]) if not is_last else nn.Conv1d(channel_mult, prev_channels[ind], 3, padding = 'same')
            ]))
        # Accomodate the output of the first conv to the final layer! (prev_channels[ind] = inner_channel at this point)
        logger.info (f"Penultimate residual layer: in_channel: {inner_channel}+{inner_channel}, out_channel: {inner_channel}")
        self.final_res_block = ResBlock(2 * inner_channel, inner_channel, cond_emb_dim=cond_channel)
      
        logger.info (f"Final conv layer: in_channel: {inner_channel}, out_channel: {out_channel}")
        self.final_conv = nn.Conv1d(inner_channel, out_channel, 1)

    def forward(self, x, time, **kwargs):
        # Embedding of time
        t = self.time_emb(time)
        
        x = self.init_conv(x)
        # input is added to last layer
        r = x.clone()

        feats = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            feats.append(x)
            
            x = block2(x, t)

            x = attn(x)
            feats.append(x)
            
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)   

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, feats.pop()), dim = 1)
            x = block1(x, t)
         
            x = torch.cat((x, feats.pop()), dim = 1)
            x = block2(x, t)

            x = attn(x)

            x = upsample(x)
       
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
       
        return self.final_conv(x)
