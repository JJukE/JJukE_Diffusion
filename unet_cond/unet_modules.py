""" Modules for general conditional U-Net models
from https://github.com/CompVis/latent-diffusion/tree/main and
from https://github.com/Kitsunetic/DFusion/tree/master/dfusion/models/kitsunetic
"""
from abc import abstractmethod
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from jjuke.utils import default, zero_module, conv_nd, avg_pool_nd

from .attention import SpatialTransformer, AttentionBlock, checkpoint


class TimestepBlock(nn.Module):
    """ Any module where forward() takes timestep embeddings as a second arg """
    @abstractmethod
    def forward(self, x, emb):
        """ Apply the module to `x` given `emb` timestep embeddings. """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """ Create sinusoidal timeste embedding
    
    Args:
        timesteps: 1D tensor of N indices, one per batch element. (fractional)
        dim: Dimension of the output.
        max_period: Controls the minimum freq of the embeddings.
    
    Returns:
        (N, dim) tensor of positional embeddings
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)
                          * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
                          / half)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding
    


class Upsample(nn.Module):
    """ Upsampling layer with an optional convolution
    
    Args:
        channels: Channels in the inputs and outputs.
        use_conv: Determine if a convolution is applied.
        unet_dim: Determines if the signal is 1D, 2D, or 3D. If 3D, then
              upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, unet_dim=2, out_channels=None, padding=1):
        super().__init__()
        
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.use_conv = use_conv
        self.unet_dim = unet_dim
        if use_conv:
            self.conv = conv_nd(unet_dim, self.channels, self.out_channels, 3, padding=padding)
    
    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.unet_dim == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3]*2, x.shape[4]*2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    """ Learned 2x upsampling without padding """
    def __init__(self, channels, out_channels=None, kernel_size=5):
        super().__init__()
        
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        
        self.up = nn.ConvTranspose2d(self.channels, self.out_channels, kernel_size=kernel_size, stride=2)
    
    def forward(self, x):
        return self.up(x)


class Downsample(nn.Module):
    """ Downsampling layer with an optional convolution.
    
    Args:
        channels: Channels in the inputs and outputs.
        use_conv: Determine if a convolution is applied.
        unet_dim: Determines if the signal is 1D, 2D, or 3D. If 3D, then
              upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, unet_dim=2, out_channels=None, padding=1):
        super().__init__()
        
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.use_conv = use_conv
        self.unet_dim = unet_dim
        stride = 2 if unet_dim != 3 else (1, 2, 2)
        
        if use_conv:
            self.op = conv_nd(
                unet_dim, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(unet_dim, kernel_size=stride, stride=stride)
    
    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """ Residual block that can optionally change the number of channels
    
    Args:
        channels: The number of input channels.
        emb_channels: The number of timestep embedding channels.
        dropout: Dropout rate.
        out_channels: If specified, the number of out channels.
        use_conv: If True and out_channels is specified, use a spatial conv
                  instead of a smaller 1x1 conv to change the channels in
                  the skip connection.
        unet_dim: Determines if the signal is 1D, 2D, or 3D.
        use_checkpoint: If True, use gradient checkpointing on this module.
        sampling_type: If "up", use this block for upsampling, and if "down",
                       use this block for downsampling. ["up", "down"]
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        unet_dim=2,
        use_checkpoint=False,
        up=False,
        down=False,
        num_groups=32
    ):
        super().__init__()
        
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = default(out_channels, channels)
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=self.channels, eps=1e-6, affine=True),
            nn.SiLU(),
            conv_nd(unet_dim, channels, self.out_channels, 3, padding=1)
        )
        
        self.updown = up or down
        
        if up:
            self.h_upd = Upsample(channels, False, unet_dim)
            self.x_upd = Upsample(channels, False, unet_dim)
        elif down:
            self.h_upd = Downsample(channels, False, unet_dim)
            self.x_upd = Downsample(channels, False, unet_dim)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels)
        ) if emb_channels is not None else None
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=self.out_channels, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(conv_nd(unet_dim, self.out_channels, self.out_channels, 3, padding=1)) # for use
            # conv_nd(unet_dim, self.out_channels, self.out_channels, 3, padding=1) # for code test
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(unet_dim, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(unet_dim, channels, self.out_channels, 1)
    
    def forward(self, x, emb):
        """ Apply the block to a Tensor, conditioned on a timestep embedding
        
        Args:
            x: (n, c, ...) tensor of features.
            emb: (n, emb_channels) tensor of timestep embeddings
        
        Returns:
            (n, c, ...) tensor of outputs.
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, emb)
        else:
            return self._forward(x, emb)

    def _forward(self, x, emb=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        
        if self.emb_layers is not None:
            emb_out = self.emb_layers(emb) # .type(h.dtype)
            emb_out = rearrange(emb_out, "... -> ..." + " ".join(["()"] * (x.dim() - emb_out.dim()))) # unsqueeze as x
        
            # if self.use_scale_shift_norm:
            #     out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            #     scale, shift = torch.chunk(emb_out, 2, dim=1)
            #     h = out_norm(h) * (1 + scale) + shift
            #     h = out_rest(h)
            # else:
            h = h + emb_out
            h = self.out_layers(h)
        
        else:
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h
