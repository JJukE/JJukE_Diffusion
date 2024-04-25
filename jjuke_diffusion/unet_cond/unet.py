from functools import partial

import torch
from torch import nn
from jjuke.utils import conv_nd, zero_module

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .unet_modules import ResBlock, AttentionBlock, SpatialTransformer, \
    TimestepEmbedSequential, Downsample, Upsample, timestep_embedding


class UNetModel(nn.Module):
    """ General UNet model with attention and timestep embedding
    
    Args:
        unet_dim: Determines if the signal is 1D, 2D or 3D.
        in_channels: Channels in the input tensor.
        out_channels: Channels in the output tensor.
        model_channels: Base channel count for the model.
        num_res_blocks: Number of residual blocks per downsampling.
        attention_resolutions: Collection of downsample rates at which attention will take place.
                               May be a set, list, or tuple. For example, if it contains 4,
                               then at 4x downsampling, attention will be used.
        dropout: Dropout probability.
        channel_mult: Channel multiplier for each level of the Unet.
        conv_resample: If True, use learned convolutions for upsampling and downsampling.
        num_heads: Number of attention heads in each attention layer.
        dim_head: If specified, ignore num_heads and instead use a fixed channel width per attention head.
        resblock_updown: Use residual blocks for upsampling and downsampling.
        attention_type: Type of attention.
        
        Conditioning
        attention_fn: Determine what function to use for the attention. It should be "spatial_transformer"
                      if using cross-attention conditioning.
        context_dim: Dimension of cross-attention conditioning
        transformer_depth: Depth of the transformer
        num_classes: If specified, then this model will be class-conditional with `num_classes` classes.
        
        
        
    """
    def __init__(
        self,
        unet_dim=2,
        in_channels=3,
        out_channels=3,
        model_channels=64,
        num_res_blocks=2, # depth of the model
        # spatial_res=128, # spatial resolution (h, w of feature maps)
        attention_resolutions=[2], # downsampling factor = (spatial resolution (h, w of feature maps) / cross-attention resolution(s))
        dropout=0.,
        channel_mult=[1, 2, 4, 8],
        conv_resample=True,
        # use_checkpoint=False, # use gradient checkpointing to reduce memory usage
        num_heads=-1,
        dim_head=-1,
        resblock_updown=False,
        attention_type="qkv_legacy", # ["qkv", "qkv_legacy", "xformers", "memory_efficient_attention"]
        attention_fn="attention_block", # ["attention_block", "spatial_transformer", "no_attention"]
        context_dim=None,
        transformer_depth=1,
        num_classes=None,
        # n_embed=None, # custom support for prediction of discrete ids into codebook of first stage vq model
        no_time_emb=False,
        is_autoencoder=False,
        num_groups=32
    ):
        super().__init__()
        
        if attention_fn == "spatial_transformer":
            assert context_dim is not None, "The dimension of cross-attention conditioning should be included."
        
        if context_dim is not None:
            assert attention_fn == "spatial_transformer", "Spatial transformer should be used for cross-attention conditioning."
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)
        
        if num_heads == -1 and dim_head == -1:
            raise ValueError("Either num_heads or num_head_channels has to be set, but both are -1".format(num_heads, dim_head))
        
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.no_time_emb = no_time_emb
        self.is_autoencoder = is_autoencoder
        self.attention_type = attention_type
        self.attention_fn = attention_fn
        
        # time embedding
        if not no_time_emb or self.num_classes is not None:
            time_embed_dim = model_channels * 4
        else:
            time_embed_dim = None
        
        if not no_time_emb:
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
        
        # class embedding
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        
        # layers option
        res_fn = partial(
            ResBlock,
            emb_channels=time_embed_dim,
            dropout=dropout,
            num_groups=num_groups,
            unet_dim=unet_dim,
            # use_checkpoint=use_checkpoint,
        )
        if attention_fn == "attention_block":
            attn_fn = lambda ch, num_heads, dim_head: AttentionBlock(
                channels=ch,
                n_heads=num_heads,
                dim_head=dim_head,
                # use_checkpoint=use_checkpoint,
                attention_type=attention_type,
                num_groups=num_groups
            )
        elif attention_fn == "spatial_transformer":
            attn_fn = lambda ch, num_heads, dim_head: SpatialTransformer(
                in_channels=ch,
                n_heads=num_heads,
                dim_head=dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                attention_type=attention_type
            )
        elif attention_fn == "no_attention":
            attn_fn = lambda *args, **kwargs: nn.Identity()
        else:
            raise NotImplementedError
        
        # downsampling part
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(unet_dim, in_channels, model_channels, 3, padding=1))
        ])
        self._feature_size = model_channels
        input_block_channels = [model_channels]
        channel = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [res_fn(channel, out_channels=mult*model_channels)]
                channel = mult*model_channels
                if ds in attention_resolutions:
                    d_head, num_heads = self.get_attn_attr(dim_head, num_heads, channel)
                    layers.append(attn_fn(channel, num_heads, d_head))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += channel
                input_block_channels.append(channel)
            if level != len(channel_mult) - 1:
                out_channel = channel
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        res_fn(channel, out_channels=out_channel, down=True)
                        if resblock_updown
                        else Downsample(channel, conv_resample, unet_dim=unet_dim, out_channels=out_channel)
                    ))
                channel = out_channel
                input_block_channels.append(channel)
                ds *= 2
                self._feature_size += channel
        
        # middle part
        d_head, num_heads = self.get_attn_attr(dim_head, num_heads, channel)
        self.middle_block = TimestepEmbedSequential(
            res_fn(channel),
            attn_fn(channel, num_heads, d_head),
            res_fn(channel)
        )
        self._feature_size += channel
        
        # upsampling part
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                i_channel = channel + input_block_channels.pop() if not is_autoencoder else channel
                layers = [res_fn(i_channel, out_channels=model_channels*mult)]
                channel = model_channels * mult
                if ds in attention_resolutions:
                    d_head, num_heads = self.get_attn_attr(dim_head, num_heads, channel)
                    layers.append(attn_fn(channel, num_heads, d_head))
                if level and i == num_res_blocks:
                    out_channel = channel
                    layers.append(
                        res_fn(channel, out_channels=out_channel, up=True)
                        if resblock_updown
                        else Upsample(channel, conv_resample, unet_dim=unet_dim, out_channels=out_channel)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += channel
        
        if not is_autoencoder:
            assert len(input_block_channels) == 0
        
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=channel, eps=1e-6, affine=True),
            nn.SiLU(),
            zero_module(conv_nd(unet_dim, model_channels, out_channels, 3, padding=1)), # for use
            # conv_nd(unet_dim, model_channels, out_channels, 3, padding=1) # for code test
        )
    
    
    def get_attn_attr(self, dim_head, num_heads, channel):
        if dim_head == -1:
            d_head = channel // num_heads
        else:
            num_heads = channel // dim_head
            d_head = dim_head
        if self.attention_type == "qkv_legacy":
            # num_heads = 1
            d_head = channel // num_heads if (self.attention_fn == "spatial_transformer") else dim_head
        return d_head, num_heads
    
    
    def convert_to_fp16(self):
        """ Convert the torso of the model to float16 """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """ Convert the torso of the model to float32 """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
    
    @property
    def inner_dtype(self):
        """ Get the dtype used by the torso of the model """
        return next(self.input_blocks.parameters()).dtype
    
    
    def get_embedding(self, t=None, y=None):
        emb = 0
        if not self.no_time_emb:
            assert t is not None, "Must specify 't' if the model has time embedding."
            t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
            emb += self.time_embed(t_emb)
        if self.num_classes is not None:
            assert y is not None, "Must specify 'y' if the model is class-conditional"
            emb += self.label_emb(y)
        return emb
    
    def encode(self, x, emb=None, context=None):
        if not self.no_time_emb or self.num_classes is not None:
            assert emb is not None
        
        hs = []
        h = x.type(self.inner_dtype)
        
        for module in self.input_blocks:
            h = module(h, emb, context)
            if not self.is_autoencoder:
                hs.append(h)

        return h, hs
    
    def decode(self, h, hs=None, emb=None, context=None):
        if not self.is_autoencoder:
            assert hs is not None
        if not self.no_time_emb or self.num_classes is not None:
            assert emb is not None
        
        h = self.middle_block(h, emb, context)
        
        for module in self.output_blocks:
            if not self.is_autoencoder:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        
        h = self.out(h)

        return h
    
    def forward(self, x, t=None, context=None, y=None):
        emb = self.get_embedding(t=t, y=y)
        h, hs = self.encode(x, emb, context=context)
        h = self.decode(h, hs, emb, context=context)
        
        return h


if __name__ == "__main__":
    # model = UNetModel(unet_dim=1, in_channels=1, out_channels=1, dim_head=32, attention_type="xformers")
    # x = torch.rand(2, 1, 1024) # (b, in_c, n)
    # t = torch.randint(0, 1000, (2,))
    # out = model(x, t)
    # print(out)
    # print("1D model output shape: ", out.shape) # (b, out_c, n)
    
    # model = UNetModel(unet_dim=2, in_channels=3, out_channels=1, dim_head=32, attention_type="qkv")
    # x = torch.rand(8, 3, 32, 32) # (b, c_in, h, w)
    # t = torch.randint(0, 1000, (8,))
    # out = model(x, t)
    # print(out)
    # print("2D model output shape: ", out.shape) # (b, c_out, h, w)
    
    # model = UNetModel(unet_dim=3, in_channels=1, out_channels=1, dim_head=32, attention_type="memory_efficient_attention").cuda()
    # x = torch.rand(8, 1, 32, 32, 32).cuda() # (b, c_in, x, y, z)
    # t = torch.randint(0, 1000, (8,)).cuda()
    # out = model(x, t)
    # print(out)
    # print("3D model output shape: ", out.shape) # (b, c_out, x, y, z)
    
    # for feature maps
    x = torch.rand(1, 96, 128, 128).cuda()
    t = torch.randint(0, 1000, (1,)).cuda()
    cond = torch.rand(1, 1280, 77).cuda()
    model = UNetModel(
        unet_dim=2,
        in_channels=96,
        out_channels=96,
        model_channels=320,
        attention_resolutions=[16, 8, 4],
        channel_mult=[1, 2, 4, 4],
        num_heads=8,
        attention_type="qkv_legacy",
        attention_fn="spatial_transformer",
        transformer_depth=1,
        context_dim=1280
    ).cuda()
    out = model(x, t, cond)
    print(out)
    print("Conditioned model output shape: ", out.shape)
