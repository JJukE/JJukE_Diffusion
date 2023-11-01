import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from jjuke.utils import default, max_neg_value, zero_module, conv_nd


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class CheckpointFunction(torch.autograd.Funciton):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    
    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the Tensor storage
            # in place, which is not allowed for detach()'d Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    
    Args:
    func: A function to evaluate,
    inputs: Argument sequence to pass to `func`.
    params: A sequence of parameters `func` depends on but does not explicitly
            take as arguments.
    param flag: If False, disable gradient checkpointing
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class GEGLU(nn.Module):
    """ Nonlinear projection for Feed Forward Layer """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        self.proj = nn.Linear(dim_in, dim_out*2)
    
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        
        hidden_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, hidden_dim)
        
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


# Attention modules    
def count_flops_attn(model, _x, y):
    """ Counter for the `thop` package to count the operations in attention
    
    Use case:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops}
        )
    """
    b, c, *data_structure = y[0].shape
    seq_len = int(np.prod(data_structure))
    # Two matmuls with the same number of ops.
    # First: weight matrix
    # Second: combination of the value vectors
    matmul_ops = 2 * b * (seq_len ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """ QKV attention module for both self-attention and cross-attention that reshapes and then splits """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv, k=None, v=None, mask=None):
        """ Apply QKV attention
        
        Notation:
            b : Batch size
            m : Number of heads of multi-head attention
            d : Dimension(channels) of each head (of queries, keys, and values)
            c : Channels (m * d)
            n : Sequence length (ex. height * width in 2D)
        
        Args:
            qkv: (b, 3*c, n) tensor of Qs, Ks, and Vs or (b, c, n) tensor of queries
            k: If specified, (b, c, n) tensor of keys
            v: If specified, (b, c, n) tensor of values
        
        Returns:
            (b, c, n) tensor after attention
        """
        if k is None and v is None:
            b, width, n = qkv.shape
            assert width % (3 * self.n_heads) == 0
            d = width // (3 * self.n_heads)
            q, k, v = rearrange(qkv, "b (m D) n -> (b m) D n", m=self.n_heads).split(d, dim=1) # (B, d, n), B=b*m, D=3*d
        elif k is not None and v is not None:
            q, k, v = qkv, k, v
            b, n, d = q.shape[0], q.shape[2], q.shape[1] // self.n_heads
        else:
            raise NotImplementedError
        # b, width, n = qkv.shape # width = 3*c = 3*m*d
        # assert width % (3 * self.n_heads) == 0
        # d = width // (3 * self.n_heads)
        
        # q, k, v = rearrange(qkv, "b (m D) n -> (b m) D n", m=self.n_heads).split(d, dim=1) # (B, d, n), B = b*m, D = 3*d
        scale = 1 / math.sqrt(math.sqrt(d))
        
        # calculate attention score (more stable with f16 than dividing afterwards)
        weight = torch.einsum("B d i, B d j -> B i j", q*scale, k*scale) # (B, n, n)
        
        # masking
        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b m) () j", m=self.n_heads)
            weight.masked_fill_(~mask, max_neg_value(weight))
        
        # calculate attention weight (attention distribution)
        weight = torch.softmax(weight, dim=-1).type(weight.dtype)
        
        # calculate attention output (context vector)
        a = torch.einsum("B i n, B d n -> B d i", weight, v) # (B, d, n)
        a = rearrange(a, "(b m) d n -> b (m d) n", m=self.n_heads) # (b, c, n)
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)
        

class QKVAttention(nn.Module):
    """ QKV attention module that splits and then reshapes """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
    
    def forward(self, qkv):
        """ Apply QKV attention
        
        Notation:
            b : Batch size
            m : Number of heads of multi-head attention
            d : Dimension(channels) of each head (of queries, keys, and values)
            c : Channels (m * d)
            n : Sequence length (ex. height * width in 2D)
        
        Args:
            qkv: (b, 3*c, n) tensor of Qs, Ks, and Vs or (b, c, n) tensor of queries
            k: If specified, (b, c, n) tensor of keys
            v: If specified, (b, c, n) tensor of values
        
        Returns:
            (b, c, n) tensor after attention
        """
        if k is None and v is None:
            b, width, n = qkv.shape
            assert width % (3 * self.n_heads) == 0
            d = width // (3 * self.n_heads)
            q, k, v = qkv.chunk(3, dim=1) # (b, c, n)
        elif k is not None and v is not None:
            q, k, v = qkv, k, v
            b, n = q.shape[0], q.shape[2]
        else:
            raise NotImplementedError
        # b, width, n = qkv.shape
        # assert width % (3 * self.n_heads) == 0
        
        # d = width // (3 * self.n_heads)
        # q, k, v = qkv.chunk(3, dim=1) # (b c n)
        scale = 1 / math.sqrt(math.sqrt(d))
        
        q = rearrange(q*scale, "b (m d) n -> (b m) d n", m=self.n_heads) # (B d n), B=b*m
        k = rearrange(k*scale, "b (m d) n -> (b m) d n", m=self.n_heads) # (B d n)
        v = rearrange(v, "b (m d) n -> (b m) d n", m=self.n_heads) # (B d n)
        
        # calculate attention score (more stable with f16 than dividing afterwards)
        weight = torch.einsum("B d i, B d j -> b i j", q, k) # (B n n)
        
        # calculate attention weight (attention distribution)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        
        # calculate attention ouptut (context vector)
        a = torch.einsum("B i n, B d n -> B d i", weight, v) # (B d n)
        a = rearrange(a, "(b m) d n -> b (m d) n", m=self.n_heads) # (b c n)
        return a


# class LinearAttention(nn.Module):
#     """ Linear self-attention module (one of the variants for computational efficency) """
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
        
#         self.heads = heads
#         hidden_dim = dim_head * heads
        
#         self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1, bias=False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    
#     def forward(self, x):
#         """
#         Notation:
#         b : Batch size
#         m : Number of heads of multi-head attention
#         d : Dimension of each head (of queries, keys, and values)
#         c : channels (m * d)
#         n : Sequence length (height * width in 2D)
#         """
#         b, c, h, w = x.shape
        
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: rearrange(t, "b (m d) h w -> b m d (h w)", m=self.heads), qkv)
        
#         k = k.softmax(dim=-1)
        
#         context = torch.einsum("b m i n, b m j n -> b m i j", k, v)
        
#         out = torch.einsum("b m i j, b m i n -> b m j n", context, q)
#         out = rearrange(out, "b m d (h w) -> b (m d) h w", m=self.heads, h=h, w=w)
#         return self.to_out(out)


# class SpatialSelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
        
#         self.in_channels = in_channels
        
#         self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True) # TODO: when replaced with (Chan)LayerNorm?
#         self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
#     def forwrad(self, x):
#         """
#         Notation:
#         b : Batch size
#         m : Number of heads of multi-head attention
#         d : Dimension of each head (of queries, keys, and values)
#         c : channels (m * d)
#         n : Sequence length (height * width in 2D)
#         """
#         h_ = x
        
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)
        
#         # get q, k, v
#         b, c, h, w = q.shape
#         q = rearrange(q, "b c h w -> b (h w) c")
#         k = rearrange(k, "b c h w -> b c (h w)")
#         v = rearrange(v, "b c h w -> b c (h w)")
        
#         # attention score
#         w_ = torch.einsum("b i j, b j k -> b i k", q, k) # q^\top k, (b, n, n)
        
#         w_ = w_ * (int(c)**(-0.5)) # (q^top k)/sqrt{d_k}
#         w_ = F.softmax(w_, dim=2)
        
#         # attend to value
#         w_ = rearrange(w_, "b i j -> b j i") # (b, n, n)
#         h_ = torch.einsum("b i j, b j k -> b i k", v, w_) # (b, c, n)
#         h_ = rearrange(h_, "b c (h w) -> b c h w", h=h) # (b, c, h, w)
#         return x+h_ # residual connection


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., attention_type="qkv_legacy"):
        """ Self-attention if context_dim is None, cross-attention otherwise """
        super().__init__()
        
        hidden_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.attention_type = attention_type
        
        self.to_q = conv_nd(1, query_dim, hidden_dim, 1, bias=False)
        self.to_k = conv_nd(1, context_dim, hidden_dim, 1, bias=False)
        self.to_v = conv_nd(1, context_dim, hidden_dim, 1, bias=False)
        
        if attention_type == "qkv_legacy":
            self.attention = QKVAttentionLegacy(heads)
        elif attention_type == "qkv":
            self.attention = QKVAttention(heads)
        elif attention_type == "xformers":
            try:
                from xformers.components.attention import ScaledDotProduct
                self.attention = ScaledDotProduct
            except ImportError:
                print("There's no xformers installed! Using QKVAttentionLegacy instead.")
                self.attention = QKVAttentionLegacy(heads)
        elif attention_type in "memory_efficient_attention|flash|flash16".split("|"):
            try:
                from xformers.ops import memory_efficient_attention
                self.attention = memory_efficient_attention
            except ImportError:
                print("There's no xformers installed! Using QKVAttentionLegacy instead.")
                self.attention = QKVAttentionLegacy(heads)
        else:
            raise NotImplementedError(attention_type)
        
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forwrad(self, x, context=None, mask=None):
        """
        Notation:
        b : Batch size
        m : Number of heads of multi-head attention
        d : Dimension of each head (of queries, keys, and values)
        c : channels (m * d)
        n : Sequence length (height * width in 2D)
        """
        x = rearrange(x, "b n c -> b c n")
        
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        if self.attention_type in ("qkv", "qkv_legacy"):
            out = self.attention(q, k, v)
            out = rearrange(out, "b c n -> b n c")
        else:
            # q, k, v = map(lambda t: rearrange(t, "b (m d) n -> b n m d", m=self.heads).contiguous(), (q, k, v))
            q, k, v = map(lambda t: rearrange(t, "b (m d) n -> (b m) n d", m=self.heads).contiguous(), (q, k, v)) # (B n d)
            out = self.attention(q, k, v)
            # out = rearrange(out, "b n m d -> b n (m d)").contiguous() # (b n c)
            out = rearrange(out, "(b m) n d -> b n (m d)", m=self.heads).contiguous() # (b n c)
        return self.to_out(out)
        
        # q, k, v = map(lambda t: rearrange(t, "b n (m d) -> (b m) n d", m=self.heads), (q, k, v))
        
        # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale # scaled dot product, (b*m, n, n)
        
        # if mask is not None:
        #     mask = rearrange(mask, "b ... -> b (...)")
        #     mask = repeat(mask, "b j -> (b m) () j", m=self.heads)
        #     sim.masked_fill_(~mask, max_neg_value(sim))
        
        # attn = sim.softmax(dim=-1)
        
        # out = torch.einsum("b i n, b n d -> b i d", attn, v)
        # out = rearrange(out, "(b m) n d -> b n (m d)", m=self.heads)
        # return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        dim_head,
        dropout=0.,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        attention_type="qkv_legacy"
    ):
        super().__init__()
        
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=dim_head,
            dropout=dropout,
            attention_type=attention_type
        ) # self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=dim_head,
            dropout=dropout,
            attention_type=attention_type
        ) # self-attention if context is None
        
        self.norm1 = nn.LayerNorm(dim) # TODO: difference when replaced by ChanLayerNorm
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint
    
    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    
    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """ Transformer block for 2D data """
    def __init__(
        self,
        in_channels,
        n_heads,
        dim_head,
        depth=1,
        dropout=0.,
        context_dim=None,
        attention_type="qkv_legacy",
        num_groups=32
    ):
        super().__init__()
        
        self.in_channels = in_channels
        hidden_dim = n_heads * dim_head
        
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = conv_nd(2, in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    hidden_dim,
                    n_heads,
                    dim_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    attention_type=attention_type
                ) for d in range(depth)
            ])
        self.proj_out = zero_module(conv_nd(2, hidden_dim, in_channels, kernel_size=1, stride=1, padding=0))
    
    def forward(self, x, context=None):
        """
        Notation:
        b : Batch size
        c : channels of 2D input data
        h : height
        w : width
        n : Sequence length (height * width in 2D)
        """
        # NOTE: If context is None, cross-attention defaults to self-attention
        b, c, *dims = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        x = rearrange(x, "b c ... -> b (...) c") # (b, n, c)
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
        
        x = rearrange(x, "b (...) c -> b c ...", **{"d{}".format(i): dim for i, dim in enumerate(dims, start=1)}) # (b, c, h, w) for 2D
        x = self.proj_out(x)
        return x + x_in


class AttentionBlock(nn.Module):
    """ Attention block that allows spatial positions to attend to each other """
    def __init__(
        self,
        channels,
        n_heads=1,
        dim_head=-1,
        use_checkpoint=False,
        attention_type="qkv_legacy", # ["qkv", "qkv_legacy", "xformers", "falsh", "flash16"]
        num_groups=32
    ):
        super().__init__()
        
        self.channels = channels
        if dim_head == -1:
            self.n_heads = n_heads
        else:
            assert (
                channels % dim_head == 0
            ), "q, k, v channels {} is not divisible by number of head channels {}.".format(channels, dim_head)
            self.n_heads = channels // dim_head
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
        self.qkv = conv_nd(1, channels, channels*3, 1)
        
        if attention_type == "qkv":
            # split qkv before split heads -> use new attention order
            self.attention = QKVAttention(self.n_heads)
        elif attention_type == "qkv_legacy":
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.n_heads)
        elif attention_type == "xformers":
            try:
                from xformers.components.attention import ScaledDotProduct
                self.attention = ScaledDotProduct
            except ImportError:
                print("There's no xformers installed! Using QKVAttentionLegacy instead.")
                self.attention = QKVAttentionLegacy(self.n_heads)
        elif attention_type in "memory_efficient_attention|flash|flash16".split("|"):
            try:
                from xformers.ops import memory_efficient_attention
                self.attention = memory_efficient_attention
            except ImportError:
                print("There's no xformers installed! Using QKVAttentionLegacy instead.")
                self.attention = QKVAttentionLegacy(self.n_heads)
        else:
            raise NotImplementedError(attention_type)
        
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
    
    def forward(self, x):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forwrad, x)
    
    def _forward(self, x):
        b, c, *dims = x.shape
        
        # Flatten the data structure dimensions into sequence length (n)
        x = rearrange("b c ... -> b c (...)")
        x = self.norm(x)
        
        qkv = self.qkv(x)
        
        if self.attention_type in "qkv|qkv_legacy".split("|"):
            h_ = self.attention(qkv)
        else:
            # q, k, v = rearrange(qkv, "b (num m d) n -> num b n m d", num=3, m=self.n_heads)
            q, k, v = rearrange(qkv, "b (num m d) n -> num (b m) n d", num=3, m=self.n_heads).contiguous() # (B n c)
            # q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            if self.attention_type == "flash16" and q.device.type != "cpu":
                dtype = q.dtype
                q, k, v = q.half(), k.half(), v.half()
            h_ = self.attention(q, k, v)
            if self.attention_type == "flash16" and q.device.type != "cpu":
                h_ = h_.to(dtype)
            # h_ = rearrange(h_, "b n m d -> b (m d) n").contiguous() # (b c n)
            h_ = rearrange(h_, "(b m) n d -> b (m d) n", m=self.n_heads).contiguous()
        
        h_ = self.proj_out(h_)
        
        # Reshape back to the original data structure # TODO: check if both two ways work
        # pattern = "b c ... -> b c " + " ".join(["d{}".format(i) for i in range(1, len(dims)+1)])
        # data_dim_dict = {"d{}".format(i): dim for i, dim in enumerate(dims, start=1)}
        # h_ = rearrange(h_, pattern, **data_dim_dict)
        h_ = rearrange(h_, "b c ... -> b c ...", **{"d{}".format(i): dim for i, dim in enumerate(dims, start=1)})
        x = rearrange(x, "b c ... -> b c ...", **{"d{}".format(i): dim for i, dim in enumerate(dims, start=1)})
        
        return x + h_