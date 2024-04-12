import copy
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import math
from einops import rearrange


def sequential(dims, end_with_relu=False):
    """
    Instantiate a Sequential with ReLUs
    """

    def linear(i):
        if i == len(dims) - 2:
            return [nn.Linear(dims[i], dims[i + 1])]
        else:
            return [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]

    modules = [linear(i) for i in range(len(dims) - 1)]
    if end_with_relu:
        modules.append([nn.ReLU()])
    modules = sum(modules, [])
    return nn.Sequential(*modules)


class MyTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, kdim=None, vdim=None,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 use_norm=True):
        super(MyTransformerDecoderLayer, self).__init__()
        self.use_norm = use_norm
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead,
                                                    kdim=kdim, vdim=vdim,
                                                    dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MyTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2, attn_output_weights = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask, need_weights=True)
        tgt = tgt + self.dropout1(tgt2)
        if self.use_norm:
            tgt = self.norm1(tgt)
        tgt2, xattn_output_weights = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask, need_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        if self.use_norm:
            tgt = self.norm2(tgt)
        tgt3 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt3)
        if self.use_norm:
            tgt = self.norm3(tgt)
        return tgt, xattn_output_weights, attn_output_weights


class MyTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch
             (optional).
            memory_key_padding_mask: the mask for the memory keys per batch
             (optional).
        Returns:
            output: output of decoder
            output2: attended values on memory
            xatt: cross attention weights
            selfatt: self attention weights

        """
        output = tgt

        for mod in self.layers:
            (
                output, x_weights, self_weights
            ) = mod(output, memory, tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        # memory: 1024xBx318 -> Bx1x318x1024
        # x_weights: Bx2x1024 -> Bx2x1x1024
        output_feats = (
                memory.permute([1, 2, 0]).unsqueeze(1) * x_weights.unsqueeze(2)
        ).sum(3).permute([1, 0, 2])

        return output, output_feats, x_weights, self_weights


class MultiHeadAttention(nn.Module):
    """
    Tweaked code from https://theaisummer.com/einsum-attention/
    Args:
        dim_q: query dimension
        dim_kv: key/value dimension
        heads: the number of distinct representations to learn
        dim_head: the dim of the head. In general dim_head<dim.
        However, it may not necessarily be (dim/heads)
    Returns: output
       output[0]: result of self/cross attention, size: BxSxD
       output[1]: seperate attention for each head, size: BxHxSxD
    """

    def __init__(self, dim_q, dim_kv=None, heads=8, dim_head=None):
        super().__init__()
        if dim_kv is None:
            dim_kv = dim_q
        self.dim_head = (int(dim_q / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim_q, _dim, bias=False)
        self.to_kv = nn.Linear(dim_kv, _dim * 2, bias=False)
        self.W_0 = nn.Linear(_dim, dim_q, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, memory=None, mask=None):
        # allows module to be used as self or cross attention
        if memory is None:
            memory = x
        assert x.dim() == 3
        assert memory.dim() == 3

        # Step 1
        q = self.to_q(x)  # [batch, tokens, dim*heads ]
        kv = self.to_kv(memory)  # [batch, tokens, dim*2*heads ]

        # Step 2
        # decomposition to q,v,k and cast to tuple
        # q should only differ with k and v in terms of the sequens length: t
        q = rearrange(q, 'b t (d h) -> b h t d', h=self.heads)
        k, v = tuple(rearrange(kv, 'b t (d k h) -> k b h t d',
                               k=2, h=self.heads))

        # Step 3
        # resulted shape will be: [batch, heads, q_tokens, memory_tokens]
        scaled_dot_prod = torch.einsum('b h t d, b h s d -> b h t s', q, k) * \
                          self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 4. Calc result per batch and per head h
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)

        # Step 5. Re-compose: merge heads with dim_head d
        out = rearrange(out, "b h t d -> b t (h d)")

        # Step 6. Apply final linear transformation layer
        return self.W_0(out), attention


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, dim_mem=None, sem_dim=300, heads=8, dim_head=None,
                 dim_linear_block=256, dropout=0.1):
        super().__init__()
        # conditional mlp
        self.cond_mlp1 = ConditionalMLP([dim, dim_linear_block, dim], sem_dim)

        # cross-attention
        self.mhxa = MultiHeadAttention(dim_q=dim, dim_kv=dim_mem, heads=heads,
                                       dim_head=dim_head)
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.norm_3 = nn.LayerNorm(dim)

        # conditional mlp
        self.cond_mlp2 = ConditionalMLP([dim, dim_linear_block, dim], sem_dim)


    def forward(self, x, y, memory, time, mem_mask=None):
        # MLP conditioned on y
        x_= self.cond_mlp1(x, y)
        x = self.norm_1(self.drop(x_) + x)

        # multi-head cross-attention
        x = rearrange(x, 'b d -> b 1 d')
        x_, att_cross = self.mhxa(x, memory=memory, mask=mem_mask)
        x = self.norm_2(self.drop(x_) + x)
        x = rearrange(x, 'b 1 d -> b d')

        # Final MLP conditioned on y
        return self.norm_3(self.cond_mlp2(x, y) + x), att_cross


class TransformerDecoder(nn.Module):
    def __init__(self, dim, dim_mem, sem_dim=300, layers=6, heads=4,
                 dim_head=None):
        super().__init__()
        # time step mlp
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, 2 * dim)
        )

        self.block_list = [
            TransformerDecoderBlock(dim=dim, dim_mem=dim_mem, sem_dim=sem_dim,
                                    heads=heads, dim_head=dim_head)
            for _ in range(layers)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, y, memory, time, mask=None):
        # calculate time embedding
        scale, shift = self.time_mlp(time).chunk(2, dim=1)
        x = x * (scale +1) + shift

        att_self, att_cross = None, None
        for layer in self.layers:
            x, att_cross = layer(x, y, memory, time, mask)
        return x, att_cross


class ConditionalLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_cond):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = sequential([dim_cond, dim_cond//2, dim_in*dim_out])
        self.bias = sequential([dim_cond, dim_cond//2, dim_out])

    def forward(self, x, cond):
        L = rearrange(self.linear(cond), 'b (o i) -> b o i', o=self.dim_out)
        B = self.bias(cond)
        return torch.einsum('boi, bi -> bo', L, x) + B


class ConditionalMLP(nn.Module):
    def __init__(self, dims, dim_cond):
        super().__init__()
        modules = [ConditionalLinear(dims[i], dims[i+1], dim_cond)
                   for i in range(len(dims) - 1)]
        self.sequential = nn.Sequential(*modules)

    def forward(self, x, cond):
        for s in self.sequential[:-1]:
            x = F.relu(s(x, cond))
        x = self.sequential[-1](x, cond)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))