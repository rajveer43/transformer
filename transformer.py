import copy
import math
from typing import Optional, Any, Union, Callable, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor, nn
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList


class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None):
        # norm-first ignored.
        super(Transformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                layer_norm_eps, batch_first, norm_first,
                                                device=device)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                layer_norm_eps, batch_first, norm_first,
                                                device=device)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class. (
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, device=device)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, device=device)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, device=device)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, device=device)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, device=device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        if self.norm_first:
            x += self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x += self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x += self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout3(x)


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:

    bsz, tgt_len, E = q.shape
    q /= math.sqrt(E)

    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    # (bsz * num_heads, tgt_len, embed_dim) x (bsz * num_heads, embed_dim, src_len) ->
    # (bsz * num_heads, tgt_len, src_len)
    attn = torch.bmm(q, k.transpose(-2, -1))

    # attn_mask (bsz * num_heads, tgt_len, tgt_len)
    if attn_mask is not None:
        # attn += attn_mask
        attn = attn.masked_fill(attn_mask, float(-1e9))

    attn = torch.softmax(attn, dim=-1)
    # why over dim=-1? because we compute the importance of src (dim=-1) on tgt (dim=-2).
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    # (bs * num_heads, tgt_len, src_len) x (bs * num_heads, src_len, embed_dim) -> (bs * num_heads, tgt_len, embed_dim)
    output = torch.bmm(attn, v)
    return output, attn


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0., kdim=None, vdim=None, batch_first=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        # self.kdim = kdim if kdim is not None else embed_dim # In transformer kdim = dmodel/h (embed_dim/num_heads) = head_dim
        # self.vdim = vdim if vdim is not None else embed_dim # In transformer vdim = dmodel/h (embed_dim/num_heads) = head_dim
        # self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads # in default scenario head_dim=64
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.kdim = kdim if kdim is not None else self.head_dim
        self.vdim = kdim if kdim is not None else self.head_dim

        # self.q_proj_weight = Linear(embed_dim, self.kdim, bias=False) # All the W_i^Qs are here work seperately.
        # self.k_proj_weight = Linear(embed_dim, self.kdim, bias=False)
        # self.v_proj_weight = Linear(embed_dim, self.vdim, bias=False)

        self.q_proj_weight = Linear(embed_dim, num_heads * self.kdim, bias=False)  # All the W_i^Qs are here work seperately.
        self.k_proj_weight = Linear(embed_dim, num_heads * self.kdim, bias=False)
        self.v_proj_weight = Linear(embed_dim, num_heads * self.vdim, bias=False)
        self.fc = Linear(num_heads * self.vdim, self.embed_dim)

        self.dropout_p = dropout
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, need_weights: bool = True):

        r"""

        Args:
            query:
            key:
            value:
            attn_mask:
            key_padding_mask:
            need_weights:

        Returns:

        """
        num_heads = self.num_heads
        head_dim = self.head_dim

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # for our case in transformer.
        # query: (tgt_len, bs, embed_dim)
        # key: (src_len, bs, embed_dim)
        # value: (src_len, bs, embed_dim)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.size(0)

        q = self.q_proj_weight(query) # (query_len, bs, heads * kdim)
        k = self.k_proj_weight(key)
        v = self.v_proj_weight(value)

        # split embedding vectors into nheads pieces
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)
        # q: (bsz * nheads, tgt_len, head_dim)
        # k: (bsz * nheads, src_len, head_dim)
        # v: (bsz * nheads, src_len, head_dim)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.logical_or(key_padding_mask)

        attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, self.dropout_p)
        # attn_output: (bsz * nheads, tgt_len, head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # attn_output: (tgt_len, bsz, embed_dim=nheads*head_dim)
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1).contiguous()

        attn_output = self.fc(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.contiguous().view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
