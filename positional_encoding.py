import math

import torch
from torch import nn, Tensor
from torch.nn import Dropout, Linear, Module, Parameter


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 batch_first: bool = False,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        if batch_first:
            pos_embedding = pos_embedding.unsqueeze(0)
        else:
            pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

        self.batch_first = batch_first

    def forward(self, embedding: Tensor):
        if not self.batch_first:
            return self.dropout(embedding + self.pos_embedding[:embedding.size(0), :])
        else:
            return self.dropout(embedding + self.pos_embedding[:, :embedding.size(1)])

