import torch
from torch import nn, Tensor
from torch.nn import Module

from positional_encoding import PositionalEncoding
from transformer import TransformerEncoderLayer, TransformerEncoder


class PointwiseFeedForwardNetwork(Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class ViT(Module):
    def __init__(self, image_res: int, patch_res: int, num_classes: int, latent_dim: int = 768, num_layers: int = 12,
                 nheads: int = 12, dim_feedforward: int = 3072, nchannels: int = 3, dropout: int = 0.1,
                 batch_first: bool = True):
        super(ViT, self).__init__()
        assert image_res % patch_res == 0, "Image resolution must be divisible by patch resolution."
        self.image_res = image_res
        self.patch_res = patch_res

        self.npatches = (image_res // patch_res)**2
        self.patch_dim = nchannels * patch_res**2
        self.patch_embedding = nn.Linear(self.patch_dim, latent_dim)

        self.pe = PositionalEncoding(latent_dim, dropout=dropout, batch_first=batch_first, maxlen=self.npatches+1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        encoder_layer = TransformerEncoderLayer(latent_dim, nheads, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, x: Tensor):
        r"""

        Args:
            x: (bsz, C, H, W)

        Returns:

        """
        bsz = x.size(0)
        x = x.transpose(1, -1).contiguous().view(-1, self.npatches, self.patch_dim)
        x = self.patch_embedding(x) # x: (bsz, npatches, latent_dim)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pe(x)
        x = self.transformer(x)[:, 0]
        return self.mlp_head(x)