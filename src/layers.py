import torch
from torch import nn
import einops
from einops import rearrange, einsum
from math import sqrt
import numpy

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        std = sqrt(2 / (in_features + out_features))
        init_weights = torch.nn.init.trunc_normal_(
            torch.zeros((out_features, in_features), dtype=dtype, device=device),
            mean = 0,
            std = std,
            a = -3 * std,
            b = 3 * std
        )
        self.W = nn.Parameter(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        init_weights = torch.nn.init.trunc_normal_(
            torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device),
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )
        self.embedding_matrix = nn.Parameter(init_weights)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.to(torch.long)
        # token_ids: (batch_size, sequence_length)
        # advanced indexing into embedding matrix to produce output
        # output: (batch_size, sequence_length, embedding_dim)
        return self.embedding_matrix[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gi = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x: (batch_size, sequence_length, d_model)
        x_square_mean = einops.reduce(x**2, "batch seq_len d_model -> batch seq_len", "mean")
        rms = torch.sqrt(x_square_mean + self.eps)
        rms = rearrange(rms, "batch seq_len -> batch seq_len 1")
        rms_norm = (x / rms) * self.gi

        return rms_norm.to(in_dtype)
