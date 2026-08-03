"""Small causal transformer for modular addition."""

import torch
from torch import Tensor, nn


class GrokkingTransformer(nn.Module):
    """Two-layer decoder-style transformer with learned positions."""

    def __init__(
        self,
        modulus: int = 97,
        width: int = 128,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.modulus = modulus
        self.token_embedding = nn.Embedding(modulus, width)
        self.position_embedding = nn.Parameter(torch.randn(2, width) * 0.02)
        block = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, modulus, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """Predict the modular sum from the final sequence position."""
        hidden = self.token_embedding(tokens) + self.position_embedding
        mask = torch.triu(torch.ones(2, 2, device=tokens.device, dtype=torch.bool), diagonal=1)
        hidden = self.layers(hidden, mask=mask, is_causal=True)
        return self.head(self.norm(hidden[:, -1]))
