"""
Spatial Prober: image tokens with self-attention, coordinate queries with cross-attention.

Queries are normalized coordinates in [0, 1]^2 (x horizontal, y vertical).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn

from .block import TransformerBlock
from .embeddings import CoordinateEmbedding, PatchEmbedding, SpatialPositionalEncoding


class SpatialProber(nn.Module):
    """
    Point-probe classification: logits per query coordinate over image patches.

    Forward:
      images (B, 3, H, H) -> patch embed + 2D PE -> 2x self-attn -> image tokens
      queries (B, N, 2) -> coordinate embed -> cross-attn -> logits (B, N, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_self_blocks: int = 2,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        img_size: int = 224,
        patch_size: int = 16,
        *,
        bias: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_self_blocks = int(num_self_blocks)

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            d_model=d_model,
        )
        self.spatial_pe = SpatialPositionalEncoding(
            d_model=d_model,
            grid_size=self.patch_embed.grid_size,
        )

        self.image_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    bias=bias,
                )
                for _ in range(num_self_blocks)
            ]
        )

        self.coord_embed = CoordinateEmbedding(
            in_dim=2,
            hidden_dim=d_model,
            d_model=d_model,
        )

        self.cross_block = TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            bias=bias,
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, images: Tensor, queries: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            images: (B, 3, img_size, img_size)
            queries: (B, N, 2) in [0, 1] — x column-wise, y row-wise.

        Returns:
            logits: (B, N, num_classes)
            attn_weights: (B, num_heads, N, num_patches) from cross-attention block.
        """
        if images.ndim != 4:
            raise ValueError("images must be rank-4: (B, 3, H, W).")
        if queries.ndim != 3 or queries.shape[-1] != 2:
            raise ValueError("queries must have shape (B, N, 2).")

        b, c, h, w = images.shape
        if c != 3 or h != w:
            raise ValueError("images must be (B, 3, H, H).")

        t = self.spatial_pe(self.patch_embed(images))

        for blk in self.image_blocks:
            t = blk(t)

        q = self.coord_embed(queries)
        q, attn_w = self.cross_block(q, context=t, return_attn=True)

        logits = self.head(self.final_norm(q))
        return logits, attn_w


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SpatialProber(num_classes=5, d_model=512, num_heads=8)
    imgs = torch.randn(2, 3, 224, 224)
    coord = torch.rand(2, 16, 2)
    logits, attn = model(imgs, coord)
    print("SpatialProber logits:", logits.shape, "attn:", attn.shape)
    assert logits.shape == (2, 16, 5)
    assert attn.shape == (2, 8, 16, 196)
    print("prober.py smoke test OK.")
