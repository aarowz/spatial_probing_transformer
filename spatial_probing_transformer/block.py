"""
Pre-LN Transformer block supporting self-attention and cross-attention.

Uses MultiHeadAttention from model.py with optional context for KV.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .model import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block:
      x = x + Dropout(Attn(LN_q(x), context=LN_kv(ctx)))
      x = x + Dropout(MLP(LN_ff(x)))

    Self-attn: pass context=None (KV come from the same normalized query stream).
    Cross-attn: pass context with shape (B, S_kv, D).
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        *,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.mlp_ratio = int(mlp_ratio)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, bias=bias)

        hidden = mlp_ratio * d_model
        self.norm_ff = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: (B, S_q, D) query stream.
            context: optional (B, S_kv, D) key/value stream; if None, self-attention on x.
            return_attn: if True, also return attention weights (B, H, S_q, S_kv).

        Returns:
            x after block, optionally with attn weights.
        """
        q = self.norm_q(x)
        kv = self.norm_kv(context) if context is not None else None

        if return_attn:
            attn_out, attn_w = self.attn(q, context=kv, return_attn=True)
            x = x + self.drop(attn_out)
            x = x + self.drop(self.mlp(self.norm_ff(x)))
            return x, attn_w

        attn_out = self.attn(q, context=kv, return_attn=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm_ff(x)))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    blk = TransformerBlock(d_model=512, num_heads=8, mlp_ratio=4, dropout=0.0)

    x = torch.randn(2, 196, 512)
    y = blk(x)
    print("self-attn:", x.shape, "->", y.shape)
    assert y.shape == (2, 196, 512)

    q = torch.randn(2, 16, 512)
    ctx = torch.randn(2, 196, 512)
    z, attn = blk(q, context=ctx, return_attn=True)
    print("cross-attn:", q.shape, "ctx", ctx.shape, "->", z.shape, "attn:", attn.shape)
    assert z.shape == (2, 16, 512)
    assert attn.shape == (2, 8, 16, 196)

    print("block.py smoke test OK.")
