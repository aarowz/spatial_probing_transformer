"""
Spatial-Probing Transformer (PoC): custom Multi-Head Attention core.

This module intentionally implements scaled dot-product attention and multi-head
projection math directly (no `nn.Transformer` / `nn.MultiheadAttention`) to
support probing and visualization of attention weights.

Shape conventions:
- Sequence tensors are batch-first: (B, S, D)
- Per-head tensors: (B, H, S, Dh) where Dh = D // H
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """
    (B, S, D) -> (B, H, S, Dh)
    """
    b, s, d = x.shape
    if d % num_heads != 0:
        raise ValueError(f"d_model ({d}) must be divisible by num_heads ({num_heads}).")
    dh = d // num_heads
    return x.reshape(b, s, num_heads, dh).transpose(1, 2).contiguous()


def _merge_heads(x: Tensor) -> Tensor:
    """
    (B, H, S, Dh) -> (B, S, D)
    """
    b, h, s, dh = x.shape
    return x.transpose(1, 2).contiguous().reshape(b, s, h * dh)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    return_attn: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Scaled dot-product attention.

    Args:
        q, k, v: (B, H, S_q, Dh), (B, H, S_k, Dh), (B, H, S_k, Dh)
        attn_mask: Optional additive or boolean mask broadcastable to (B, H, S_q, S_k).
            - If boolean: True means "keep", False means "mask out".
            - If float: added directly to logits (use -inf for masked positions).
        key_padding_mask: Optional boolean mask of shape (B, S_k), where True indicates
            a padding position to be masked out (i.e., not attendable as a key).
        return_attn: if True, also return attention weights (B, H, S_q, S_k).

    Returns:
        out: (B, H, S_q, Dh)
        attn (optional): (B, H, S_q, S_k)
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4: (B, H, S, Dh).")

    bq, hq, sq, dhq = q.shape
    bk, hk, sk, dhk = k.shape
    bv, hv, sv, dhv = v.shape

    if not (bq == bk == bv and hq == hk == hv):
        raise ValueError("Batch/head dims must match across q, k, v.")
    if not (sk == sv):
        raise ValueError("Key/value sequence lengths must match.")
    if not (dhq == dhk == dhv):
        raise ValueError("Head dim (Dh) must match across q, k, v.")

    scale = 1.0 / math.sqrt(dhq)
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, S_q, S_k)

    if attn_mask is not None:
        # Broadcast to (B, H, S_q, S_k) if possible.
        if attn_mask.dtype == torch.bool:
            attn_logits = attn_logits.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_logits = attn_logits + attn_mask

    if key_padding_mask is not None:
        if key_padding_mask.dtype != torch.bool or key_padding_mask.ndim != 2:
            raise ValueError("key_padding_mask must be boolean with shape (B, S_k).")
        if key_padding_mask.shape != (bq, sk):
            raise ValueError(
                f"key_padding_mask must have shape (B, S_k)=({bq}, {sk}), got {tuple(key_padding_mask.shape)}."
            )
        # True indicates padding -> mask out for all heads and all queries.
        pad = key_padding_mask[:, None, None, :]  # (B, 1, 1, S_k)
        attn_logits = attn_logits.masked_fill(pad, float("-inf"))

    attn = torch.softmax(attn_logits, dim=-1)  # (B, H, S_q, S_k)
    out = torch.matmul(attn, v)  # (B, H, S_q, Dh)

    if return_attn:
        return out, attn
    return out


class MultiHeadAttention(nn.Module):
    """
    Batch-first multi-head (self/cross) attention:
      x: (B, S_q, D)
      context: (B, S_kv, D) if provided, else x (self-attn)
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, *, bias: bool = True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads

        self.W_q = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_k = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_v = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: (B, S_q, D)
            context: optional (B, S_kv, D). If None, uses self-attention (context=x).
            attn_mask: optional additive or boolean mask broadcastable to (B, H, S_q, S_kv).
            key_padding_mask: optional boolean mask (B, S_kv) where True indicates padding keys.
            return_attn: if True, also return attention weights (B, H, S_q, S_kv).
        """
        if x.ndim != 3:
            raise ValueError("x must be rank-3: (B, S, D).")
        b, s_q, d = x.shape
        if d != self.d_model:
            raise ValueError(f"x last dim must be d_model={self.d_model}, got {d}.")

        ctx = x if context is None else context
        if ctx.ndim != 3:
            raise ValueError("context must be rank-3: (B, S_ctx, D).")
        if ctx.shape[0] != b or ctx.shape[2] != self.d_model:
            raise ValueError("context must have shape (B, S_ctx, d_model) matching x batch/d_model.")

        q = self.W_q(x)  # (B, S_q, D)
        k = self.W_k(ctx)  # (B, S_kv, D)
        v = self.W_v(ctx)  # (B, S_kv, D)

        qh = _split_heads(q, self.num_heads)  # (B, H, S_q, Dh)
        kh = _split_heads(k, self.num_heads)  # (B, H, S_kv, Dh)
        vh = _split_heads(v, self.num_heads)  # (B, H, S_kv, Dh)

        attn_out = scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_attn=return_attn,
        )

        if return_attn:
            out_h, attn = attn_out  # type: ignore[misc]
            out = _merge_heads(out_h)  # (B, S_q, D)
            out = self.W_o(out)
            return out, attn

        out_h = attn_out  # type: ignore[assignment]
        out = _merge_heads(out_h)  # (B, S_q, D)
        out = self.W_o(out)
        return out


if __name__ == "__main__":
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=512, num_heads=8)

    x = torch.randn(2, 196, 512)
    y, attn = mha(x, return_attn=True)
    print("self-attn:", x.shape, "->", y.shape, "attn:", attn.shape)

    ctx = torch.randn(2, 196 + 1, 512)
    y2, attn2 = mha(x, context=ctx, return_attn=True)
    print("cross-attn:", x.shape, "x ctx", ctx.shape, "->", y2.shape, "attn:", attn2.shape)
