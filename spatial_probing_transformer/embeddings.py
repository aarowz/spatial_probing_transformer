"""
Spatial-Probing Transformer (PoC): patch, coordinate, and 2D sinusoidal embeddings.

Shape conventions:
- Images: (B, C, H, W)
- Sequence tensors: batch-first (B, S, D)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    ViT-style patchify via Conv2d: (B, C, H, H) -> (B, num_patches, d_model).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 512,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.d_model = int(d_model)

        grid = self.img_size // self.patch_size
        self.grid_size = grid
        self.num_patches = grid * grid

        self.proj = nn.Conv2d(
            in_chans,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_chans, img_size, img_size)

        Returns:
            (B, num_patches, d_model)
        """
        if x.ndim != 4:
            raise ValueError("x must be rank-4: (B, C, H, W).")
        b, c, h, w = x.shape
        if c != self.in_chans or h != self.img_size or w != self.img_size:
            raise ValueError(
                f"x must have shape (B, {self.in_chans}, {self.img_size}, {self.img_size}), "
                f"got {(b, c, h, w)}."
            )

        out = self.proj(x)  # (B, d_model, grid, grid)
        out = out.flatten(2).transpose(1, 2).contiguous()  # (B, num_patches, d_model)
        return out


class CoordinateEmbedding(nn.Module):
    """
    Maps continuous coordinates to d_model: (B, N, in_dim) -> (B, N, d_model).
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 512,
        d_model: int = 512,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.d_model = int(d_model)

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model),
        )

    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: (B, N, in_dim)

        Returns:
            (B, N, d_model)
        """
        if coords.ndim != 3:
            raise ValueError("coords must be rank-3: (B, N, in_dim).")
        if coords.shape[-1] != self.in_dim:
            raise ValueError(f"coords last dim must be in_dim={self.in_dim}, got {coords.shape[-1]}.")
        return self.net(coords)


def _sinusoidal_axis_encoding(pos_1d: Tensor, d_half: int) -> Tensor:
    """
    Standard sinusoidal PE for one axis: (N,) positions -> (N, d_half).

    Uses pairs [sin, cos, sin, cos, ...] with frequencies 1 / 10000^(2i/d_half).
    """
    if d_half % 2 != 0:
        raise ValueError(f"d_half ({d_half}) must be even for sin/cos pairs.")
    n = pos_1d.shape[0]
    device = pos_1d.device
    dtype = pos_1d.dtype

    pe = torch.zeros(n, d_half, device=device, dtype=torch.float32)
    position = pos_1d.reshape(n, 1).float()
    half_dim = d_half // 2
    div_term = torch.exp(
        torch.arange(0, d_half, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d_half)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.to(dtype)


class SpatialPositionalEncoding(nn.Module):
    """
    Fixed 2D sinusoidal positional encoding for an H x W token grid (CV row-major order).

    Uses `torch.meshgrid(..., indexing="ij")` so the first index is row (y) and the second
    is column (x), consistent with (B, C, H, W) images. Patches are flattened in row-major
    order: token index t = i * W + j for row i, column j.
    """

    def __init__(self, d_model: int = 512, grid_size: int = 14):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4 for y/x sin-cos halves.")
        self.d_model = int(d_model)
        self.grid_size = int(grid_size)
        self.num_patches = self.grid_size * self.grid_size

        d_half = self.d_model // 2

        rows = torch.arange(self.grid_size)
        cols = torch.arange(self.grid_size)
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")  # each (H, W)

        # Row-major flatten: matches PatchEmbedding's flatten(2).transpose(1, 2)
        pos_y = grid_y.reshape(-1).to(dtype=torch.float32)
        pos_x = grid_x.reshape(-1).to(dtype=torch.float32)

        pe_y = _sinusoidal_axis_encoding(pos_y, d_half)  # (num_patches, d_half)
        pe_x = _sinusoidal_axis_encoding(pos_x, d_half)  # (num_patches, d_half)
        pe = torch.cat([pe_y, pe_x], dim=-1)  # (num_patches, d_model)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, num_patches, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, num_patches, d_model)

        Returns:
            x + fixed spatial PE, same shape as x.
        """
        if x.ndim != 3:
            raise ValueError("x must be rank-3: (B, S, D).")
        b, s, d = x.shape
        if s != self.num_patches or d != self.d_model:
            raise ValueError(
                f"x must have shape (B, {self.num_patches}, {self.d_model}), got {(b, s, d)}."
            )
        return x + self.pe


if __name__ == "__main__":
    torch.manual_seed(0)

    pe_mod = PatchEmbedding(img_size=224, patch_size=16, in_chans=3, d_model=512)
    img = torch.randn(2, 3, 224, 224)
    tok = pe_mod(img)
    print("PatchEmbedding:", img.shape, "->", tok.shape)
    assert tok.shape == (2, 196, 512)

    ce = CoordinateEmbedding(in_dim=2, hidden_dim=512, d_model=512)
    coords = torch.randn(2, 10, 2)
    ce_out = ce(coords)
    print("CoordinateEmbedding:", coords.shape, "->", ce_out.shape)
    assert ce_out.shape == (2, 10, 512)

    spe = SpatialPositionalEncoding(d_model=512, grid_size=14)
    z = torch.randn(2, 196, 512)
    z2 = spe(z)
    print("SpatialPositionalEncoding:", z.shape, "->", z2.shape)
    assert z2.shape == (2, 196, 512)
    assert spe.pe.requires_grad is False
    assert "pe" not in dict(spe.named_parameters())

    # Row/col convention: indexing="ij" -> grid_y[i,j]==i, grid_x[i,j]==j
    rows = torch.arange(spe.grid_size)
    cols = torch.arange(spe.grid_size)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
    for i, j in [(0, 0), (3, 7), (13, 13)]:
        assert int(grid_y[i, j].item()) == i
        assert int(grid_x[i, j].item()) == j

    # Row-0 tokens (indices 0..W-1) vs col-0 tokens (indices i*W) should differ as encodings
    w = spe.grid_size
    pe_flat = spe.pe.squeeze(0)  # (196, 512)
    row0_slice = pe_flat[0:w]  # first row of patch grid
    col0_slice = pe_flat[torch.arange(0, spe.num_patches, w)]  # first column
    assert not torch.allclose(row0_slice, col0_slice)

    chain = spe(pe_mod(img))
    print("PatchEmbedding -> SpatialPositionalEncoding:", img.shape, "->", chain.shape)
    assert chain.shape == (2, 196, 512)

    print("embeddings.py smoke test OK.")
