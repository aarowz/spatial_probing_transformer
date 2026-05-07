"""
Synthetic Point-Probe dataset: colored patches on white background, query labels from pixel class.

Coordinates are in [0, 1]^2: x = horizontal (column), y = vertical (row), matching image indexing.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


def rgb_to_class(rgb: Tensor) -> int:
    """Map an RGB vector (3,) in [0,1] to class id 0..4."""
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    if r > 0.9 and g > 0.9 and b > 0.9:
        return 0  # white background
    if r > 0.9 and g < 0.1 and b < 0.1:
        return 1  # red
    if r < 0.1 and g > 0.9 and b < 0.1:
        return 2  # green
    if r < 0.1 and g < 0.1 and b > 0.9:
        return 3  # blue
    if r > 0.9 and g > 0.9 and b < 0.1:
        return 4  # yellow
    # Fallback: nearest by L1 to prototypes
    protos = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    dist = (protos - rgb).abs().sum(dim=-1)
    return int(dist.argmin().item())


def queries_to_patch_indices(
    queries: Tensor,
    *,
    img_size: int = 224,
    grid_size: int = 14,
    patch_size: int = 16,
) -> Tensor:
    """
    Map normalized queries (B, N, 2) to flat patch indices in [0, grid_size**2).

    x -> column (width), y -> row (height).
    """
    x = queries[..., 0]
    y = queries[..., 1]
    px = (x * img_size).long().clamp(0, img_size - 1)
    py = (y * img_size).long().clamp(0, img_size - 1)
    col = px // patch_size
    row = py // patch_size
    return row * grid_size + col


class PointProbeDataset(Dataset):
    """
    Presentation Mode:
      - Each image contains exactly two 16x16 patch-aligned squares:
        - Red (class 1)
        - Blue (class 3)
      - The first query is always the exact center of the Red square.
      - Remaining queries are uniform random in [0,1]^2.

    Returns:
        image: (3, img_size, img_size) float in [0, 1]
        queries: (n_queries, 2) float in [0, 1]
        labels: (n_queries,) long class indices in [0, num_classes-1]
    """

    COLORS = {
        1: (1.0, 0.0, 0.0),  # red
        3: (0.0, 0.0, 1.0),  # blue
    }

    def __init__(
        self,
        length: int = 10_000,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        n_queries: int = 16,
        k_patches: int = 8,
        num_classes: int = 5,
        base_seed: int = 0,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")
        self.length = int(length)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.n_queries = int(n_queries)
        self.k_patches = int(k_patches)
        self.num_classes = int(num_classes)
        self.base_seed = int(base_seed)

        if self.k_patches > self.num_patches:
            raise ValueError("k_patches cannot exceed num_patches.")

    def __len__(self) -> int:
        return self.length

    def _make_image(self, g: torch.Generator) -> Tuple[Tensor, int, int]:
        """
        White background with exactly two colored squares (Red and Blue).

        Returns:
            img: (3, H, W)
            red_row, red_col: patch-grid coordinates of the Red square
        """
        img = torch.ones(3, self.img_size, self.img_size, dtype=torch.float32)

        perm = torch.randperm(self.num_patches, generator=g)
        red_idx = int(perm[0].item())
        blue_idx = int(perm[1].item())

        red_row, red_col = red_idx // self.grid_size, red_idx % self.grid_size
        blue_row, blue_col = blue_idx // self.grid_size, blue_idx % self.grid_size

        def fill_patch(row: int, col: int, rgb: Tuple[float, float, float]) -> None:
            rs, re = row * self.patch_size, (row + 1) * self.patch_size
            cs, ce = col * self.patch_size, (col + 1) * self.patch_size
            img[0, rs:re, cs:ce] = rgb[0]
            img[1, rs:re, cs:ce] = rgb[1]
            img[2, rs:re, cs:ce] = rgb[2]

        fill_patch(red_row, red_col, self.COLORS[1])
        fill_patch(blue_row, blue_col, self.COLORS[3])

        return img, red_row, red_col

    def _labels_for_queries(self, img: Tensor, queries: Tensor) -> Tensor:
        """Pixel class labels for each query; queries (N, 2)."""
        x = queries[:, 0]
        y = queries[:, 1]
        px = (x * self.img_size).long().clamp(0, self.img_size - 1)
        py = (y * self.img_size).long().clamp(0, self.img_size - 1)
        labels = torch.empty(self.n_queries, dtype=torch.long)
        for i in range(self.n_queries):
            rgb = img[:, py[i], px[i]]
            labels[i] = rgb_to_class(rgb)
        return labels

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        g = torch.Generator()
        g.manual_seed(self.base_seed + int(idx))

        img, red_row, red_col = self._make_image(g)

        # query[0] is the exact pixel center of the Red square (normalized to [0,1]).
        qx = (red_col * self.patch_size + self.patch_size / 2) / self.img_size
        qy = (red_row * self.patch_size + self.patch_size / 2) / self.img_size
        q0 = torch.tensor([qx, qy], dtype=torch.float32)

        queries_rest = torch.rand(self.n_queries - 1, 2, generator=g, dtype=torch.float32)
        queries = torch.cat([q0[None, :], queries_rest], dim=0)
        labels = self._labels_for_queries(img, queries)

        return img, queries, labels


def render_sample(dataset: PointProbeDataset, idx: int = 0) -> str:
    """Tiny debug string for one sample (no matplotlib dependency)."""
    img, q, lab = dataset[idx]
    return (
        f"idx={idx} image={tuple(img.shape)} queries={tuple(q.shape)} "
        f"labels_unique={torch.unique(lab).tolist()}"
    )


if __name__ == "__main__":
    ds = PointProbeDataset(length=100, n_queries=16, k_patches=8)
    img, q, lab = ds[0]
    print("sample shapes:", img.shape, q.shape, lab.shape)
    print(render_sample(ds, 0))
    print("presentation check: query[0] label =", int(lab[0].item()), "(expected 1=red)")
    hist = torch.bincount(lab, minlength=5)
    print("label histogram (first sample):", hist.tolist())
    assert img.shape == (3, 224, 224)
    assert q.shape == (16, 2)
    assert lab.shape == (16,)
    assert int(lab[0].item()) == 1
    print("data.py smoke test OK.")
