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
    Each sample: white canvas, K solid-colored patches (classes 1-4), N random queries in [0,1]^2.

    Returns:
        image: (3, img_size, img_size) float in [0, 1]
        queries: (n_queries, 2) float in [0, 1]
        labels: (n_queries,) long class indices in [0, num_classes-1]
    """

    COLORS = {
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.0, 1.0),
        4: (1.0, 1.0, 0.0),
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

    def _make_image(self, g: torch.Generator) -> Tensor:
        """Random colored patches on white background. Returns (3, H, W)."""
        img = torch.ones(3, self.img_size, self.img_size, dtype=torch.float32)

        perm = torch.randperm(self.num_patches, generator=g)
        chosen = perm[: self.k_patches]

        for idx in chosen:
            cls = int(torch.randint(1, 5, (1,), generator=g).item())
            r, gc, b = self.COLORS[cls]
            row = int(idx) // self.grid_size
            col = int(idx) % self.grid_size
            rs, re = row * self.patch_size, (row + 1) * self.patch_size
            cs, ce = col * self.patch_size, (col + 1) * self.patch_size
            img[0, rs:re, cs:ce] = r
            img[1, rs:re, cs:ce] = gc
            img[2, rs:re, cs:ce] = b

        return img

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

        img = self._make_image(g)
        queries = torch.rand(self.n_queries, 2, generator=g, dtype=torch.float32)
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
    hist = torch.bincount(lab, minlength=5)
    print("label histogram (first sample):", hist.tolist())
    assert img.shape == (3, 224, 224)
    assert q.shape == (16, 2)
    assert lab.shape == (16,)
    print("data.py smoke test OK.")
