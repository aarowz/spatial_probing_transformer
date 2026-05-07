"""
Visualization utilities for cross-attention maps over ViT-style patch grids.

Requires matplotlib when calling visualize_attention.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def visualize_attention(
    image: Tensor,
    attn: Tensor,
    save_path: Union[str, os.PathLike[str]],
    *,
    grid_size: int = 14,
    img_size: int = 224,
    query_xy: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    dpi: int = 120,
) -> str:
    """
    Plot the input image (left) and per-head attention overlays (right grid).

    Args:
        image: (3, img_size, img_size), float in [0, 1].
        attn: (num_heads, grid_size * grid_size) — one query, one batch element.
        save_path: Output PNG path.
        grid_size: Patch grid side length (default 14 for 224 / 16).
        img_size: Spatial size (must match image).
        query_xy: Optional (x, y) in [0, 1]; x = column, y = row (image coords).
        title: Optional figure title.
        dpi: Figure DPI for savefig.

    Returns:
        Resolved path string written to disk.
    """
    import matplotlib.pyplot as plt

    path_str = os.fspath(save_path)
    out_dir = os.path.dirname(path_str)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image must be (3, H, W), got {tuple(image.shape)}.")
    if int(image.shape[1]) != img_size or int(image.shape[2]) != img_size:
        raise ValueError(
            f"image spatial dims must be ({img_size}, {img_size}), got {(image.shape[1], image.shape[2])}."
        )

    if attn.ndim != 2:
        raise ValueError(f"attn must be (num_heads, S), got {tuple(attn.shape)}.")
    num_heads, seq = attn.shape
    expected = grid_size * grid_size
    if seq != expected:
        raise ValueError(f"attn last dim must be {expected} (= grid_size**2), got {seq}.")

    image = image.detach().float().cpu()
    attn = attn.detach().float().cpu()

    attn_grid = attn.reshape(num_heads, grid_size, grid_size)
    attn_up = F.interpolate(
        attn_grid.unsqueeze(1),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    img_np = image.permute(1, 2, 0).clamp(0, 1).numpy()

    # Layout: one tall axes for image (left), num_heads in ceil(H/4) x 4 grid (right)
    ncols_right = min(4, num_heads)
    nrows_right = (num_heads + ncols_right - 1) // ncols_right
    fig_w = 3.0 * (1 + ncols_right)
    fig_h = 3.0 * max(2, nrows_right)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(nrows_right, ncols_right + 1, width_ratios=[1.4] + [1.0] * ncols_right)

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img_np)
    ax_img.set_title("input")
    ax_img.axis("off")

    px_plot: Optional[float] = None
    py_plot: Optional[float] = None
    if query_xy is not None:
        qx, qy = query_xy
        px = int(torch.tensor(qx * img_size).clamp(0, img_size - 1).item())
        py = int(torch.tensor(qy * img_size).clamp(0, img_size - 1).item())
        px_plot = float(px)
        py_plot = float(py)
        ax_img.plot(px_plot, py_plot, "rx", markersize=8, markeredgewidth=2)

    for h in range(num_heads):
        r = h // ncols_right
        c = 1 + (h % ncols_right)
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img_np)
        ax.imshow(attn_up[h].numpy(), cmap="viridis", alpha=0.5)
        ax.set_title(f"head {h}")
        ax.axis("off")
        if px_plot is not None and py_plot is not None:
            ax.plot(px_plot, py_plot, "rx", markersize=6, markeredgewidth=2)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(path_str, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return path_str


if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    image = torch.rand(3, 224, 224)
    attn = torch.softmax(torch.randn(8, 196), dim=-1)

    with tempfile.TemporaryDirectory() as tmp:
        save_path = os.path.join(tmp, "attn_vis.png")
        p = visualize_attention(
            image,
            attn,
            save_path,
            query_xy=(0.5, 0.5),
            title="smoke test",
        )
        print("saved:", p)
        assert os.path.exists(p), p
        print("vis_utils.py smoke test OK.")
