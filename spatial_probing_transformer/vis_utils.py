"""
Visualization utilities for cross-attention maps over ViT-style patch grids.

Requires matplotlib when calling visualize_attention.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

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
    predicted_class: Optional[int] = None,
    actual_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    dpi: int = 120,
) -> str:
    """
    3-panel visualization for cross-attention probing:
      - Left: the question (image + query marker)
      - Middle: the reasoning (mean cross-attention heatmap + token grid)
      - Right: the answer (predicted vs actual class)

    Args:
        image: (3, img_size, img_size), float in [0, 1].
        attn: (num_heads, grid_size * grid_size) — one query, one batch element.
        save_path: Output PNG path.
        grid_size: Patch grid side length (default 14 for 224 / 16).
        img_size: Spatial size (must match image).
        query_xy: Optional (x, y) in [0, 1]; x = column, y = row (image coords).
        predicted_class: Optional predicted class index.
        actual_class: Optional ground-truth class index.
        class_names: Optional list mapping class index to display name.
        title: Optional figure title.
        dpi: Figure DPI for savefig.

    Returns:
        Resolved path string written to disk.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

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

    attn_grid = attn.reshape(num_heads, grid_size, grid_size)  # (H, 14, 14)

    # Mean over heads: (14, 14) -> upsample to (224, 224) for overlay
    attn_mean = attn_grid.mean(dim=0, keepdim=False)  # (14, 14)
    attn_mean_up = F.interpolate(
        attn_mean[None, None, ...],
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)  # (224, 224)

    img_np = image.permute(1, 2, 0).clamp(0, 1).numpy()

    if class_names is None:
        class_names = ["white", "red", "green", "blue", "yellow"]

    fig, (ax_q, ax_r, ax_a) = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Spatial Probing: Fusing Coordinates with Visual Tokens (Struct2D-inspired)",
        fontsize=14,
        y=0.98,
    )
    if title is not None:
        fig.text(0.5, 0.93, title, ha="center", va="top", fontsize=10)

    px_plot: Optional[float] = None
    py_plot: Optional[float] = None
    if query_xy is not None:
        qx, qy = query_xy
        px = int(torch.tensor(qx * img_size).clamp(0, img_size - 1).item())
        py = int(torch.tensor(qy * img_size).clamp(0, img_size - 1).item())
        px_plot = float(px)
        py_plot = float(py)
        # High-contrast marker: black halo + white outline circle
        ax_q.add_patch(Circle((px_plot, py_plot), radius=14, fill=False, edgecolor="black", linewidth=4.0))
        ax_q.add_patch(Circle((px_plot, py_plot), radius=14, fill=False, edgecolor="white", linewidth=2.5))
        ax_r.add_patch(Circle((px_plot, py_plot), radius=14, fill=False, edgecolor="black", linewidth=4.0))
        ax_r.add_patch(Circle((px_plot, py_plot), radius=14, fill=False, edgecolor="white", linewidth=2.5))

    # Left: question
    ax_q.imshow(img_np)
    ax_q.set_title("Query: What is at this (x,y)?")
    ax_q.axis("off")

    # Middle: reasoning (mean attention + subtle patch grid)
    ax_r.imshow(img_np)
    ax_r.imshow(attn_mean_up.numpy(), cmap="viridis", alpha=0.55)
    patch_size = img_size // grid_size
    for i in range(1, grid_size):
        ax_r.axhline(i * patch_size, color="white", linewidth=0.5, alpha=0.3)
        ax_r.axvline(i * patch_size, color="white", linewidth=0.5, alpha=0.3)
    ax_r.set_title("Transformer's Focus (Cross-Attention)")
    ax_r.axis("off")

    # Right: answer
    ax_a.axis("off")
    ax_a.set_title("Prediction Result")
    if predicted_class is None or actual_class is None:
        ax_a.text(0.5, 0.5, "(prediction unavailable)", ha="center", va="center", fontsize=14)
    else:
        pred_name = class_names[predicted_class] if 0 <= predicted_class < len(class_names) else str(predicted_class)
        act_name = class_names[actual_class] if 0 <= actual_class < len(class_names) else str(actual_class)
        correct = predicted_class == actual_class
        ax_a.text(0.5, 0.65, f"Predicted: {pred_name}", ha="center", va="center", fontsize=16)
        ax_a.text(0.5, 0.50, f"Actual:    {act_name}", ha="center", va="center", fontsize=16)
        ax_a.text(
            0.5,
            0.32,
            "Correct" if correct else "Incorrect",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=("green" if correct else "red"),
        )
        if query_xy is not None:
            ax_a.text(0.5, 0.18, f"query_xy=({query_xy[0]:.3f}, {query_xy[1]:.3f})", ha="center", va="center", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.90])
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
            predicted_class=1,
            actual_class=1,
            title="smoke test",
        )
        print("saved:", p)
        assert os.path.exists(p), p
        print("vis_utils.py smoke test OK.")
