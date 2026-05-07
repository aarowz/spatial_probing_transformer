"""
Minimal training loop for SpatialProber on synthetic PointProbeDataset.

Artifacts (attention weights, plots) are written under ``<repo_root>/outputs/``.

Run from repo root after editable install::

    pip install -e .
    python scripts/train.py

Or::

    python -m spatial_probing_transformer.train
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import PointProbeDataset, queries_to_patch_indices
from .prober import SpatialProber
from .vis_utils import visualize_attention


# Repo root: parent of the ``spatial_probing_transformer`` package directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# PoC hyperparameters (no CLI)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_QUERIES = 16
STEPS = 500
LR = 3e-4
NUM_CLASSES = 5
LOG_EVERY = 50
PLOT_EVERY = 250  # save attention PNG every N steps (step-based; no epochs in this loop)
DATA_LEN = 50_000
IMG_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMG_SIZE // PATCH_SIZE


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    qs = torch.stack([b[1] for b in batch])
    labs = torch.stack([b[2] for b in batch])
    return imgs, qs, labs


def _save_attention_plot(
    model: SpatialProber,
    val_imgs: torch.Tensor,
    val_queries: torch.Tensor,
    plots_dir: str,
    *,
    step_tag: str,
    sample_idx: int = 0,
    query_idx: int = 0,
) -> str:
    """
    One validation image + one query: cross-attn (num_heads, 196) -> PNG under plots_dir.

    Uses ground-truth query coordinates for the red marker in visualize_attention.
    """
    model.eval()
    imgs = val_imgs.to(DEVICE)
    queries = val_queries.to(DEVICE)
    with torch.no_grad():
        _, attn = model(imgs, queries)  # (B, H, N, 196)

    b, q = sample_idx, query_idx
    image = val_imgs[b].detach().cpu().float()
    attn_h = attn[b, :, q, :].detach().cpu().float()  # (num_heads, 196)
    qx = float(val_queries[b, q, 0].item())
    qy = float(val_queries[b, q, 1].item())

    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, f"attn_{step_tag}.png")
    visualize_attention(
        image,
        attn_h,
        out_path,
        grid_size=GRID_SIZE,
        img_size=IMG_SIZE,
        query_xy=(qx, qy),
        title=f"cross-attn sample={b} query={q} ({step_tag})",
    )
    print(f"saved attention plot to {out_path}")
    return out_path


def train() -> None:
    outputs_dir = REPO_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = str(outputs_dir / "plots")

    ds = PointProbeDataset(
        length=DATA_LEN,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        n_queries=N_QUERIES,
        k_patches=8,
        num_classes=NUM_CLASSES,
        base_seed=42,
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Fixed first batch (no shuffle) for reproducible attention plots
    eval_loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    fixed_imgs, fixed_queries, fixed_labels = next(iter(eval_loader))

    model = SpatialProber(
        num_classes=NUM_CLASSES,
        d_model=512,
        num_heads=8,
        num_self_blocks=2,
        mlp_ratio=4,
        dropout=0.0,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
    ).to(DEVICE)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    it = iter(loader)
    for step in range(1, STEPS + 1):
        try:
            imgs, queries, labels = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, queries, labels = next(it)

        imgs = imgs.to(DEVICE)
        queries = queries.to(DEVICE)
        labels = labels.to(DEVICE)

        logits, _attn = model(imgs, queries)
        loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean().item()
            print(f"step {step:4d}  loss={loss.item():.4f}  acc={acc:.4f}")

        if step % PLOT_EVERY == 0:
            _save_attention_plot(model, fixed_imgs, fixed_queries, plots_dir, step_tag=f"step_{step}")
            model.train()

    # Validation-style batch + attention sanity check (fixed eval batch for metrics + plots)
    model.eval()
    with torch.no_grad():
        val_imgs = fixed_imgs.to(DEVICE)
        val_queries = fixed_queries.to(DEVICE)
        val_labels = fixed_labels.to(DEVICE)

        logits, attn = model(val_imgs, val_queries)
        val_loss = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), val_labels.reshape(-1))
        pred = logits.argmax(dim=-1)
        val_acc = (pred == val_labels).float().mean().item()
        print(f"val  loss={val_loss.item():.4f}  acc={val_acc:.4f}")

        attn_path = str(outputs_dir / "attn.pt")
        torch.save(attn.cpu(), attn_path)
        print(f"saved attention weights to {attn_path} shape={tuple(attn.shape)}")

        # Mean over heads -> predicted attended patch index per query
        attn_mean = attn.mean(dim=1)  # (B, N, 196)
        pred_patch = attn_mean.argmax(dim=-1)

        gt_patch = queries_to_patch_indices(
            val_queries,
            img_size=IMG_SIZE,
            grid_size=GRID_SIZE,
            patch_size=PATCH_SIZE,
        )

        b0 = 0
        print("sanity: query idx | gt_patch | pred_patch (mean-attn argmax)")
        for qi in range(min(8, N_QUERIES)):
            print(
                f"  q[{qi}]: gt={int(gt_patch[b0, qi].item())} pred={int(pred_patch[b0, qi].item())}"
            )

        match_frac = (pred_patch == gt_patch).float().mean().item()
        print(f"fraction pred_patch == gt_patch (mean over batch): {match_frac:.4f}")

        _save_attention_plot(model, fixed_imgs, fixed_queries, plots_dir, step_tag="final")


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
    print("train.py finished.")
