#!/usr/bin/env python3
"""
Train SpatialProber from the repository root.

Requires an editable install (``pip install -e .``) so ``spatial_probing_transformer`` is on the path.
"""

from __future__ import annotations

import torch

from spatial_probing_transformer.train import train


if __name__ == "__main__":
    torch.manual_seed(0)
    train()
