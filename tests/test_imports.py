"""Smoke tests: package imports and SpatialProber forward shapes."""

from __future__ import annotations

import torch


def test_import_package():
    import spatial_probing_transformer as spt

    assert hasattr(spt, "SpatialProber")


def test_spatial_prober_forward():
    from spatial_probing_transformer.prober import SpatialProber

    m = SpatialProber(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    q = torch.rand(1, 4, 2)
    logits, attn = m(x, q)
    assert logits.shape == (1, 4, 5)
    assert attn.shape == (1, 8, 4, 196)
