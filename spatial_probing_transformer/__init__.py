"""
Spatial-Probing Transformer: custom attention, ViT-style embeddings, and coordinate probing.
"""

from .block import TransformerBlock
from .data import PointProbeDataset, queries_to_patch_indices
from .embeddings import CoordinateEmbedding, PatchEmbedding, SpatialPositionalEncoding
from .model import MultiHeadAttention, scaled_dot_product_attention
from .prober import SpatialProber
from .vis_utils import visualize_attention

__all__ = [
    "CoordinateEmbedding",
    "MultiHeadAttention",
    "PatchEmbedding",
    "PointProbeDataset",
    "SpatialPositionalEncoding",
    "SpatialProber",
    "TransformerBlock",
    "queries_to_patch_indices",
    "scaled_dot_product_attention",
    "visualize_attention",
]
