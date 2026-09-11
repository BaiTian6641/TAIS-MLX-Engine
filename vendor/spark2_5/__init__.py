"""Spark2.5: dense, per-head gated attention with per-layer-type rotary embeddings."""
from .model import Model, ModelArgs, sanitize

__all__ = ['Model', 'ModelArgs', 'sanitize']
