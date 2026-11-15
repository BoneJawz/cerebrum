"""Minimal CPU-first neural architecture research core."""

from .config import ARCHITECTURE_VERSION, ModelConfig
from .network import DistributedVisualAssociator

__all__ = ["ARCHITECTURE_VERSION", "ModelConfig", "DistributedVisualAssociator"]
