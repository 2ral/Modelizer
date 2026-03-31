# This file implements the legacy model architecture from Modelizer v1.
# based on https://pytorch.org/tutorials/beginner/translation_transformer.html and adapted for Modelizer

from .legacy import LegacyModel, LegacyConfig

__all__ = [
    "LegacyModel",
    "LegacyConfig",
]
