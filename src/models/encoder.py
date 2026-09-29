"""Backward-compatible import for the canonical route encoder."""

from src.models.deepreta_system import RobustDeeprETAEncoder

DeeprETAEncoder = RobustDeeprETAEncoder

__all__ = ["DeeprETAEncoder", "RobustDeeprETAEncoder"]
