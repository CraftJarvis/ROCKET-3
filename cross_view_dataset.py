"""Compatibility imports for the former top-level dataset module."""

from rocket3.dataset import (
    CrossViewDataModule,
    CrossViewDataset,
    CrossViewDrawFrameCallback,
    mask_to_bounding_box_batch,
)

__all__ = [
    "CrossViewDataModule",
    "CrossViewDataset",
    "CrossViewDrawFrameCallback",
    "mask_to_bounding_box_batch",
]
