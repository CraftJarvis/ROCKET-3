"""Compatibility imports for the former top-level Minecraft callbacks."""

from rocket3.minecraft.geometry import (
    draw_voxel_wireframe,
    get_voxel_convex_hull,
    is_voxel_occluded,
    voxel_map_from_info,
    voxel_to_screen_and_mask,
)
from rocket3.minecraft.tasks import (
    BlockConfig,
    ResetConfig,
    RocketOnlineCallback,
    SpawnBlocks,
)

__all__ = [
    "BlockConfig",
    "ResetConfig",
    "RocketOnlineCallback",
    "SpawnBlocks",
    "draw_voxel_wireframe",
    "get_voxel_convex_hull",
    "is_voxel_occluded",
    "voxel_map_from_info",
    "voxel_to_screen_and_mask",
]
