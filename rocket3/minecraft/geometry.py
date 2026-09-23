"""Projection and voxel-visibility helpers for Minecraft goal views."""

import math
from math import floor

import cv2
import numpy as np

PLAYER_EYE_HEIGHT = 1.62


def project_voxel_to_screen(
    voxel_pos, player_pos, pitch, yaw, screen_width=640, screen_height=360, fov_y=70
):
    """Project a voxel center and cube corners into a goal camera image.

    Uses the camera geometry of paper Eqs. 5-8. ``voxel_pos`` is the cube's
    bottom center in world coordinates; ``player_pos`` is at the player's feet.
    Returns a pixel center (possibly ``None``) and visible projected corners;
    both are ``None`` when no corner is visible. The corners are geometry for a
    later coarse polygon mask, not a SAM mask.
    """

    # Paper Eq. 5 uses the camera origin U rather than the player's feet.
    px, py, pz = player_pos
    player_eye = np.array([px, py + PLAYER_EYE_HEIGHT, pz])

    # A unit cube supplies several projected points around the target.
    vx, vy, vz = voxel_pos
    corners = []
    for dx in [-0.5, 0.5]:
        for dy in [0.0, 1.0]:
            for dz in [-0.5, 0.5]:
                corners.append(np.array([vx + dx, vy + dy, vz + dz]))

    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    cos_yaw, sin_yaw = math.cos(-yaw_rad), math.sin(-yaw_rad)
    cos_pitch, sin_pitch = math.cos(-pitch_rad), math.sin(-pitch_rad)

    # Paper Eq. 6 derives the horizontal FoV from the vertical FoV and aspect.
    fov_y_rad = math.radians(fov_y)
    aspect_ratio = screen_width / screen_height
    fov_x_rad = 2 * math.atan(math.tan(fov_y_rad / 2) * aspect_ratio)

    def world_to_screen(point_world):
        """Transform a world point to image pixels; discard points behind/outside."""
        rel = point_world - player_eye

        # Camera yaw, then pitch, before perspective division (Eqs. 5-8).
        x1 = cos_yaw * rel[0] - sin_yaw * rel[2]
        z1 = sin_yaw * rel[0] + cos_yaw * rel[2]
        y1 = rel[1]

        y2 = cos_pitch * y1 - sin_pitch * z1
        z2 = sin_pitch * y1 + cos_pitch * z1
        x2 = x1

        if z2 <= 0:
            return None

        nx = x2 / (z2 * math.tan(fov_x_rad / 2))
        ny = y2 / (z2 * math.tan(fov_y_rad / 2))
        u = (nx + 1) / 2 * screen_width
        v = (1 - ny) / 2 * screen_height

        return (
            (screen_width - int(u), int(v))
            if (0 <= u <= screen_width and 0 <= v <= screen_height)
            else None
        )

    screen_corners = [world_to_screen(corner) for corner in corners]

    # Off-screen/behind-camera corners cannot contribute to the polygon.
    screen_corners = [pt for pt in screen_corners if pt is not None]
    if len(screen_corners) == 0:
        return None, None

    center_screen = world_to_screen(np.array([vx, vy + 0.5, vz]))
    return center_screen, screen_corners


def is_voxel_occluded(
    target_voxel, voxel_map: list, player_pos, pitch, yaw, max_distance=100, step=0.3
):
    """Approximate visibility by ray sampling through nearby simulator voxels.

    Three points inside the target cube are checked. This is a geometry filter
    for goal selection, not the pixel segmentation used by the paper's SAM2.
    """
    # Index block coordinates for repeated ray lookups.
    voxel_dict = {}
    for key, val in voxel_map:
        voxel_dict[(floor(key[0]), floor(key[1]), floor(key[2]))] = val

    px, py, pz = player_pos
    origin = np.array([px, py + PLAYER_EYE_HEIGHT, pz])
    # target_voxel is the bottom center of a unit cube.
    tx, ty, tz = target_voxel
    target_min = np.array([tx - 0.5, ty, tz - 0.5])
    target_max = np.array([tx + 0.5, ty + 1, tz + 0.5])

    for lam in [0.4, 0.5, 0.6]:
        target_center = target_min * (1 - lam) + target_max * lam
        direction = target_center - origin
        distance = np.linalg.norm(direction)
        if distance > max_distance:
            return True
        direction /= distance

        # Missing cells are treated as air/grass by this local heuristic.
        steps = int(distance / step)
        for i in range(steps - 1, -1, -1):
            point = origin + direction * (i * step)
            vx, vy, vz = (
                int(floor(point[0])),
                int(floor(point[1])),
                int(floor(point[2])),
            )
            if (vx, vy, vz) == (floor(tx), floor(ty), floor(tz)):
                continue
            if voxel_dict.get((vx, vy, vz), "minecraft:grass") != "minecraft:grass":
                return True
    return False


def draw_voxel_wireframe(img, projected_points, color=(0, 255, 0), thickness=1):
    """
    在图像 img 上画立方体轮廓线。

    参数:
        img: OpenCV 图像
        projected_points: 长度为 8 的点列表 [(u, v), ...]
        color: 线条颜色
        thickness: 线条粗细
    """
    if len(projected_points) != 8:
        return

    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),  # 左面
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),  # 右面
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # 连接左右
    ]

    for i, j in edges:
        pt1 = projected_points[i]
        pt2 = projected_points[j]
        if pt1 is not None and pt2 is not None:
            cv2.line(img, pt1, pt2, color, thickness)


def projected_voxel_hull(screen_corners):
    """Build a coarse 2-D silhouette from visible projected cube corners."""
    valid_points = [pt for pt in screen_corners if pt is not None]

    if len(valid_points) < 3:
        return None

    pts = np.array(valid_points, dtype=np.int32).reshape(-1, 1, 2)

    hull = cv2.convexHull(pts, returnPoints=True)  # (N, 1, 2)

    hull_points = hull.reshape(-1, 2)

    return hull_points


def world_voxels_from_info(info):
    """Convert simulator offsets for blocks/mobs to absolute world positions."""
    player_pos = np.array(
        [info["player_pos"]["x"], info["player_pos"]["y"], info["player_pos"]["z"]]
    )
    voxel_map = []
    for line in info["voxels"]:
        voxel_map.append(
            (
                (
                    line["x"] + player_pos[0],
                    line["y"] + player_pos[1],
                    line["z"] + player_pos[2],
                ),
                line["type"],
            )
        )
    for line in info["mobs"]:
        voxel_map.append(
            (
                (
                    line["x"] + player_pos[0],
                    line["y"] + player_pos[1],
                    line["z"] + player_pos[2],
                ),
                line["name"],
            )
        )

    return voxel_map


# Names used by the original top-level module remain importable.
voxel_to_screen_and_mask = project_voxel_to_screen
get_voxel_convex_hull = projected_voxel_hull
voxel_map_from_info = world_voxels_from_info
