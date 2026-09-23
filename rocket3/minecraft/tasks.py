"""Task spawning, goal selection and reward callbacks for Minecraft."""

import random
import re
from dataclasses import dataclass

import cv2
import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback

from .geometry import (
    is_voxel_occluded,
    project_voxel_to_screen,
    projected_voxel_hull,
    world_voxels_from_info,
)

TARGET_SCAN_RADIUS = 7
COMPLETION_DISTANCE = 5
TARGET_POSITION_TOLERANCE = 0.5
STEP_COST = 0.05
DISTANCE_REWARD_SCALE = 10
MINE_PENALTY = 0.5
SUCCESS_REWARD = 5


@dataclass
class BlockSpawnConfig:
    """Random block spawning limits relative to the reset location."""

    names: str | list[str]
    number: int
    options: int
    range_x: list[int]  # [-10, 10]
    range_y: list[int]  # [0, 2]
    range_z: list[int]  # [-10, 10]


class SpawnTargetBlocks(MinecraftCallback):
    """Create varied interaction targets when a Minecraft world resets."""

    def __init__(self, block_configs: list[BlockSpawnConfig]):
        self.block_configs = block_configs

    def after_reset(self, sim, obs, info):
        for block_config in self.block_configs:
            if isinstance(block_config.names, str):
                names = [block_config.names]
            else:
                names = block_config.names
            names = random.sample(names, block_config.options)
            for name in names:
                for _ in range(block_config.number):
                    x = random.randint(block_config.range_x[0], block_config.range_x[1])
                    y = random.randint(block_config.range_y[0], block_config.range_y[1])
                    z = random.randint(block_config.range_z[0], block_config.range_z[1])
                    obs, reward, done, info = sim.env.execute_cmd(
                        f"/setblock ~{x} ~{y} ~{z} minecraft:{name}"
                    )
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info


@dataclass
class CrossViewTaskConfig:
    """Candidate goal cameras, voxel scan bounds and target type filter.

    Each view is ``(dx, dy, dz, yaw, pitch)`` relative to the reset position.
    """

    views: list[tuple]
    voxel_range: list[float]
    re_pattern: str
    obj_id: int


class CrossViewTaskCallback(MinecraftCallback):
    """Generate one masked cross-view target and score its interaction.

    This online example currently detects a target *block* disappearing. It is
    narrower than the paper's Approach, Break, Interact and Hunt task mix.
    """

    def __init__(self, reset_configs: list[CrossViewTaskConfig]):
        self.task_configs = reset_configs

    def _capture_goal_camera(self, sim, spawn_position, view):
        """Teleport and step the simulator until the candidate view settles."""
        dx, dy, dz, yaw, pitch = view
        spawn_x, spawn_y, spawn_z = spawn_position
        for _ in range(10):
            command = (
                f"/tp @p {spawn_x + dx} {spawn_y + dy} {spawn_z + dz} {yaw} {pitch}"
            )
            obs, _, _, info = sim.env.execute_cmd(command)
            action = sim.env.action_space.no_op()
            action.update(
                {
                    "mobs": self.task_config.voxel_range.copy(),
                    "voxels": self.task_config.voxel_range.copy(),
                }
            )
            obs, _, _, info = sim.env.step(action)
            obs, info = sim._wrap_obs_info(obs, info)
        return obs, info

    def _visible_targets(self, sim, info, view_id, pitch, yaw):
        """Find matching visible blocks and build coarse goal masks."""
        player_pos = np.array([info["player_pos"][axis] for axis in ("x", "y", "z")])
        nearby_voxels = world_voxels_from_info(info)
        screen_width, screen_height = 640, 360
        candidates = []
        for position, block_name in nearby_voxels:
            voxel = np.array(position)
            if not re.match(self.task_config.re_pattern, block_name):
                continue
            screen_coords, screen_corners = project_voxel_to_screen(
                voxel,
                player_pos,
                pitch,
                yaw,
                screen_width=screen_width,
                screen_height=screen_height,
                fov_y=70,
            )
            if screen_coords is None or is_voxel_occluded(
                voxel, nearby_voxels, player_pos, pitch, yaw, step=0.5
            ):
                continue

            hull = projected_voxel_hull(screen_corners)
            if hull is None:
                continue
            # The paper's Appendix A uses SAM2; this example projects a cube.
            target_mask = np.zeros((screen_height, screen_width), dtype=np.uint8)
            cv2.fillPoly(target_mask, [hull], color=1)
            candidates.append(
                {
                    "view_id": view_id,
                    "voxel": voxel,
                    "string": block_name,
                    "coords": screen_coords,
                    "corners": screen_corners,
                    "image": info["pov"].copy(),
                    "obj_mask": target_mask,
                    "rsz_image": cv2.resize(
                        info["pov"], sim.obs_size, interpolation=cv2.INTER_LINEAR
                    ),
                    "rsz_obj_mask": cv2.resize(
                        target_mask, sim.obs_size, interpolation=cv2.INTER_NEAREST
                    ),
                    "obj_id": self.task_config.obj_id,
                    "done": False,
                }
            )
        return candidates

    def after_reset(self, sim, obs, info):
        """Capture candidate O_g views, choose a target, then restore O_1."""
        self.task_config = random.choice(self.task_configs)
        spawn_position = tuple(info["player_pos"][axis] for axis in ("x", "y", "z"))
        candidates = []
        shuffled_views = self.task_config.views.copy()
        random.shuffle(shuffled_views)
        for view_id, view in enumerate(shuffled_views):
            # Use the first candidate camera with at least one visible target.
            if candidates:
                break
            obs, info = self._capture_goal_camera(sim, spawn_position, view)
            candidates = self._visible_targets(sim, info, view_id, view[4], view[3])

        spawn_x, spawn_y, spawn_z = spawn_position
        back_cmd = f"/tp @p {spawn_x} {spawn_y} {spawn_z} 0 0"
        for _ in range(5):
            obs, _, _, info = sim.env.execute_cmd(back_cmd)
        obs, info = sim._wrap_obs_info(obs, info)

        if candidates:
            self.target = random.choice(candidates)
            # Draw an overlay for inspection; the policy receives the separate
            # resized RGB image and binary mask saved in target.
            goal_image = self.target["image"].copy()
            annotated_goal = goal_image.copy()
            x, y = (
                int(self.target["coords"][0]),
                int(self.target["coords"][1]),
            )
            hull = projected_voxel_hull(self.target["corners"])
            cv2.fillPoly(annotated_goal, [hull], color=(0, 255, 0))
            cv2.circle(annotated_goal, (x, y), 3, (255, 255, 0), -1)
            info["reset_cross_view_list"] = [(goal_image, annotated_goal)]
        else:
            # Keep the observation schema valid even if no goal was found.
            self.target = {
                "rsz_image": np.zeros(
                    (sim.obs_size[1], sim.obs_size[0], 3), dtype=np.uint8
                ),
                "rsz_obj_mask": np.zeros(
                    (sim.obs_size[1], sim.obs_size[0]), dtype=np.uint8
                ),
                "obj_id": -1,
                "done": False,
            }
        px, py, pz = (
            info["player_pos"]["x"],
            info["player_pos"]["y"],
            info["player_pos"]["z"],
        )
        self.previous_distance = (
            self.compute_distance(np.array([px, py, pz]), self.target["voxel"])
            if "voxel" in self.target
            else 0.0
        )
        self.previous_info = info.copy()
        obs = self.build_cross_view(obs)
        return obs, info

    def build_cross_view(self, obs):
        """Expose fixed O_g/M_g/event conditioning at every rollout step."""
        obs["cross_view"] = {
            "cross_view_image": self.target["rsz_image"].copy(),
            "cross_view_obj_id": self.target["obj_id"],
            "cross_view_obj_mask": self.target["rsz_obj_mask"].copy(),
        }
        return obs

    def before_step(self, sim, action):
        # Scan nearby voxels so after_step can verify target-block removal.
        radius = TARGET_SCAN_RADIUS
        action["voxels"] = [-radius, radius] * 3
        return action

    def compute_distance(self, player_pos: np.ndarray, voxel_pos: np.ndarray):
        return np.linalg.norm(player_pos - voxel_pos)

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        """Apply this example's shaped reward and terminal target check.

        Unlike the paper's binary outcome reward (Appendix A), this callback
        rewards progress toward a block, penalizes mining and gives +5 when
        the chosen nearby block disappears.
        """
        obs = self.build_cross_view(obs)
        if "view_id" not in self.target or self.target["done"]:
            return obs, reward, terminated, truncated, info

        # Dense distance progress, with a small per-step cost.
        px, py, pz = (
            info["player_pos"]["x"],
            info["player_pos"]["y"],
            info["player_pos"]["z"],
        )
        distance = self.compute_distance(np.array([px, py, pz]), self.target["voxel"])
        reward += (
            self.previous_distance - distance - STEP_COST
        ) / DISTANCE_REWARD_SCALE
        self.previous_distance = distance

        # Penalize mined blocks, including accidental destruction.
        for block_name, count in info["mine_block"].items():
            if count > self.previous_info["mine_block"].get(block_name, 0):
                reward -= MINE_PENALTY
        self.previous_info = info.copy()

        # Target absence counts as success only within the nearby voxel scan;
        # beyond five blocks, an unobserved target must not count as removed.
        if distance > COMPLETION_DISTANCE:
            return obs, reward, terminated, truncated, info
        target_still_present = False
        for voxel in info["voxels"]:
            world_position = np.array(
                [voxel["x"] + px, voxel["y"] + py, voxel["z"] + pz]
            )
            if np.all(
                np.abs(world_position - self.target["voxel"])
                <= TARGET_POSITION_TOLERANCE
            ):
                target_still_present = True
        if not target_still_present:
            reward += SUCCESS_REWARD
            self.target["done"] = True
            terminated = True
        return obs, reward, terminated, truncated, info

    def __repr__(self):
        return f"CrossViewTaskCallback(config={self.task_configs})"


# Older scripts can keep importing the original public names.
BlockConfig = BlockSpawnConfig
SpawnBlocks = SpawnTargetBlocks
ResetConfig = CrossViewTaskConfig
RocketOnlineCallback = CrossViewTaskCallback
