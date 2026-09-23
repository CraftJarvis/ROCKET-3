import math
import random
import re
from dataclasses import dataclass
from math import floor
from typing import List, Tuple, Union

import cv2
import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback


def voxel_to_screen_and_mask(
    voxel_pos, player_pos, pitch, yaw, screen_width=640, screen_height=360, fov_y=70
):
    """
    将 Minecraft 中一个体素位置转换为玩家视角下的屏幕中心坐标 + 屏幕上的轮廓 mask。

    参数:
        voxel_pos: (x, y, z) 体素底面中心点的世界坐标
        player_pos: (x, y, z) 玩家脚下位置
        pitch: 玩家 pitch 角（上下，单位为度）
        yaw: 玩家 yaw 角（左右，单位为度）
        screen_width: 屏幕宽度（像素）
        screen_height: 屏幕高度（像素）
        fov_y: 垂直视野角度（单位为度）

    返回:
        (u, v): 屏幕中心坐标（像素）
        mask: 该 voxel 的投影轮廓，若不可见则为 None
    """

    # 玩家眼睛位置
    eye_height = 1.62
    px, py, pz = player_pos
    player_eye = np.array([px, py + eye_height, pz])

    # 定义体素的 8 个顶点（单位立方体，底面中心为 voxel_pos）
    vx, vy, vz = voxel_pos
    corners = []
    for dx in [-0.5, 0.5]:
        for dy in [0.0, 1.0]:
            for dz in [-0.5, 0.5]:
                corners.append(np.array([vx + dx, vy + dy, vz + dz]))

    # 角度转弧度
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    cos_yaw, sin_yaw = math.cos(-yaw_rad), math.sin(-yaw_rad)
    cos_pitch, sin_pitch = math.cos(-pitch_rad), math.sin(-pitch_rad)

    # 相机参数
    fov_y_rad = math.radians(fov_y)
    aspect_ratio = screen_width / screen_height
    fov_x_rad = 2 * math.atan(math.tan(fov_y_rad / 2) * aspect_ratio)

    # 投影函数
    def world_to_screen(point_world):
        rel = point_world - player_eye

        # 绕 Y 轴（Yaw）
        x1 = cos_yaw * rel[0] - sin_yaw * rel[2]
        z1 = sin_yaw * rel[0] + cos_yaw * rel[2]
        y1 = rel[1]

        # 绕 X 轴（Pitch）
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

    # 对每个角点投影
    screen_corners = [world_to_screen(corner) for corner in corners]

    # 去除不可见点
    screen_corners = [pt for pt in screen_corners if pt is not None]
    if len(screen_corners) == 0:
        return None, None

    # 返回中心点和 mask（2D 多边形点集）
    center_screen = world_to_screen(np.array([vx, vy + 0.5, vz]))
    return center_screen, screen_corners


def is_voxel_occluded(
    target_voxel, voxel_map: list, player_pos, pitch, yaw, max_distance=100, step=0.3
):
    # 构建体素索引：{(x, y, z): type}，使用离散表示
    voxel_dict = {}
    for key, val in voxel_map:
        voxel_dict[(floor(key[0]), floor(key[1]), floor(key[2]))] = val

    # 玩家眼睛位置
    eye_height = 1.62
    px, py, pz = player_pos
    origin = np.array([px, py + eye_height, pz])
    # 目标体素范围（底面中心 → 体积）
    # tx, ty, tz = floor(target_voxel[0]), floor(target_voxel[1]), floor(target_voxel[2])
    tx, ty, tz = target_voxel  #! 是否进行离散化，需要提前计算；在这里表示物体底面中心
    target_min = np.array([tx - 0.5, ty, tz - 0.5])
    target_max = np.array([tx + 0.5, ty + 1, tz + 0.5])

    for lam in [0.4, 0.5, 0.6]:
        # for lam in [0.5]:
        target_center = target_min * (1 - lam) + target_max * lam
        # 射线方向
        direction = target_center - origin
        distance = np.linalg.norm(direction)
        if distance > max_distance:
            return True
        direction /= distance

        # 射线采样
        steps = int(distance / step)
        for i in range(steps - 1, -1, -1):
            point = origin + direction * (i * step)
            # 否则检查当前点所在体素
            # vx, vy, vz = int(floor(point[0] + 0.5)), int(floor(point[1])), int(floor(point[2] + 0.5))
            vx, vy, vz = (
                int(floor(point[0])),
                int(floor(point[1])),
                int(floor(point[2])),
            )
            if (vx, vy, vz) == (floor(tx), floor(ty), floor(tz)):
                continue
            if voxel_dict.get((vx, vy, vz), "minecraft:grass") != "minecraft:grass":
                return True  # 有遮挡体素
    return False  # 没被遮挡


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


def get_voxel_convex_hull(screen_corners):
    """
    根据立方体的屏幕顶点投影，返回轮廓的凸包点集（顺时针）。

    参数:
        screen_corners: 长度为8的屏幕坐标点列表 [(u, v), ...]

    返回:
        hull_points: 凸包点集，按顺时针排列的 np.ndarray, 形状为 (N, 2)
    """
    # 过滤掉 None 或超出屏幕的点
    valid_points = [pt for pt in screen_corners if pt is not None]

    if len(valid_points) < 3:
        return None  # 无法形成凸包

    # 转换为 np 格式并 reshape
    pts = np.array(valid_points, dtype=np.int32).reshape(-1, 1, 2)

    # 获取凸包
    hull = cv2.convexHull(pts, returnPoints=True)  # shape (N,1,2)

    # reshape 为 (N,2)
    hull_points = hull.reshape(-1, 2)

    return hull_points


def voxel_map_from_info(info):
    """
    从 info 中获取 voxel_map
    """
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


@dataclass
class BlockConfig:
    names: Union[str, List[str]]
    number: int
    options: int
    range_x: List[int]  # [-10, 10]
    range_y: List[int]  # [0, 2]
    range_z: List[int]  # [-10, 10]


class SpawnBlocks(MinecraftCallback):

    def __init__(self, block_configs: List[BlockConfig]):
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
class ResetConfig:
    views: List[Tuple]
    voxel_range: List[float]
    re_pattern: str
    obj_id: int
    # reward_configs: Dict[str, Any]


class RocketOnlineCallback(MinecraftCallback):

    def __init__(self, reset_configs: List[ResetConfig]):
        self.reset_configs = reset_configs

    def after_reset(self, sim, obs, info):
        self.reset_config = random.choice(self.reset_configs)
        ox, oy, oz = (
            info["player_pos"]["x"],
            info["player_pos"]["y"],
            info["player_pos"]["z"],
        )
        cross_view_list = []
        candidates = []
        shuffled_views = self.reset_config.views.copy()
        random.shuffle(shuffled_views)
        for idx, (dx, dy, dz, yaw, pitch) in enumerate(shuffled_views):
            if len(candidates) > 0:
                break
            for _ in range(10):
                cmd = f"/tp @p {ox + dx} {oy + dy} {oz + dz} {yaw} {pitch}"
                obs, reward, done, info = sim.env.execute_cmd(cmd)
                action = sim.env.action_space.no_op()
                action.update(
                    {
                        "mobs": self.reset_config.voxel_range.copy(),
                        "voxels": self.reset_config.voxel_range.copy(),
                    }
                )
                obs, reward, done, info = sim.env.step(action)
                obs, info = sim._wrap_obs_info(obs, info)
            player_pos = np.array(
                [
                    info["player_pos"]["x"],
                    info["player_pos"]["y"],
                    info["player_pos"]["z"],
                ]
            )
            voxel_map = voxel_map_from_info(info)
            screen_width, screen_height = 640, 360
            # mask_image = info['pov'].copy()
            for key, string in voxel_map:
                voxel = np.array(key)
                if not re.match(self.reset_config.re_pattern, string):
                    continue
                screen_coords, screen_corners = voxel_to_screen_and_mask(
                    voxel,
                    player_pos,
                    pitch,
                    yaw,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    fov_y=70,
                )
                if screen_coords is None:
                    continue
                if is_voxel_occluded(
                    voxel, voxel_map, player_pos, pitch, yaw, step=0.5
                ):
                    continue

                hull = get_voxel_convex_hull(screen_corners)
                if hull is None:
                    continue
                segmemtation = np.zeros((screen_height, screen_width), dtype=np.uint8)
                cv2.fillPoly(segmemtation, [hull], color=(1))

                candidates.append(
                    {
                        "view_id": idx,
                        "voxel": voxel,
                        "string": string,
                        "coords": screen_coords,
                        "corners": screen_corners,
                        "image": info["pov"].copy(),
                        "obj_mask": segmemtation,
                        "rsz_image": cv2.resize(
                            info["pov"].copy(),
                            sim.obs_size,
                            interpolation=cv2.INTER_LINEAR,
                        ),
                        "rsz_obj_mask": cv2.resize(
                            segmemtation, sim.obs_size, interpolation=cv2.INTER_NEAREST
                        ),
                        "obj_id": self.reset_config.obj_id,
                        "done": False,
                    }
                )

                # x, y =  (int(screen_coords[0]), int(screen_coords[1]))
                # # draw_voxel_wireframe(mask_image, screen_corners, color=(255, 255, 255), thickness=1)
                # hull = get_voxel_convex_hull(screen_corners)
                # cv2.fillPoly(mask_image, [hull], color=(0, 255, 0))
                # cv2.circle(mask_image, (x, y), 3, (255, 255, 0), -1)

            # cross_view.append((info['pov'].copy(), mask_image)) #! save all cross-view images

        back_cmd = f"/tp @p {ox} {oy} {oz} 0 0"
        for i in range(5):
            obs, reward, done, info = sim.env.execute_cmd(back_cmd)
        obs, info = sim._wrap_obs_info(obs, info)

        if len(candidates) > 0:
            self.lucky_voxel = random.choice(candidates)  #! 解决空列表的问题
            mask_image = self.lucky_voxel["image"]
            x, y = (
                int(self.lucky_voxel["coords"][0]),
                int(self.lucky_voxel["coords"][1]),
            )
            hull = get_voxel_convex_hull(self.lucky_voxel["corners"])
            cv2.fillPoly(mask_image, [hull], color=(0, 255, 0))
            cv2.circle(mask_image, (x, y), 3, (255, 255, 0), -1)
            cross_view_list.append((self.lucky_voxel["image"], mask_image))
            info["reset_cross_view_list"] = cross_view_list
        else:
            print("初始化 cross-view image 为空")
            self.lucky_voxel = {
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
        self.last_distance = (
            self.compute_distance(np.array([px, py, pz]), self.lucky_voxel["voxel"])
            if "voxel" in self.lucky_voxel
            else 0.0
        )
        self.last_info = info.copy()
        obs = self.build_cross_view(obs)
        return obs, info

    def build_cross_view(self, obs):
        obs["cross_view"] = {
            "cross_view_image": self.lucky_voxel["rsz_image"].copy(),
            "cross_view_obj_id": self.lucky_voxel["obj_id"],
            "cross_view_obj_mask": self.lucky_voxel["rsz_obj_mask"].copy(),
        }
        return obs

    def before_step(self, sim, action):
        action["voxels"] = [-7, 7, -7, 7, -7, 7]  # 破坏方块时，物体一定在很附近的地方
        return action

    def compute_distance(self, player_pos: np.ndarray, voxel_pos: np.ndarray):
        distance = np.linalg.norm(player_pos - voxel_pos)
        return distance

    def after_step(self, sim, obs, reward, terminated, truncated, info):
        obs = self.build_cross_view(obs)
        if "view_id" not in self.lucky_voxel or self.lucky_voxel["done"]:
            return obs, reward, terminated, truncated, info

        #! 1. add distance reward
        px, py, pz = (
            info["player_pos"]["x"],
            info["player_pos"]["y"],
            info["player_pos"]["z"],
        )
        distance = self.compute_distance(
            np.array([px, py, pz]), self.lucky_voxel["voxel"]
        )
        reward += (self.last_distance - distance - 0.05) / 10
        self.last_distance = distance

        #! 2. add breaking reward
        for k, v in info["mine_block"].items():
            if v - self.last_info["mine_block"].get(k, 0) > 0:
                reward -= 0.5
        self.last_info = info.copy()

        #! 3. add task completion reward
        # 如果距离小于等于 5, 那么方块信息可以被获取到 info['voxels']，如果 xyz 相差均在 0.5 以内，则是目标方块
        if distance > 5:
            return obs, reward, terminated, truncated, info
        find = False
        for voxel in info["voxels"]:
            abs_voxel_pos = np.array(
                [voxel["x"] + px, voxel["y"] + py, voxel["z"] + pz]
            )
            if np.all(np.abs(abs_voxel_pos - self.lucky_voxel["voxel"]) <= 0.5):
                find = True
        if not find:
            # 没找到意味着 voxel 造到破坏，任务完成
            reward += 5
            self.lucky_voxel["done"] = True
            terminated = True
        return obs, reward, terminated, truncated, info

    def __repr__(self):
        return f"RocketOnlineCallback(config={self.reset_configs})"
