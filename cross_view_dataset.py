"""MineStudio data adapter for cross-view imitation learning (paper Sec. 4).

A sampled frame, preferably with a valid target mask, becomes the reference
view (O_g, M_g) for each target segment. The sample supplies per-frame object
labels for the auxiliary heads in paper Eq. 4. This is an adapter for existing
trajectories, not the paper's Minecraft task-synthesis pipeline.
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from minestudio.data.minecraft import RawDataModule, RawDataset
from minestudio.data.minecraft.callbacks import SegmentationDrawFrameCallback


def mask_to_bounding_box_batch(masks):
    """Return normalized ``(x_min, y_min, x_max, y_max)`` for ``(B,H,W)`` masks.

    An empty mask has the all-zero box used for an absent target.
    """
    n_rows = masks.shape[1]
    n_cols = masks.shape[2]
    # Reduce each mask to occupied rows and columns before finding extrema.
    rows = np.any(masks, axis=2)  # (b, 224)
    cols = np.any(masks, axis=1)  # (b, 224)

    y_min = np.argmax(rows, axis=1)  # (b,)
    y_max = n_rows - np.argmax(np.flip(rows, axis=1), axis=1) - 1  # (b,)

    x_min = np.argmax(cols, axis=1)  # (b,)
    x_max = n_cols - np.argmax(np.flip(cols, axis=1), axis=1) - 1  # (b,)

    # Coordinates are normalized independently by image width and height.
    x_min = x_min / n_cols
    x_max = x_max / n_cols
    y_min = y_min / n_rows
    y_max = y_max / n_rows

    bounding_boxes = np.stack((x_min, y_min, x_max, y_max), axis=1)
    # argmax on an empty mask would otherwise yield a spurious far edge.
    bounding_boxes[~rows.any(axis=1)] = 0.0

    return bounding_boxes


class CrossViewDataset(RawDataset):
    """Attach one sampled goal frame to each trajectory segment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_cross_view(
        self,
        episode: str,
        frame_range: Tuple[int, int],
        max_retries: int = 3,
        event_constrain=None,
    ) -> Tuple[int, Dict]:
        """Try reference frames in a segment; fall back to the last sampled one.

        Paper Sec. 4 samples a frame with a valid target mask for O_g/M_g.
        Real data can lack one, so the fallback is explicit here.
        """
        candidate_choices = list(range(frame_range[0], frame_range[1] + 1))
        for i in range(max_retries):
            frame_id = random.choice(candidate_choices)
            candidate_choices.remove(frame_id)
            frame = self.kernel_manager.read(
                episode, frame_id, 1, 1, event_constrain=event_constrain
            )
            if (
                frame["segmentation"]["obj_mask"][0].sum() > 0
                or len(candidate_choices) == 0
            ):
                return frame_id, frame
        return frame_id, frame

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Build a temporal window with stable goal conditioning and labels."""
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        episode, relative_idx = self.locate_item(idx)
        start = max(0, relative_idx * self.win_len)
        item = self.kernel_manager.read(episode, start, self.win_len, self.skip_frame)

        # The same event/frame range reuses one reference frame, keeping the
        # goal fixed across its window rather than resampling at every step.
        cross_view = {
            "cross_view_image": [],
            "cross_view_obj_id": [],
            "cross_view_obj_mask": [],
            "cross_view_point": [],
            "cross_view_event": [],
            "cross_view_frame_id": [],
        }
        cross_view_mapping = {}
        for wid, frame_range in enumerate(item["segmentation"]["frame_range"]):
            frame_range_key = f"{frame_range[0]}_{frame_range[1]}"
            if frame_range[0] == -1:
                # No target: use -1 event ID plus blank O_g/M_g sentinels.
                cross_view["cross_view_image"].append(np.zeros_like(item["image"][wid]))
                cross_view["cross_view_obj_id"].append(-1)
                cross_view["cross_view_obj_mask"].append(
                    np.zeros_like(item["segmentation"]["obj_mask"][wid])
                )
                cross_view["cross_view_point"].append((-1, -1))
                cross_view["cross_view_event"].append("")
                cross_view["cross_view_frame_id"].append(-1)
                continue
            if frame_range_key not in cross_view_mapping:
                cross_view_frame_range = (
                    frame_range[0],
                    min(frame_range[1], self.episodes_with_length[episode] - 1),
                )
                current_event = item["segmentation"]["event"][wid]
                cross_view_frame_id, single_frame_item = self.sample_cross_view(
                    episode,
                    cross_view_frame_range,
                    max_retries=5,
                    event_constrain=current_event,
                )
                cross_view_mapping[frame_range_key] = {
                    "cross_view_image": single_frame_item["image"][0],
                    "cross_view_obj_id": single_frame_item["segmentation"]["obj_id"][0],
                    "cross_view_obj_mask": single_frame_item["segmentation"][
                        "obj_mask"
                    ][0],
                    "cross_view_point": single_frame_item["segmentation"]["point"][0],
                    "cross_view_event": single_frame_item["segmentation"]["event"][0],
                    "cross_view_frame_id": single_frame_item["segmentation"][
                        "frame_id"
                    ][0],
                }
            cross_view["cross_view_image"].append(
                cross_view_mapping[frame_range_key]["cross_view_image"]
            )
            cross_view["cross_view_obj_id"].append(
                cross_view_mapping[frame_range_key]["cross_view_obj_id"]
            )
            cross_view["cross_view_obj_mask"].append(
                cross_view_mapping[frame_range_key]["cross_view_obj_mask"]
            )
            cross_view["cross_view_point"].append(
                cross_view_mapping[frame_range_key]["cross_view_point"]
            )
            cross_view["cross_view_event"].append(
                cross_view_mapping[frame_range_key]["cross_view_event"]
            )
            cross_view["cross_view_frame_id"].append(
                cross_view_mapping[frame_range_key]["cross_view_frame_id"]
            )

        for key in cross_view:
            if not isinstance(cross_view[key][0], str):
                cross_view[key] = np.stack(cross_view[key], axis=0)
        item["cross_view"] = cross_view

        # Preserve MineStudio's temporal validity mask under a stable key.
        for key in list(item.keys()):
            if key.endswith("mask"):
                mask = item.pop(key)
        item["mask"] = mask
        # A value of 1 keeps the previous action; 0 substitutes the learned
        # dropout token. Only about a quarter of positions retain it.
        item["prev_action_dropout"] = (
            np.random.uniform(0, 1, item["mask"].size) > 0.75
        ).astype(np.float32)
        item["segmentation"]["bbox"] = mask_to_bounding_box_batch(
            item["segmentation"]["obj_mask"]
        )

        item["text"] = "raw"
        item["timestamp"] = np.arange(start, start + self.win_len, self.skip_frame)
        item["episode"] = episode
        episode_samples = math.ceil(self.episodes_with_length[episode] / self.win_len)
        item["progress"] = f"{relative_idx}/{episode_samples}"
        item = self.to_tensor(item)
        return item


class CrossViewDrawFrameCallback(SegmentationDrawFrameCallback):
    """Stack the agent and annotated goal images for dataset inspection."""

    def draw_frames(
        self, frames: Union[np.ndarray, List], infos: Dict, sample_idx: int
    ) -> np.ndarray:
        cache_frames = []
        for frame_idx, frame in enumerate(frames):
            frame_up = frame.copy()
            cross_view_info = infos["cross_view"]
            frame_down = cross_view_info["cross_view_image"][sample_idx][
                frame_idx
            ].numpy()
            obj_id = cross_view_info["cross_view_obj_id"][sample_idx][frame_idx].item()
            obj_mask = cross_view_info["cross_view_obj_mask"][sample_idx][
                frame_idx
            ].numpy()
            point = (
                cross_view_info["cross_view_point"][sample_idx][frame_idx][1].item(),
                cross_view_info["cross_view_point"][sample_idx][frame_idx][0].item(),
            )
            event = cross_view_info["cross_view_event"][sample_idx][frame_idx]
            cross_view_frame_id = cross_view_info["cross_view_frame_id"][sample_idx][
                frame_idx
            ].item()
            frame_down = self.draw_frame(
                frame_down, point, obj_mask, obj_id, event, cross_view_frame_id
            )

            frame = np.concatenate([frame_up, frame_down], axis=0)
            cache_frames.append(frame)
        return cache_frames


class CrossViewDataModule(RawDataModule):
    """Use the cross-view adapter for both MineStudio data splits."""

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = CrossViewDataset(split="train", **self.data_params)
        self.val_dataset = CrossViewDataset(split="val", **self.data_params)
