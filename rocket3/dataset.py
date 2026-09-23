"""MineStudio data adapter for cross-view imitation learning (paper Sec. 4).

A sampled frame, preferably with a valid target mask, becomes the reference
view (O_g, M_g) for each target segment. The sample supplies per-frame object
labels for the auxiliary heads in paper Eq. 4. This is an adapter for existing
trajectories, not the paper's Minecraft task-synthesis pipeline.
"""

import math
import random

import numpy as np
import torch
from minestudio.data.minecraft import RawDataModule, RawDataset
from minestudio.data.minecraft.callbacks import SegmentationDrawFrameCallback


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Return normalized ``(x_min, y_min, x_max, y_max)`` for ``(B,H,W)`` masks.

    An empty mask has the all-zero box used for an absent target.
    """
    n_rows = masks.shape[1]
    n_cols = masks.shape[2]
    # Reduce each mask to occupied rows and columns before finding extrema.
    rows = np.any(masks, axis=2)  # (B, H)
    cols = np.any(masks, axis=1)  # (B, W)

    y_min = np.argmax(rows, axis=1)
    y_max = n_rows - np.argmax(np.flip(rows, axis=1), axis=1) - 1

    x_min = np.argmax(cols, axis=1)
    x_max = n_cols - np.argmax(np.flip(cols, axis=1), axis=1) - 1

    # Coordinates are normalized independently by image width and height.
    x_min = x_min / n_cols
    x_max = x_max / n_cols
    y_min = y_min / n_rows
    y_max = y_max / n_rows

    bounding_boxes = np.stack((x_min, y_min, x_max, y_max), axis=1)
    # argmax on an empty mask would otherwise yield a spurious far edge.
    bounding_boxes[~rows.any(axis=1)] = 0.0

    return bounding_boxes


class CrossViewTrajectoryDataset(RawDataset):
    """Attach one sampled goal frame to each trajectory segment."""

    def sample_cross_view(
        self,
        episode: str,
        frame_range: tuple[int, int],
        max_retries: int = 3,
        event_constrain=None,
    ) -> tuple[int, dict]:
        """Try reference frames in a segment; fall back to the last sampled one.

        Paper Sec. 4 samples a frame with a valid target mask for O_g/M_g.
        Real data can lack one, so the fallback is explicit here.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        candidate_choices = list(range(frame_range[0], frame_range[1] + 1))
        if not candidate_choices:
            raise ValueError(f"Empty cross-view frame range: {frame_range}")
        for _ in range(min(max_retries, len(candidate_choices))):
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Build a temporal window with stable goal conditioning and labels."""
        if not 0 <= idx < len(self):
            raise IndexError(f"Index <{idx}> out of range <{len(self)}>")
        episode, relative_idx = self.locate_item(idx)
        start = max(0, relative_idx * self.win_len)
        sample = self.kernel_manager.read(episode, start, self.win_len, self.skip_frame)

        # The same event/frame range reuses one reference frame, keeping the
        # goal fixed across its window rather than resampling at every step.
        goal_condition = {
            "cross_view_image": [],
            "cross_view_obj_id": [],
            "cross_view_obj_mask": [],
            "cross_view_point": [],
            "cross_view_event": [],
            "cross_view_frame_id": [],
        }
        reference_by_range = {}
        for step_index, frame_range in enumerate(sample["segmentation"]["frame_range"]):
            frame_range_key = tuple(frame_range)
            if frame_range[0] == -1:
                # No target: use -1 event ID plus blank O_g/M_g sentinels.
                goal_condition["cross_view_image"].append(
                    np.zeros_like(sample["image"][step_index])
                )
                goal_condition["cross_view_obj_id"].append(-1)
                goal_condition["cross_view_obj_mask"].append(
                    np.zeros_like(sample["segmentation"]["obj_mask"][step_index])
                )
                goal_condition["cross_view_point"].append((-1, -1))
                goal_condition["cross_view_event"].append("")
                goal_condition["cross_view_frame_id"].append(-1)
                continue
            if frame_range_key not in reference_by_range:
                cross_view_frame_range = (
                    frame_range[0],
                    min(frame_range[1], self.episodes_with_length[episode] - 1),
                )
                current_event = sample["segmentation"]["event"][step_index]
                _, reference_frame = self.sample_cross_view(
                    episode,
                    cross_view_frame_range,
                    max_retries=5,
                    event_constrain=current_event,
                )
                reference_by_range[frame_range_key] = {
                    "cross_view_image": reference_frame["image"][0],
                    "cross_view_obj_id": reference_frame["segmentation"]["obj_id"][0],
                    "cross_view_obj_mask": reference_frame["segmentation"]["obj_mask"][
                        0
                    ],
                    "cross_view_point": reference_frame["segmentation"]["point"][0],
                    "cross_view_event": reference_frame["segmentation"]["event"][0],
                    "cross_view_frame_id": reference_frame["segmentation"]["frame_id"][
                        0
                    ],
                }
            for key in goal_condition:
                goal_condition[key].append(reference_by_range[frame_range_key][key])

        for key in goal_condition:
            if not isinstance(goal_condition[key][0], str):
                goal_condition[key] = np.stack(goal_condition[key], axis=0)
        sample["cross_view"] = goal_condition

        # MineStudio RawDataset uses action_mask for temporal validity.
        validity_mask = sample.pop("action_mask")
        for key in list(sample.keys()):
            if key.endswith("mask"):
                sample.pop(key)
        sample["mask"] = validity_mask
        # A value of 1 keeps the previous action; 0 substitutes the learned
        # dropout token. Only about a quarter of positions retain it.
        sample["prev_action_dropout"] = (
            np.random.uniform(0, 1, sample["mask"].size) > 0.75
        ).astype(np.float32)
        sample["segmentation"]["bbox"] = masks_to_boxes(
            sample["segmentation"]["obj_mask"]
        )

        sample["text"] = "raw"
        sample["timestamp"] = np.arange(start, start + self.win_len, self.skip_frame)
        sample["episode"] = episode
        episode_samples = math.ceil(self.episodes_with_length[episode] / self.win_len)
        sample["progress"] = f"{relative_idx}/{episode_samples}"
        sample = self.to_tensor(sample)
        return sample


class CrossViewPreviewCallback(SegmentationDrawFrameCallback):
    """Stack the agent and annotated goal images for dataset inspection."""

    def draw_frames(
        self, frames: np.ndarray | list, infos: dict, sample_idx: int
    ) -> list[np.ndarray]:
        preview_frames = []
        for frame_idx, frame in enumerate(frames):
            agent_frame = frame.copy()
            goal_info = infos["cross_view"]
            annotated_goal = goal_info["cross_view_image"][sample_idx][
                frame_idx
            ].numpy()
            obj_id = goal_info["cross_view_obj_id"][sample_idx][frame_idx].item()
            obj_mask = goal_info["cross_view_obj_mask"][sample_idx][frame_idx].numpy()
            point = (
                goal_info["cross_view_point"][sample_idx][frame_idx][1].item(),
                goal_info["cross_view_point"][sample_idx][frame_idx][0].item(),
            )
            event = goal_info["cross_view_event"][sample_idx][frame_idx]
            cross_view_frame_id = goal_info["cross_view_frame_id"][sample_idx][
                frame_idx
            ].item()
            annotated_goal = self.draw_frame(
                annotated_goal, point, obj_mask, obj_id, event, cross_view_frame_id
            )

            frame = np.concatenate([agent_frame, annotated_goal], axis=0)
            preview_frames.append(frame)
        return preview_frames


class CrossViewDataModule(RawDataModule):
    """Use the cross-view adapter for both MineStudio data splits."""

    def setup(self, stage: str | None = None):
        self.train_dataset = CrossViewTrajectoryDataset(
            split="train", **self.data_params
        )
        self.val_dataset = CrossViewTrajectoryDataset(split="val", **self.data_params)


mask_to_bounding_box_batch = masks_to_boxes
CrossViewDataset = CrossViewTrajectoryDataset
CrossViewDrawFrameCallback = CrossViewPreviewCallback
