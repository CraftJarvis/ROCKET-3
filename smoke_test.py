"""Check a ROCKET-3 checkpoint with one synthetic CPU inference step.

This verifies policy loading and the paper's action/auxiliary output interface
without starting the Minecraft simulator or distributed PPO training.
"""

import argparse
from pathlib import Path

import torch

from model import BINARY_KEYS, load_cross_view_rocket


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    policy = load_cross_view_rocket(str(checkpoint)).eval()
    # O_t, O_g and M_g are all 224 x 224 in the model (paper Appendix D).
    observation = {
        "image": torch.zeros((1, 1, 224, 224, 3), dtype=torch.uint8),
        "cross_view": {
            "cross_view_image": torch.zeros((1, 1, 224, 224, 3), dtype=torch.uint8),
            "cross_view_obj_mask": torch.zeros((1, 1, 224, 224), dtype=torch.uint8),
            "cross_view_obj_id": torch.tensor([[2]], dtype=torch.long),
        },
    }
    if policy.use_prev_action:
        action = {
            key.replace("_", "."): torch.zeros((1, 1), dtype=torch.long)
            for key in BINARY_KEYS
        }
        action["camera"] = torch.zeros((1, 1, 2), dtype=torch.float32)
        observation["env_prev_action"] = action
        observation["prev_action_dropout"] = torch.ones((1, 1))

    with torch.inference_mode():
        latents, _ = policy(observation)
        sampled_action = policy.pi_head.sample(latents["pi_logits"], deterministic=True)

    assert latents["vpred"].shape[:2] == (1, 1)
    assert latents["point"].shape == (1, 1, 2)
    assert latents["bbox"].shape == (1, 1, 4)
    assert {"buttons", "camera"} <= sampled_action.keys()
    print(
        f"Checkpoint OK: {checkpoint.name}; "
        f"view tokens={policy.num_view_tokens}; "
        f"previous action={policy.use_prev_action}"
    )


if __name__ == "__main__":
    main()
