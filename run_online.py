"""Start ROCKET-3 online reinforcement learning with a local checkpoint."""

import argparse
from functools import partial
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a ROCKET-3 checkpoint in MineStudio format.",
    )
    parser.add_argument(
        "--ray-address",
        default="localhost:9899",
        help="Address of the Ray cluster used by MineStudio (default: localhost:9899).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    from minestudio.online.rollout.start_manager import start_rolloutmanager
    from minestudio.online.trainer.start_trainer import start_trainer
    from omegaconf import OmegaConf

    from online_configs.rocket_log import env_generator, online_dict, policy_generator

    config_path = Path(__file__).resolve().parent / "online_configs" / "rocket_log.py"
    online_cfg = OmegaConf.create(online_dict)
    policy_factory = partial(policy_generator, str(checkpoint))
    config_source = config_path.read_text(encoding="utf-8")

    start_rolloutmanager(
        policy_factory, env_generator, online_cfg, address=args.ray_address
    )
    start_trainer(policy_factory, env_generator, online_cfg, config_source)


if __name__ == "__main__":
    main()
