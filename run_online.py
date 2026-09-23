"""Start MineStudio's ROCKET-3 online PPO path (paper Sec. 4 / Appendix B).

MineStudio owns distributed rollout collection and PPO optimization; this
entry point supplies the local policy, environment and training configuration.
"""

import argparse
from functools import partial
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a ROCKET-3 state dictionary or MineStudio checkpoint.",
    )
    parser.add_argument(
        "--ray-address",
        default="localhost:9899",
        help="Address of the Ray cluster used by MineStudio (default: localhost:9899).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Validate the checkpoint, then launch rollout workers and trainer."""
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    from minestudio.online.rollout.start_manager import start_rolloutmanager
    from minestudio.online.trainer.start_trainer import start_trainer
    from omegaconf import OmegaConf

    from rocket3.training import ONLINE_CONFIG, make_minecraft_env, make_policy

    config_path = Path(__file__).resolve().parent / "rocket3" / "training.py"
    online_cfg = OmegaConf.create(ONLINE_CONFIG)
    policy_factory = partial(make_policy, str(checkpoint))
    # MineStudio records the exact configuration source with training outputs.
    config_source = config_path.read_text(encoding="utf-8")

    start_rolloutmanager(
        policy_factory, make_minecraft_env, online_cfg, address=args.ray_address
    )
    start_trainer(policy_factory, make_minecraft_env, online_cfg, config_source)


if __name__ == "__main__":
    main()
