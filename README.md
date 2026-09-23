<h1 align="center">ROCKET-3</h1>

<p align="center"><strong>Scalable Multi-Task Reinforcement Learning for Generalizable<br>
Spatial Intelligence in Visuomotor Agents</strong></p>

<p align="center">
  <a href="https://phython96.github.io/">Shaofei Cai*</a> ·
  <a href="https://muzhancun.github.io/">Zhancun Mu*</a> ·
  Haiwen Xia · Bowei Zhang ·
  <a href="https://liuanji.github.io/">Anji Liu</a> · Yitao Liang<br>
  <a href="https://craftjarvis.github.io/">CraftJarvis</a>
</p>

<p align="center">
  <a href="https://craftjarvis.github.io/ROCKET-3/">Project</a> ·
  <a href="https://arxiv.org/abs/2507.23698">Paper</a> ·
  <a href="https://huggingface.co/papers/2507.23698">Hugging Face paper</a> ·
  <a href="https://huggingface.co/CraftJarvis/ROCKET-3-1.5x">Model</a>
</p>

<p align="center">
  <img src="stitched_video_framerate_hq.gif" alt="ROCKET-3 demonstration" width="100%">
</p>

Research code for the ROCKET-3 paper. The
[project website](https://craftjarvis.github.io/ROCKET-3/) describes the method
and results.

## Repository scope

| Path | Purpose |
| --- | --- |
| `rocket3/policy.py` | Cross-view policy architecture and checkpoint loader |
| `rocket3/dataset.py` | MineStudio adapter for cross-view pretraining data |
| `rocket3/minecraft/geometry.py` | Voxel projection and visibility helpers |
| `rocket3/minecraft/tasks.py` | Minecraft task generation and reward callbacks |
| `rocket3/training.py` | Online RL configuration and environment/policy factories |
| `run_online.py` | Online RL entry point |
| `smoke_test.py` | CPU checkpoint load and one-step inference check |

The former top-level modules (`model.py`, `cross_view_dataset.py`,
`rocket_callbacks.py`, and `online_configs/rocket_log.py`) re-export their old
symbols for existing scripts. New code should import from the `rocket3` package.

This snapshot contains the model and the Minecraft online training path. Dataset
generation, the complete pretraining pipeline, evaluation in other environments,
and the exact dependency versions used for the paper are not included here. The
model loader accepts a ROCKET-3 state dictionary or a training checkpoint
containing `state_dict` and `hyper_parameters.model`. It infers the view token
count and previous-action setting from a state dictionary. Other architecture
settings must match the defaults in `rocket3/policy.py` or be supplied to the loader.

## Setup and training entry point

The training code uses [MineStudio](https://github.com/CraftJarvis/MineStudio)
directly for the policy base class, Minecraft simulator, dataset, rollout manager,
and trainer. MineStudio requires Python 3.10+, Java 8 for the simulator, and a
rendering setup such as Xvfb or VirtualGL. The default configuration is for
distributed, multi-GPU training and expects a running Ray cluster at
`localhost:9899`; use `--ray-address` for a different cluster. Install the Python
dependencies with:

```bash
python -m pip install -r requirements.txt
```

From the repository root, supply a trusted local checkpoint:

```bash
python run_online.py --checkpoint /path/to/rocket3.ckpt \
  --ray-address localhost:9899
```

The checkpoint is deliberately excluded from Git. Update
`rocket3/training.py` for your compute resources and task setup before
training. On Linux with Python 3.10, Java 8, Xvfb and CUDA, a clean install of
`requirements.txt` loaded the checkpoint, completed a Minecraft reset and two
policy-driven steps, then completed one reduced MineStudio PPO update using one
rollout worker, one environment, one trainer worker and four-step fragments.
The default multi-GPU, 4000-iteration configuration and training convergence
have not been validated.

To check a local checkpoint without starting Minecraft or Ray:

```bash
python smoke_test.py --checkpoint /path/to/rocket3.ckpt
# On a CUDA machine:
python smoke_test.py --checkpoint /path/to/rocket3.ckpt --device cuda
```

MineStudio 1.1.6 pins OpenCV 4.8.0.74. PyPI has [withdrawn that build and
published a libwebp advisory](https://pypi.org/project/opencv-python-headless/4.8.0.74/);
updating MineStudio's dependency declaration is still needed for a clean
upgrade to a fixed OpenCV release.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for formatting and static checks.

## Citation

```bibtex
@misc{cai2025scalable,
  title={Scalable Multi-Task Reinforcement Learning for Generalizable Spatial Intelligence in Visuomotor Agents},
  author={Shaofei Cai and Zhancun Mu and Haiwen Xia and Bowei Zhang and Anji Liu and Yitao Liang},
  year={2025},
  eprint={2507.23698},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```
