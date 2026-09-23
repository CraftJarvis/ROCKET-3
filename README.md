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
| `model.py` | Cross-view policy architecture and checkpoint loader |
| `cross_view_dataset.py` | MineStudio adapter for cross-view pretraining data |
| `rocket_callbacks.py` | Minecraft task generation and reward callbacks |
| `online_configs/rocket_log.py` | Multi-task online RL configuration |
| `run_online.py` | Online RL entry point |

This snapshot contains the model and the Minecraft online training path. Dataset
generation, the complete pretraining pipeline, evaluation in other environments,
and the exact dependency versions used for the paper are not included here. The
published model repository and a local MineStudio checkpoint may use different
formats; the training entry point below expects a local MineStudio checkpoint.

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
`online_configs/rocket_log.py` for your compute resources and task setup before
training. The command has not been verified end to end in a clean environment;
the dependency versions and simulator setup still need a reproducibility pass.

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
