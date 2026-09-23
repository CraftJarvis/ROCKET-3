# Code style

Use Python 3.10 or newer. Install the development tools with:

```bash
python -m pip install -r requirements-dev.txt
```

Before submitting a change, run:

```bash
ruff check .
black --check .
python -m compileall -q .
```

To format and sort imports:

```bash
ruff check --fix .
black .
```

Place policy and data code in `rocket3/`, Minecraft geometry and callbacks in
`rocket3/minecraft/`, and online training setup in `rocket3/training.py`.
The old top-level modules are compatibility shims; add new behavior to the
package rather than those shims.

Keep machine-specific paths, service addresses, credentials, checkpoints, and training
outputs out of commits. Pass local paths and service URLs through configuration.
Keep new examples and documentation aligned with the ROCKET-3 model and training
entry point. Document any additional experiments and their dependencies.
