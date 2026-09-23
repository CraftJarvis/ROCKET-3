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
ruff check --select I --fix .
black .
```

Keep machine-specific paths, service addresses, credentials, checkpoints, and training
outputs out of commits. Pass local paths and service URLs through configuration.
Keep new examples and documentation aligned with the ROCKET-3 model and training
entry point. Document any additional experiments and their dependencies.
