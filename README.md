# Drone PufferLib

Vision-only reinforcement learning navigation in procedural mazes using Gymnasium, PyTorch, and a PufferLib-oriented training adapter.

## Entry Points

```bash
python -m drone_pufferlib.training.train --config configs/train.yaml
python -m drone_pufferlib.training.evaluate --config configs/eval.yaml --checkpoint artifacts/checkpoints/latest.pt
python -m drone_pufferlib.tools.manual_play --config configs/env.yaml
```

## What This Includes

- A seeded procedural maze generator with easy and medium difficulty presets
- A custom first-person RGB navigation environment
- A CNN actor-critic policy and PPO training loop
- Evaluation, rollout video export, TensorBoard logging, and tests

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Notes

- The project is CPU-first and defaults to 24 parallel environments on a 12-thread machine.
- The `pufferlib` dependency is isolated behind a local adapter so the environment and trainer can evolve independently.
