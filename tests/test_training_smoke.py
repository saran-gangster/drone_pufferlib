from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pufferlib = pytest.importorskip("pufferlib")
pufferl = pytest.importorskip("pufferlib.pufferl")

from drone_pufferlib.training.adapter import configure_pufferl_runtime, make_vector_env
from drone_pufferlib.training.common import build_model, build_puffer_train_config, load_checkpoint, save_checkpoint
from drone_pufferlib.utils.config import load_yaml

configure_pufferl_runtime(pufferl)


def test_training_smoke(tmp_path: Path):
    env_cfg = load_yaml("configs/env.yaml")
    env_cfg["difficulty"] = "easy"
    model_cfg = {"channels": [32, 64, 64], "hidden_size": 512}
    device = torch.device("cpu")
    model = build_model(env_cfg, model_cfg, device)
    vector_env = make_vector_env(
        env_cfg,
        seeds=[0, 1],
        multiprocessing=False,
        difficulty="easy",
        batch_size=2,
    )
    puffer_train_cfg = build_puffer_train_config(
        {
            "seed": 7,
            "device": "cpu",
            "artifacts_dir": str(tmp_path),
            "training": {
                "total_env_steps": 16,
                "eval_interval": 8,
                "rollout_length": 4,
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "update_epochs": 1,
                "clip_coef": 0.2,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "entropy_coef": 0.01,
                "minibatch_size": 4,
            },
        },
        num_envs=2,
        rollout_length=4,
    )
    trainer = pufferl.PuffeRL(puffer_train_cfg, vector_env, model)
    trainer.evaluate()
    trainer.train()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, trainer.optimizer, int(trainer.global_step), {"test": True})
    load_checkpoint(checkpoint_path, model, trainer.optimizer)
    assert trainer.global_step > 0
    trainer.close()
