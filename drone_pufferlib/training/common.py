from __future__ import annotations

import math
from pathlib import Path

import torch

from drone_pufferlib.models.cnn_policy import CnnActorCritic
from drone_pufferlib.utils.config import load_env_config, load_yaml


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(env_config: dict, model_config: dict, device: torch.device) -> CnnActorCritic:
    frame_stack = int(env_config.get("frame_stack", 1))
    obs_shape = (3 * frame_stack, int(env_config["observation_size"]), int(env_config["observation_size"]))
    model = CnnActorCritic(
        obs_shape=obs_shape,
        action_dim=3,
        channels=list(model_config["channels"]),
        hidden_size=int(model_config["hidden_size"]),
    )
    return model.to(device)


def load_train_bundle(config_path: str, overrides: list[str] | None = None):
    from drone_pufferlib.utils.config import apply_overrides

    train_cfg = apply_overrides(load_yaml(config_path), overrides)
    env_cfg = load_env_config(train_cfg["env"]["base_config"], train_cfg["env"]["difficulty"])
    return train_cfg, env_cfg


def build_puffer_train_config(train_cfg: dict, *, num_envs: int, rollout_length: int, world_size: int = 1) -> dict:
    batch_size = num_envs * rollout_length
    global_batch_size = batch_size * max(world_size, 1)
    checkpoint_interval = max(1, int(train_cfg["training"]["eval_interval"]) // global_batch_size)
    effective_minibatch_size = min(int(train_cfg["training"]["minibatch_size"]), batch_size)
    effective_minibatch_size -= effective_minibatch_size % rollout_length
    effective_minibatch_size = max(rollout_length, effective_minibatch_size)
    device = train_cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "env": str(train_cfg.get("run_name", "drone_navigation")),
        "seed": int(train_cfg["seed"]),
        "torch_deterministic": True,
        "cpu_offload": False,
        "device": device,
        "optimizer": "adam",
        "anneal_lr": True,
        "precision": "float32",
        "total_timesteps": int(math.ceil(int(train_cfg["training"]["total_env_steps"]) / max(world_size, 1))),
        "learning_rate": float(train_cfg["training"]["learning_rate"]),
        "gamma": float(train_cfg["training"]["gamma"]),
        "gae_lambda": float(train_cfg["training"]["gae_lambda"]),
        "update_epochs": int(train_cfg["training"]["update_epochs"]),
        "clip_coef": float(train_cfg["training"]["clip_coef"]),
        "vf_coef": float(train_cfg["training"]["value_coef"]),
        "vf_clip_coef": float(train_cfg["training"]["clip_coef"]),
        "max_grad_norm": float(train_cfg["training"]["max_grad_norm"]),
        "ent_coef": float(train_cfg["training"]["entropy_coef"]),
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "data_dir": str(Path(train_cfg["artifacts_dir"]) / "puffer_checkpoints"),
        "checkpoint_interval": checkpoint_interval,
        "batch_size": batch_size,
        "minibatch_size": effective_minibatch_size,
        "max_minibatch_size": effective_minibatch_size,
        "bptt_horizon": rollout_length,
        "compile": False,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "prio_alpha": 0.0,
        "prio_beta0": 0.0,
        "use_rnn": False,
    }


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
