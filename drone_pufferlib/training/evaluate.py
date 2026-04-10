from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from drone_pufferlib.training.adapter import make_env
from drone_pufferlib.training.common import build_model, load_checkpoint, load_train_bundle, resolve_device
from drone_pufferlib.utils.config import apply_overrides, deep_merge, load_yaml
from drone_pufferlib.utils.seeding import set_global_seeds
from drone_pufferlib.utils.visualization import save_json, write_video


def evaluate_policy(env_config: dict, model, device, episodes: int, seed_start: int, difficulty: str, save_videos: bool = False, video_dir: str | Path | None = None, fixed_video_seeds: list[int] | None = None):
    was_training = model.training
    model.eval()
    rewards = []
    successes = []
    collisions = []
    timeouts = []
    lengths = []
    video_payload = []
    for episode in range(episodes):
        seed = seed_start + episode
        env = make_env(env_config, seed=seed, difficulty=difficulty)
        obs, _ = env.reset(seed=seed, options={"difficulty": difficulty})
        done = False
        total_reward = 0.0
        length = 0
        frames = []
        while not done:
            if save_videos and fixed_video_seeds and seed in fixed_video_seeds:
                frames.append(env.render_first_person())
            obs_tensor = torch.as_tensor(obs[None, ...], dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _, _ = model.act(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            total_reward += reward
            length += 1
        rewards.append(total_reward)
        successes.append(float(info["success"]))
        collisions.append(float(info["collision"]))
        timeouts.append(float(info["timeout"]))
        lengths.append(length)
        if save_videos and fixed_video_seeds and seed in fixed_video_seeds:
            frames.append(env.render_first_person())
            video_payload.append((seed, frames))
        env.close()

    if save_videos and video_dir is not None:
        for seed, frames in video_payload:
            write_video(Path(video_dir) / f"{difficulty}_seed_{seed}.mp4", frames)

    summary = {
        "episodes": episodes,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if timeouts else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
    if was_training:
        model.train()
    return summary


def evaluate_random_policy(env_config: dict, episodes: int, seed_start: int, difficulty: str) -> dict[str, float]:
    rewards = []
    successes = []
    collisions = []
    timeouts = []
    lengths = []
    for episode in range(episodes):
        seed = seed_start + episode
        env = make_env(env_config, seed=seed, difficulty=difficulty)
        obs, _ = env.reset(seed=seed, options={"difficulty": difficulty})
        done = False
        total_reward = 0.0
        length = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
            total_reward += reward
            length += 1
        rewards.append(total_reward)
        successes.append(float(info["success"]))
        collisions.append(float(info["collision"]))
        timeouts.append(float(info["timeout"]))
        lengths.append(length)
        env.close()
    return {
        "episodes": episodes,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "collision_rate": float(np.mean(collisions)) if collisions else 0.0,
        "timeout_rate": float(np.mean(timeouts)) if timeouts else 0.0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted.path=value")
    args = parser.parse_args()

    eval_cfg = apply_overrides(load_yaml(args.config), args.override)
    env_cfg = load_yaml(eval_cfg["env"]["base_config"])
    env_cfg["difficulty"] = eval_cfg["env"]["difficulty"]
    device = resolve_device(eval_cfg["device"])
    set_global_seeds(int(eval_cfg["seed"]))
    model = build_model(env_cfg, eval_cfg["model"], device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    artifacts_dir = Path(eval_cfg["artifacts_dir"])
    output_dir = artifacts_dir / "eval"
    fixed_seeds = list(eval_cfg["env"]["eval_fixed_seeds"])
    summary = {
        "random_easy": evaluate_random_policy(
            env_cfg,
            episodes=int(eval_cfg["evaluation"]["random_policy_episodes"]),
            seed_start=int(eval_cfg["env"]["eval_seed_start"]),
            difficulty="easy",
        ),
        "policy_easy": evaluate_policy(
            env_cfg,
            model,
            device,
            episodes=int(eval_cfg["evaluation"]["episodes"]),
            seed_start=int(eval_cfg["env"]["eval_seed_start"]),
            difficulty="easy",
            save_videos=bool(eval_cfg["evaluation"]["save_videos"]),
            video_dir=output_dir / "videos",
            fixed_video_seeds=fixed_seeds,
        ),
        "policy_medium": evaluate_policy(
            deep_merge(env_cfg, {"difficulty": "medium"}),
            model,
            device,
            episodes=int(eval_cfg["evaluation"]["medium_episodes"]),
            seed_start=int(eval_cfg["env"]["eval_seed_start"]) + 1000,
            difficulty="medium",
            save_videos=False,
        ),
    }
    save_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
