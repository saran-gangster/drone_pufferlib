from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from drone_pufferlib.envs.navigation_env import NavigationEnv


@dataclass
class AdapterInfo:
    backend_name: str
    pufferlib_available: bool
    pufferlib_version: str | None
    pufferl_available: bool


def get_adapter_info() -> AdapterInfo:
    spec = importlib.util.find_spec("pufferlib")
    if spec is None:
        return AdapterInfo("gymnasium", False, None, False)
    module = importlib.import_module("pufferlib")
    try:
        importlib.import_module("pufferlib.pufferl")
        pufferl_available = True
    except Exception:
        pufferl_available = False
    return AdapterInfo("pufferlib", True, getattr(module, "__version__", None), pufferl_available)


def make_env(config: dict[str, Any], seed: int, difficulty: str | None = None) -> NavigationEnv:
    env_config = deepcopy(config)
    if difficulty is not None:
        env_config["difficulty"] = difficulty
    return NavigationEnv(env_config, seed=seed)


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_return = 0.0
        self.episode_length = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.episode_return = 0.0
        self.episode_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_length += 1
        if terminated or truncated:
            info = dict(info)
            info["episode_return"] = float(self.episode_return)
            info["episode_length"] = int(self.episode_length)
        return obs, reward, terminated, truncated, info


def require_pufferlib():
    if importlib.util.find_spec("pufferlib") is None:
        raise RuntimeError(
            "pufferlib is required for training. Install it after torch with "
            "`pip install --no-build-isolation pufferlib==3.0.0`."
        )
    pufferlib = importlib.import_module("pufferlib")
    emulation = importlib.import_module("pufferlib.emulation")
    vector = importlib.import_module("pufferlib.vector")
    return pufferlib, emulation, vector


def configure_pufferl_runtime(pufferl_module) -> None:
    # Some machines expose nvcc even when PufferLib was built with the CPU-only
    # advantage kernel. Force the safe fallback so training still uses the GPU
    # for the policy without requiring a matching CUDA toolkit build.
    pufferl_module.ADVANTAGE_CUDA = False


def make_puffer_env(
    config: dict[str, Any],
    difficulty: str | None = None,
    render_mode: str = "rgb_array",
    episode_seed: int | None = None,
    buf=None,
    seed: int = 0,
):
    pufferlib, emulation, _ = require_pufferlib()
    env_seed = seed if episode_seed is None else int(episode_seed)
    env_config = deepcopy(config)
    env_config["render_mode"] = render_mode
    env = make_env(env_config, seed=env_seed, difficulty=difficulty)
    env = EpisodeStatsWrapper(env)
    return emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=env_seed)


def make_vector_env(
    config: dict[str, Any],
    seeds: list[int],
    multiprocessing: bool = True,
    difficulty: str | None = None,
    num_workers: int | None = None,
    batch_size: int | None = None,
    overwork: bool = False,
):
    _, _, vector = require_pufferlib()
    backend = vector.Multiprocessing if multiprocessing and len(seeds) > 1 else vector.Serial
    env_creators = [make_puffer_env for _ in seeds]
    env_args = [[] for _ in seeds]
    env_kwargs = [
        {
            "config": config,
            "difficulty": difficulty,
            "episode_seed": seed,
        }
        for seed in seeds
    ]
    kwargs = {}
    if backend is vector.Multiprocessing:
        kwargs["num_workers"] = num_workers if num_workers is not None else len(seeds)
        kwargs["batch_size"] = batch_size if batch_size is not None else len(seeds)
        kwargs["overwork"] = overwork
    return vector.make(
        env_creators,
        env_args=env_args,
        env_kwargs=env_kwargs,
        num_envs=len(seeds),
        backend=backend,
        seed=min(seeds) if seeds else 0,
        **kwargs,
    )
