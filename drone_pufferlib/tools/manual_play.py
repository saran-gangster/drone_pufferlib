from __future__ import annotations

import argparse

import cv2
import numpy as np

from drone_pufferlib.training.adapter import make_env
from drone_pufferlib.utils.config import load_yaml


KEY_TO_ACTION = {
    ord("w"): 0,
    ord("a"): 1,
    ord("d"): 2,
}


def resolve_seed(seed_arg: int | None, config_seed: int) -> int:
    return config_seed if seed_arg is None else seed_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    current_seed = resolve_seed(args.seed, int(config["seed"]))
    env = make_env(config, seed=current_seed)
    obs, info = env.reset(seed=current_seed)
    episode_return = 0.0

    while True:
        frame = env.render_first_person()
        topdown = env.render_topdown()
        topdown = cv2.resize(topdown, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([frame, topdown], axis=1)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imshow("Drone PufferLib Manual Play", combined)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        if key not in KEY_TO_ACTION:
            continue
        obs, reward, terminated, truncated, info = env.step(KEY_TO_ACTION[key])
        episode_return += reward
        print({"reward": reward, "episode_return": episode_return, **info})
        if terminated or truncated:
            print("Episode finished. Resetting.")
            current_seed += 1
            obs, info = env.reset(seed=current_seed)
            episode_return = 0.0

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
