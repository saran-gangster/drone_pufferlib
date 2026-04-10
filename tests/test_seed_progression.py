from drone_pufferlib.envs.navigation_env import NavigationEnv
from drone_pufferlib.utils.config import load_yaml


def test_reset_without_explicit_seed_advances_episode_seed():
    env = NavigationEnv(load_yaml("configs/env.yaml"), seed=17)
    _, info_a = env.reset()
    _, info_b = env.reset()
    assert info_a["episode_seed"] == 17
    assert info_b["episode_seed"] == 18


def test_explicit_seed_then_implicit_reset_advances_from_explicit_seed():
    env = NavigationEnv(load_yaml("configs/env.yaml"), seed=17)
    _, info_a = env.reset(seed=23)
    _, info_b = env.reset()
    _, info_c = env.reset()
    assert info_a["episode_seed"] == 23
    assert info_b["episode_seed"] == 24
    assert info_c["episode_seed"] == 25
