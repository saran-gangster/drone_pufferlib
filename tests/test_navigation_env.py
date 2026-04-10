import numpy as np

from drone_pufferlib.envs.navigation_env import NavigationEnv
from drone_pufferlib.utils.config import load_yaml


def make_env():
    cfg = load_yaml("configs/env.yaml")
    return NavigationEnv(cfg, seed=123)


def test_reset_returns_expected_shape():
    cfg = load_yaml("configs/env.yaml")
    env = NavigationEnv(cfg, seed=123)
    obs, info = env.reset(seed=123)
    assert obs.shape == (3 * int(cfg["frame_stack"]), 64, 64)
    assert obs.dtype == np.uint8
    assert {"success", "collision", "timeout", "distance_to_goal", "geodesic_distance", "difficulty", "episode_seed"} <= set(info.keys())


def test_turn_action_changes_heading():
    env = make_env()
    env.reset(seed=123)
    heading = env.state.heading
    env.step(1)
    assert env.state.heading != heading


def test_step_is_deterministic_given_seed_and_actions():
    env_a = make_env()
    env_b = make_env()
    actions = [1, 0, 0, 2, 0]
    obs_a, _ = env_a.reset(seed=17)
    obs_b, _ = env_b.reset(seed=17)
    np.testing.assert_array_equal(obs_a, obs_b)
    for action in actions:
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1:] == step_b[1:]


def test_render_methods_return_expected_shapes():
    env = make_env()
    env.reset(seed=123)
    rgb = env.render_first_person()
    topdown = env.render_topdown()
    assert rgb.shape == (64, 64, 3)
    assert topdown.ndim == 3
    assert topdown.shape[2] == 3


def test_continuous_geodesic_changes_with_subcell_motion():
    env = make_env()
    env.reset(seed=123)
    base = env._current_geodesic()
    spawn_row, spawn_col = env.state.maze.spawn_cell
    for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n_row = spawn_row + d_row
        n_col = spawn_col + d_col
        if 0 <= n_row < env.state.maze.grid.shape[0] and 0 <= n_col < env.state.maze.grid.shape[1]:
            if env.state.maze.grid[n_row, n_col] == 0:
                candidate = env.state.position + np.array([d_row, d_col], dtype=np.float32) * 0.25
                shifted = env._continuous_geodesic(candidate, env.state.maze.distance_field)
                assert shifted != base
                return
    raise AssertionError("Expected at least one free neighboring cell from the spawn position.")
