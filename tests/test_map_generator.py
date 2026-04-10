import numpy as np

from drone_pufferlib.envs.map_generator import compute_distance_field, generate_maze


def test_generate_maze_is_deterministic():
    first = generate_maze(size=15, extra_opening_prob=0.1, min_goal_distance=6, seed=123)
    second = generate_maze(size=15, extra_opening_prob=0.1, min_goal_distance=6, seed=123)
    assert np.array_equal(first.grid, second.grid)
    assert first.spawn_cell == second.spawn_cell
    assert first.goal_cell == second.goal_cell


def test_spawn_goal_min_distance_is_respected():
    maze = generate_maze(size=15, extra_opening_prob=0.1, min_goal_distance=6, seed=5)
    assert maze.distance_field[maze.spawn_cell] >= 6


def test_distance_field_marks_goal_zero():
    maze = generate_maze(size=15, extra_opening_prob=0.1, min_goal_distance=6, seed=9)
    field = compute_distance_field(maze.grid, maze.goal_cell)
    assert field[maze.goal_cell] == 0
