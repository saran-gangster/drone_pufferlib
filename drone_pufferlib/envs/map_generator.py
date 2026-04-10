from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


CARDINALS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)


@dataclass
class MazeSpec:
    grid: np.ndarray
    spawn_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    distance_field: np.ndarray


def _cell_positions(size: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(1, size, 2) for col in range(1, size, 2)]


def _neighbors(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    row, col = cell
    candidates = []
    for d_row, d_col in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        n_row, n_col = row + d_row, col + d_col
        if 1 <= n_row < size - 1 and 1 <= n_col < size - 1:
            candidates.append((n_row, n_col))
    return candidates


def generate_maze(size: int, extra_opening_prob: float, min_goal_distance: int, seed: int) -> MazeSpec:
    if size % 2 == 0:
        raise ValueError("Maze size must be odd.")

    rng = np.random.default_rng(seed)
    grid = np.ones((size, size), dtype=np.uint8)
    cells = _cell_positions(size)
    start = cells[rng.integers(0, len(cells))]
    stack = [start]
    visited = {start}
    grid[start] = 0

    while stack:
        current = stack[-1]
        unvisited = [cell for cell in _neighbors(current, size) if cell not in visited]
        if not unvisited:
            stack.pop()
            continue
        nxt = unvisited[rng.integers(0, len(unvisited))]
        wall = ((current[0] + nxt[0]) // 2, (current[1] + nxt[1]) // 2)
        grid[nxt] = 0
        grid[wall] = 0
        visited.add(nxt)
        stack.append(nxt)

    for row in range(1, size - 1):
        for col in range(1, size - 1):
            if grid[row, col] != 1:
                continue
            horizontal = grid[row, col - 1] == 0 and grid[row, col + 1] == 0
            vertical = grid[row - 1, col] == 0 and grid[row + 1, col] == 0
            if (horizontal or vertical) and rng.random() < extra_opening_prob:
                grid[row, col] = 0

    free_cells = list(zip(*np.where(grid == 0)))
    spawn_cell, goal_cell, distance_field = _sample_spawn_goal(
        grid=grid,
        free_cells=free_cells,
        min_goal_distance=min_goal_distance,
        rng=rng,
    )
    return MazeSpec(
        grid=grid,
        spawn_cell=spawn_cell,
        goal_cell=goal_cell,
        distance_field=distance_field,
    )


def _sample_spawn_goal(
    grid: np.ndarray,
    free_cells: list[tuple[int, int]],
    min_goal_distance: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    for _ in range(128):
        goal_cell = free_cells[rng.integers(0, len(free_cells))]
        distance_field = compute_distance_field(grid, goal_cell)
        valid_spawns = [cell for cell in free_cells if np.isfinite(distance_field[cell]) and distance_field[cell] >= min_goal_distance]
        if valid_spawns:
            spawn_cell = valid_spawns[rng.integers(0, len(valid_spawns))]
            return spawn_cell, goal_cell, distance_field
    raise RuntimeError("Failed to sample spawn and goal cells with the requested geodesic distance.")


def compute_distance_field(grid: np.ndarray, goal_cell: tuple[int, int]) -> np.ndarray:
    distance_field = np.full(grid.shape, np.inf, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque([goal_cell])
    distance_field[goal_cell] = 0.0

    while queue:
        row, col = queue.popleft()
        for d_row, d_col in CARDINALS:
            n_row = row + int(d_row)
            n_col = col + int(d_col)
            if not (0 <= n_row < grid.shape[0] and 0 <= n_col < grid.shape[1]):
                continue
            if grid[n_row, n_col] == 1:
                continue
            candidate = distance_field[row, col] + 1.0
            if candidate < distance_field[n_row, n_col]:
                distance_field[n_row, n_col] = candidate
                queue.append((n_row, n_col))
    return distance_field
