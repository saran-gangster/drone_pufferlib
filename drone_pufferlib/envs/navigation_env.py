from __future__ import annotations

from collections import deque
import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from drone_pufferlib.envs.map_generator import MazeSpec, generate_maze
from drone_pufferlib.envs.renderer import render_first_person, render_topdown


@dataclass
class EpisodeState:
    maze: MazeSpec
    position: np.ndarray
    heading: float
    step_count: int
    prev_geodesic: float
    frame_history: deque[np.ndarray]
    latest_frame: np.ndarray


class NavigationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "topdown"], "render_fps": 10}

    def __init__(self, config: dict[str, Any], seed: int | None = None):
        super().__init__()
        self.config = config
        self.default_seed = int(config.get("seed", 0) if seed is None else seed)
        self.render_mode = config.get("render_mode", "rgb_array")
        self.difficulty = config["difficulty"]
        self.difficulty_cfg = config["difficulties"][self.difficulty]
        self.image_size = int(config["observation_size"])
        self.forward_step = float(config["forward_step"])
        self.turn_radians = math.radians(float(config["turn_degrees"]))
        self.collision_radius = float(config["collision_radius"])
        self.goal_radius = float(config["goal_radius"])
        self.reward_cfg = config["reward"]
        self.fov_degrees = float(config["fov_degrees"])
        self.max_depth = float(config["max_depth"])
        self.frame_stack = int(config.get("frame_stack", 1))
        self.spawn_heading_noise = math.radians(float(config.get("spawn_heading_noise_degrees", 180.0)))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3 * self.frame_stack, self.image_size, self.image_size),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(3)
        self._episode_seed = self.default_seed
        self._auto_seed_base = self.default_seed
        self._reset_count = 0
        self.state: EpisodeState | None = None
        self.np_random = np.random.default_rng(self.default_seed)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
            self._auto_seed_base = self._episode_seed
            self._reset_count = 1
        elif options and "episode_seed" in options:
            self._episode_seed = int(options["episode_seed"])
            self._auto_seed_base = self._episode_seed
            self._reset_count = 1
        else:
            self._episode_seed = int(self._auto_seed_base + self._reset_count)
            self._reset_count += 1
        self.np_random = np.random.default_rng(self._episode_seed)

        difficulty = options.get("difficulty", self.difficulty) if options else self.difficulty
        self.difficulty = difficulty
        self.difficulty_cfg = self.config["difficulties"][difficulty]
        maze = generate_maze(
            size=int(self.difficulty_cfg["size"]),
            extra_opening_prob=float(self.difficulty_cfg["extra_opening_prob"]),
            min_goal_distance=int(self.difficulty_cfg["min_goal_distance"]),
            seed=self._episode_seed,
        )
        spawn = np.array(maze.spawn_cell, dtype=np.float32)
        goal = np.array(maze.goal_cell, dtype=np.float32)
        heading = self._sample_initial_heading(maze)
        initial_frame = self._render_agent_view(grid=maze.grid, position=spawn, heading=heading, goal_position=goal)
        frame_history = deque([initial_frame.copy() for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        prev_geodesic = self._continuous_geodesic(spawn, maze.distance_field)
        self.state = EpisodeState(
            maze=maze,
            position=spawn,
            heading=heading,
            step_count=0,
            prev_geodesic=prev_geodesic,
            frame_history=frame_history,
            latest_frame=initial_frame,
        )
        return self._get_obs(), self._info(success=False, collision=False)

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."
        collision = False
        if action == 0:
            direction = np.array([math.sin(self.state.heading), math.cos(self.state.heading)], dtype=np.float32)
            candidate = self.state.position + direction * self.forward_step
            if self._collides(candidate):
                collision = True
            else:
                self.state.position = candidate
        elif action == 1:
            self.state.heading -= self.turn_radians
        elif action == 2:
            self.state.heading += self.turn_radians
        else:
            raise ValueError(f"Unsupported action {action}")

        self.state.step_count += 1
        goal_position = np.array(self.state.maze.goal_cell, dtype=np.float32)
        geodesic = self._current_geodesic()
        progress_delta = float(np.clip(self.state.prev_geodesic - geodesic, -1.0, 1.0))
        reward = float(self.reward_cfg["step"]) + float(self.reward_cfg["progress_scale"]) * progress_delta
        success = np.linalg.norm(self.state.position - goal_position) <= self.goal_radius
        terminated = False
        truncated = False
        if success:
            reward += float(self.reward_cfg["goal"])
            terminated = True
        elif collision:
            reward += float(self.reward_cfg["collision"])
            terminated = True
        elif self.state.step_count >= int(self.difficulty_cfg["max_steps"]):
            truncated = True

        self.state.prev_geodesic = geodesic
        latest_frame = self._render_agent_view(
            grid=self.state.maze.grid,
            position=self.state.position,
            heading=self.state.heading,
            goal_position=goal_position,
        )
        self.state.latest_frame = latest_frame
        self.state.frame_history.append(latest_frame)
        info = self._info(success=success, collision=collision)
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        assert self.state is not None, "Call reset() before render()."
        if self.render_mode == "rgb_array":
            return self.render_first_person()
        if self.render_mode == "topdown":
            return self.render_topdown()
        raise ValueError(f"Unsupported render mode {self.render_mode}")

    def render_first_person(self) -> np.ndarray:
        assert self.state is not None, "Call reset() before render()."
        return np.transpose(self.state.latest_frame, (1, 2, 0)).copy()

    def render_topdown(self) -> np.ndarray:
        assert self.state is not None, "Call reset() before render()."
        return render_topdown(
            grid=self.state.maze.grid,
            position=self.state.position,
            goal_position=np.array(self.state.maze.goal_cell, dtype=np.float32),
            heading=self.state.heading,
        )

    def _get_obs(self) -> np.ndarray:
        assert self.state is not None
        return np.concatenate(list(self.state.frame_history), axis=0)

    def _render_agent_view(
        self,
        *,
        grid: np.ndarray,
        position: np.ndarray,
        heading: float,
        goal_position: np.ndarray,
    ) -> np.ndarray:
        return render_first_person(
            grid=grid,
            position=position,
            heading=heading,
            goal_position=goal_position,
            image_size=self.image_size,
            fov_degrees=self.fov_degrees,
            max_depth=self.max_depth,
        )

    def _cell(self, position: np.ndarray) -> tuple[int, int]:
        row = int(np.clip(round(float(position[0])), 0, self.state.maze.grid.shape[0] - 1))
        col = int(np.clip(round(float(position[1])), 0, self.state.maze.grid.shape[1] - 1))
        return row, col

    def _current_geodesic(self) -> float:
        return self._continuous_geodesic(self.state.position, self.state.maze.distance_field)

    def _collides(self, candidate: np.ndarray) -> bool:
        samples = [candidate]
        for angle in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            offset = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32) * self.collision_radius
            samples.append(candidate + offset)
        for sample in samples:
            row, col = self._cell(sample)
            if self.state.maze.grid[row, col] == 1:
                return True
        return False

    def _info(self, success: bool, collision: bool) -> dict[str, Any]:
        assert self.state is not None
        goal_position = np.array(self.state.maze.goal_cell, dtype=np.float32)
        return {
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(self.state.step_count >= int(self.difficulty_cfg["max_steps"]) and not success and not collision),
            "distance_to_goal": float(np.linalg.norm(self.state.position - goal_position)),
            "geodesic_distance": float(self._current_geodesic()),
            "difficulty": self.difficulty,
            "episode_seed": int(self._episode_seed),
        }

    def _sample_initial_heading(self, maze: MazeSpec) -> float:
        spawn_row, spawn_col = maze.spawn_cell
        candidate_headings = []
        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n_row = spawn_row + d_row
            n_col = spawn_col + d_col
            if 0 <= n_row < maze.grid.shape[0] and 0 <= n_col < maze.grid.shape[1] and maze.grid[n_row, n_col] == 0:
                candidate_headings.append(math.atan2(d_row, d_col))

        if not candidate_headings:
            return float(self.np_random.uniform(-math.pi, math.pi))

        base_heading = float(candidate_headings[self.np_random.integers(0, len(candidate_headings))])
        if self.spawn_heading_noise >= math.pi:
            return float(self.np_random.uniform(-math.pi, math.pi))
        return float(base_heading + self.np_random.uniform(-self.spawn_heading_noise, self.spawn_heading_noise))

    def _continuous_geodesic(self, position: np.ndarray, distance_field: np.ndarray) -> float:
        row_min = max(0, int(math.floor(float(position[0]))) - 1)
        row_max = min(distance_field.shape[0] - 1, int(math.ceil(float(position[0]))) + 1)
        col_min = max(0, int(math.floor(float(position[1]))) - 1)
        col_max = min(distance_field.shape[1] - 1, int(math.ceil(float(position[1]))) + 1)

        best = float("inf")
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                field_value = float(distance_field[row, col])
                if not np.isfinite(field_value):
                    continue
                cell_center = np.array([row, col], dtype=np.float32)
                candidate = field_value + float(np.linalg.norm(position - cell_center))
                best = min(best, candidate)

        if np.isfinite(best):
            return best

        cell = self._cell(position)
        return float(distance_field[cell])
