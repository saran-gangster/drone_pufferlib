from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RayHit:
    distance: float
    goal_glow: float
    hit_position: np.ndarray


def render_first_person(
    grid: np.ndarray,
    position: np.ndarray,
    heading: float,
    goal_position: np.ndarray,
    image_size: int,
    fov_degrees: float,
    max_depth: float,
) -> np.ndarray:
    frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    _fill_background(frame)
    half_height = image_size // 2
    fov_radians = math.radians(fov_degrees)

    for column in range(image_size):
        angle = heading + ((column / max(image_size - 1, 1)) - 0.5) * fov_radians
        hit = _cast_ray(
            grid=grid,
            position=position,
            angle=angle,
            goal_position=goal_position,
            max_depth=max_depth,
        )
        corrected = max(hit.distance * math.cos(angle - heading), 0.1)
        wall_height = int(image_size / corrected)
        wall_height = min(wall_height, image_size)
        start = max(0, half_height - wall_height // 2)
        end = min(image_size, half_height + wall_height // 2)
        color = _wall_color(hit, max_depth)
        frame[start:end, column] = color
    return np.transpose(frame, (2, 0, 1))


def render_topdown(
    grid: np.ndarray,
    position: np.ndarray,
    goal_position: np.ndarray,
    heading: float | None = None,
    scale: int = 32,
) -> np.ndarray:
    canvas = np.where(grid[..., None] == 1, 30, 220).astype(np.uint8)
    canvas = np.repeat(np.repeat(canvas, scale, axis=0), scale, axis=1)
    canvas = np.repeat(canvas, 3, axis=2)

    goal_px = (goal_position[::-1] * scale).astype(int)
    _draw_circle(canvas, goal_px, radius=max(3, scale // 4), color=(30, 210, 90))
    agent_px = (position[::-1] * scale).astype(int)
    _draw_circle(canvas, agent_px, radius=max(3, scale // 4), color=(210, 50, 50))
    if heading is not None:
        arrow = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32) * scale * 0.45
        start = tuple(agent_px.astype(int))
        end = tuple((agent_px + arrow).astype(int))
        import cv2

        cv2.arrowedLine(canvas, start, end, (45, 80, 235), thickness=max(1, scale // 10), tipLength=0.3)
    return canvas


def _cast_ray(
    grid: np.ndarray,
    position: np.ndarray,
    angle: float,
    goal_position: np.ndarray,
    max_depth: float,
    step_size: float = 0.05,
) -> RayHit:
    direction = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)
    best_goal_glow = 0.0
    distance = step_size
    while distance <= max_depth:
        sample = position + direction * distance
        goal_distance = float(np.linalg.norm(sample - goal_position))
        best_goal_glow = max(best_goal_glow, math.exp(-(goal_distance ** 2) / 0.04))
        row, col = int(round(sample[0])), int(round(sample[1]))
        if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
            return RayHit(distance=distance, goal_glow=float(best_goal_glow), hit_position=sample)
        if grid[row, col] == 1:
            return RayHit(distance=distance, goal_glow=float(best_goal_glow), hit_position=sample)
        distance += step_size
    hit_position = position + direction * max_depth
    return RayHit(distance=max_depth, goal_glow=float(best_goal_glow), hit_position=hit_position)


def _fill_background(frame: np.ndarray) -> None:
    half_height = frame.shape[0] // 2
    sky_rows = max(half_height, 1)
    floor_rows = max(frame.shape[0] - half_height, 1)

    sky_t = np.linspace(0.0, 1.0, sky_rows, dtype=np.float32)[:, None]
    sky = (1.0 - sky_t) * np.array([62, 102, 158], dtype=np.float32) + sky_t * np.array([145, 188, 228], dtype=np.float32)

    floor_t = np.linspace(0.0, 1.0, floor_rows, dtype=np.float32)[:, None]
    floor = (1.0 - floor_t) * np.array([70, 58, 45], dtype=np.float32) + floor_t * np.array([28, 24, 22], dtype=np.float32)
    row_pattern = ((np.arange(floor_rows, dtype=np.int32)[:, None] // max(1, floor_rows // 8)) % 2) * 8.0
    col_pattern = ((np.arange(frame.shape[1], dtype=np.int32)[None, :] // max(1, frame.shape[1] // 10)) % 2) * 6.0
    floor = floor[:, None, :] + row_pattern[:, :, None] - col_pattern[:, :, None]

    frame[:half_height] = np.clip(sky[:, None, :], 0, 255).astype(np.uint8)
    frame[half_height:] = np.clip(floor, 0, 255).astype(np.uint8)


def _wall_color(hit: RayHit, max_depth: float) -> np.ndarray:
    distance_ratio = min(max(hit.distance / max_depth, 0.0), 1.0)
    shade = 1.0 - 0.75 * distance_ratio
    texture_phase = (hit.hit_position[0] * 0.37 + hit.hit_position[1] * 0.61) * math.pi * 3.0
    stripe = 0.72 + 0.28 * (math.sin(texture_phase) ** 2)
    panel = 0.86 + 0.14 * ((int(math.floor(hit.hit_position[0])) + int(math.floor(hit.hit_position[1]))) % 2)
    base = np.array([170.0, 176.0, 184.0], dtype=np.float32) * shade * stripe * panel
    if hit.goal_glow > 0.05:
        goal_color = np.array([72.0, 235.0, 120.0], dtype=np.float32)
        blend = min(hit.goal_glow, 1.0)
        base = (1.0 - blend) * base + blend * goal_color
    return np.clip(base, 0, 255).astype(np.uint8)


def _draw_circle(image: np.ndarray, center: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    mask = (yy - center[1]) ** 2 + (xx - center[0]) ** 2 <= radius ** 2
    image[mask] = color
