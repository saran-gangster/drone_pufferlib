from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np


def append_metrics_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, str]] = []
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
                for key in row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_training_curves(path: str | Path, steps: Iterable[int], rewards: Iterable[float], successes: Iterable[float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(list(steps), list(rewards))
    axes[0].set_ylabel("Eval Reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(list(steps), list(successes))
    axes[1].set_ylabel("Eval Success")
    axes[1].set_xlabel("Env Steps")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_video(path: str | Path, frames: list[np.ndarray], fps: int = 10) -> None:
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
