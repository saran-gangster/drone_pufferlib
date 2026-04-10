from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class SmoothedValue:
    window_size: int = 100
    values: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def average(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
