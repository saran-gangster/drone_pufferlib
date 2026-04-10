from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions.categorical import Categorical


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class CnnActorCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int], action_dim: int, channels: list[int], hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        c, h, w = obs_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(c, channels[0], kernel_size=8, stride=4),
            nn.GroupNorm(_norm_groups(channels[0]), channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2),
            nn.GroupNorm(_norm_groups(channels[1]), channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1),
            nn.GroupNorm(_norm_groups(channels[2]), channels[2]),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            encoded_dim = self.encoder(sample).shape[-1]
        self.projection = nn.Sequential(
            nn.Linear(encoded_dim, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
        self._reset_parameters()

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float() / 255.0
        return self.projection(self.encoder(obs))

    def forward_eval(self, obs: torch.Tensor, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def forward(self, obs: torch.Tensor, state=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_eval(obs, state)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward_eval(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward_eval(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                gain = math.sqrt(2.0)
                if module is self.actor:
                    gain = 0.01
                elif module is self.critic:
                    gain = 1.0
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
