from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_value: torch.Tensor


def collect_rollout(vector_env, model, device, rollout_length: int):
    obs_buffer = []
    action_buffer = []
    log_prob_buffer = []
    reward_buffer = []
    done_buffer = []
    value_buffer = []
    episode_returns = []
    episode_successes = []

    obs = vector_env._latest_obs
    for _ in range(rollout_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, log_probs, values = model.act(obs_tensor)
        next_obs, rewards, terminations, truncations, infos = vector_env.step(actions.cpu().numpy())
        dones = np.logical_or(terminations, truncations)
        obs_buffer.append(obs_tensor.cpu())
        action_buffer.append(actions.cpu())
        log_prob_buffer.append(log_probs.cpu())
        reward_buffer.append(torch.as_tensor(rewards, dtype=torch.float32))
        done_buffer.append(torch.as_tensor(dones, dtype=torch.float32))
        value_buffer.append(values.cpu())
        if "final_info" in infos:
            for final in infos["final_info"]:
                if final is None:
                    continue
                if "episode" in final:
                    episode_returns.append(float(final["episode"]["r"]))
                elif "episode_return" in final:
                    episode_returns.append(float(final["episode_return"]))
                episode_successes.append(float(final.get("success", 0.0)))
        obs = next_obs

    vector_env._latest_obs = obs
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, next_value = model.forward(obs_tensor)

    batch = RolloutBatch(
        obs=torch.stack(obs_buffer),
        actions=torch.stack(action_buffer),
        log_probs=torch.stack(log_prob_buffer),
        rewards=torch.stack(reward_buffer),
        dones=torch.stack(done_buffer),
        values=torch.stack(value_buffer),
        next_value=next_value.cpu(),
    )
    return batch, episode_returns, episode_successes


def compute_gae(batch: RolloutBatch, gamma: float, gae_lambda: float):
    rewards = batch.rewards
    dones = batch.dones
    values = batch.values
    next_value = batch.next_value
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)

    for step in reversed(range(rewards.shape[0])):
        next_non_terminal = 1.0 - dones[step]
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    device: torch.device,
    minibatch_size: int,
    update_epochs: int,
    clip_coef: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    normalize_advantages: bool,
) -> dict[str, float]:
    obs = batch.obs.reshape(-1, *batch.obs.shape[2:]).to(device)
    actions = batch.actions.reshape(-1).to(device)
    old_log_probs = batch.log_probs.reshape(-1).to(device)
    advantages = advantages.reshape(-1).to(device)
    returns = returns.reshape(-1).to(device)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    num_updates = 0
    indices = np.arange(obs.shape[0])
    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), minibatch_size):
            idx = torch.as_tensor(indices[start : start + minibatch_size], device=device)
            new_log_probs, entropy, values = model.evaluate_actions(obs[idx], actions[idx])
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            surrogate1 = ratio * advantages[idx]
            surrogate2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * advantages[idx]
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = 0.5 * (returns[idx] - values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy_loss.item())
            num_updates += 1
    divisor = max(num_updates, 1)
    return {
        "loss": total_loss / divisor,
        "policy_loss": total_policy / divisor,
        "value_loss": total_value / divisor,
        "entropy": total_entropy / divisor,
    }
