"""Shared training utilities for Jackal Differential Drive training scripts.

Provides:
  - linear_schedule: LR decay function for PPO
  - TrainingMetricsCallback: TensorBoard metrics without a second eval env
"""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def linear_schedule(initial_value: float):
    """Linear learning rate schedule: decays from initial_value to 0 over training."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class TrainingMetricsCallback(BaseCallback):
    """Track episode success rate and reward from training rollouts.

    Works WITHOUT a second eval env — reads directly from the info dicts
    populated by each env's step(). Compatible with the Isaac Sim constraint
    of one SimulationApp per process.

    Requires environments to return {"is_success": bool} in their info dict
    when an episode ends (terminated=True or truncated=True).

    TensorBoard logs (under custom/):
      success_rate   — fraction of completed episodes that succeeded
      mean_ep_reward — mean episode reward over the tracking window
      mean_ep_length — mean episode length over the tracking window

    Args:
        window: Number of recent episodes to average over.
        log_freq: Log every N timesteps (default: one rollout = n_steps).
    """

    def __init__(self, window: int = 50, log_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self._window = window
        self._log_freq = log_freq
        self._ep_successes: deque = deque(maxlen=window)
        self._ep_rewards: deque = deque(maxlen=window)
        self._ep_lengths: deque = deque(maxlen=window)

    def _on_step(self) -> bool:
        # SB3 sets locals["infos"] and locals["dones"] each call to env.step()
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                self._ep_successes.append(float(info.get("is_success", 0.0)))
                ep_info = info.get("episode", {})
                if ep_info:
                    self._ep_rewards.append(float(ep_info.get("r", 0.0)))
                    self._ep_lengths.append(int(ep_info.get("l", 0)))

        if self.num_timesteps % self._log_freq == 0 and len(self._ep_successes) > 0:
            self.logger.record("custom/success_rate", float(np.mean(self._ep_successes)))
            if self._ep_rewards:
                self.logger.record("custom/mean_ep_reward", float(np.mean(self._ep_rewards)))
            if self._ep_lengths:
                self.logger.record("custom/mean_ep_length", float(np.mean(self._ep_lengths)))

        return True
