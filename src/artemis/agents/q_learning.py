from __future__ import annotations

import numpy as np

from artemis.constants import (
    N_ACTIONS,
    N_NON_TERMINAL,
    N_STATES,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
)
from artemis.environment import action_mask_int


class QLearningAgent:
    """Tabular Q-learning with ε-greedy over valid actions only."""

    def __init__(
        self,
        gamma: float = 0.95,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    def _select_greedy(self, q_row: np.ndarray, mask: np.ndarray) -> int:
        m = q_row.copy()
        m[mask == 0] = -np.inf
        mval = m.max()
        cands = np.where(m == mval)[0]
        if len(cands) == 0:
            return 0
        if len(cands) == 1:
            return int(cands[0])
        return int(self.rng.choice(cands))

    def masked_max(self, s: int, mask: np.ndarray) -> float:
        m = self.Q[s].copy()
        m[mask == 0] = -np.inf
        return float(m.max()) if (mask == 1).any() else 0.0

    def act(self, state: int, mask: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            valid = np.where(mask == 1)[0]
            if len(valid) == 0:
                return 0
            return int(self.rng.choice(valid))
        return self._select_greedy(self.Q[state], mask)

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        done: bool,
        mask_next: np.ndarray | None = None,
    ) -> None:
        if s >= N_NON_TERMINAL:
            return
        if s_next in (TERMINAL_SUCCESS, TERMINAL_FAILURE) or done:
            target = r
        else:
            m = action_mask_int(s_next) if mask_next is None else mask_next
            target = r + self.gamma * self.masked_max(s_next, m)
        self.Q[s, a] = (1.0 - self.alpha) * self.Q[s, a] + self.alpha * target

    def end_episode(self) -> None:
        """Hook for agents that do per-episode updates. No-op for plain Q-learning."""
        return None

    def reset(self) -> None:
        self.Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
