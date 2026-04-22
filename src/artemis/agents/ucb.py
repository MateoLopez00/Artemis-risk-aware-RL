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


class UCQAgent:
    """
    Q-learning with UCB-style action selection.
    score(s,a) = Q(s,a) + c_ucb * sqrt(ln(t) / (N(s,a) + 1))
    Bootstrap target uses standard TD over valid next-actions only.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        alpha: float = 0.1,
        c_ucb: float = 0.5,
        seed: int | None = None,
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.c_ucb = c_ucb
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.N = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)

    def masked_max(self, s: int, mask: np.ndarray) -> float:
        m = self.Q[s].copy()
        m[mask == 0] = -np.inf
        return float(m.max()) if (mask == 1).any() else 0.0

    def act(self, state: int, mask: np.ndarray | None = None) -> int:
        self.t += 1
        m = action_mask_int(state) if mask is None else mask
        if not (m == 1).any():
            return 0
        logt = float(np.log(max(1, self.t)))
        bonus = self.c_ucb * np.sqrt(logt / (self.N[state].astype(np.float64) + 1.0))
        scores = self.Q[state] + bonus
        scores = scores.copy()
        scores[m == 0] = -np.inf
        mx = scores.max()
        cands = np.where(scores == mx)[0]
        return int(cands[0]) if len(cands) == 1 else int(self.rng.choice(cands))

    def update(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        done: bool,
        mask_next: np.ndarray | None = None,
    ) -> None:
        if s < N_NON_TERMINAL:
            self.N[s, a] += 1
        if s >= N_NON_TERMINAL:
            return
        if s_next in (TERMINAL_SUCCESS, TERMINAL_FAILURE) or done:
            target = r
        else:
            mn = action_mask_int(s_next) if mask_next is None else mask_next
            target = r + self.gamma * self.masked_max(s_next, mn)
        self.Q[s, a] = (1.0 - self.alpha) * self.Q[s, a] + self.alpha * target

    def end_episode(self) -> None:
        return None

    def reset(self) -> None:
        self.t = 0
        self.Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self.N = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
