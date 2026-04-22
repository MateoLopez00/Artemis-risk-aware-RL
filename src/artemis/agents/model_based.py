"""
Model-based RL: maintain count-based MLE of (P, R) with Dirichlet pseudocounts,
re-plan with value iteration periodically, act ε-greedy on the planned Q.
"""

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
from artemis.planning import value_iteration

PSEUDO = 0.1


class ModelBasedVIAgent:
    """
    Counts transitions and rewards; MLE model with Dirichlet pseudocounts; runs VI
    every `vi_every` *episodes* by default (cheap + correct). Acts ε-greedy on planned Q.
    """

    def __init__(
        self,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        vi_every: int = 5,           # rebuild every N episodes (cheap after first few)
        pseudocount: float = 0.01,   # small prior so real counts dominate quickly
        seed: int | None = None,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.vi_every = max(1, int(vi_every))
        self.pseudo = float(pseudocount)
        self.rng = np.random.default_rng(seed)
        self._counts = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float64)
        self._r_sum = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self._visits = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self._episodes = 0
        self._q, self._v = self._initial_plan()

    def _initial_plan(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        v = np.zeros(N_STATES, dtype=np.float64)
        return q, v

    def _rebuild(self) -> None:
        # Vectorized Dirichlet-MLE over counts
        c = self._counts + self.pseudo          # [S,A,S]
        denom = c.sum(axis=-1, keepdims=True)   # [S,A,1]
        P = c / denom
        # Reward MLE: mean of observed rewards
        R = self._r_sum / np.maximum(self._visits, 1.0)

        # Critical: enforce the action mask so that illegal (s,a) pairs — e.g. safe
        # at fuel=0, which the env treats as immediate -100 failure — are not left
        # with P=uniform and R=0 by the pseudocount prior. Without this correction,
        # VI inflates V(fuel=0 states) to ~100 via phantom "free" illegal actions and
        # the greedy policy degenerates to "safe→safe→safe" everywhere.
        for s in range(N_NON_TERMINAL):
            mask = action_mask_int(s)
            for a in range(N_ACTIONS):
                if mask[a] == 0:
                    P[s, a, :] = 0.0
                    P[s, a, TERMINAL_FAILURE] = 1.0
                    R[s, a] = -100.0

        # Pin terminals to absorbing with 0 reward
        for t in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
            P[t, :, :] = 0.0
            P[t, :, t] = 1.0
            R[t, :] = 0.0

        V, _pi = value_iteration(P, R, gamma=self.gamma)
        S, A = R.shape
        P2 = P.reshape(S * A, S)
        self._q = (R + self.gamma * (P2 @ V).reshape(S, A))
        self._v = V

    def observe(
        self,
        s: int,
        a: int,
        r: float,
        s_next: int,
        done: bool,
        mask_next: np.ndarray | None = None,
    ) -> None:
        if s < N_NON_TERMINAL:
            self._counts[s, a, s_next] += 1.0
            self._r_sum[s, a] += r
            self._visits[s, a] += 1.0

    def end_episode(self) -> None:
        self._episodes += 1
        if self._episodes % self.vi_every == 0:
            self._rebuild()

    def act(self, state: int, mask: np.ndarray | None = None) -> int:
        m = action_mask_int(state) if mask is None else mask
        if not (m == 1).any():
            return 0
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.where(m == 1)[0]))
        q = self._q[state].copy()
        q[m == 0] = -np.inf
        mx = q.max()
        cands = np.where(q == mx)[0]
        return int(cands[0]) if len(cands) == 1 else int(self.rng.choice(cands))

    def new_episode(self) -> None:
        pass

    def reset(self) -> None:
        self._counts.fill(0.0)
        self._r_sum.fill(0.0)
        self._visits.fill(0.0)
        self._episodes = 0
        self._q, self._v = self._initial_plan()
