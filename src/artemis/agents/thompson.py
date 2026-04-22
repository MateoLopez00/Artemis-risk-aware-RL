"""
Posterior Sampling for RL (PSRL / Thompson sampling over MDPs):
  - Dirichlet posterior over next-state distribution for each (s,a)
  - Gaussian-like empirical posterior over immediate reward (use MLE here for simplicity)
  - At the start of each episode, sample one P, compute optimal Q via VI, act greedily.
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

_DIR_PRIOR = 0.1


def _sample_dirichlet_batch(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Vectorized Dirichlet sampling for a batch with shape [..., K]:
    sample Gamma(alpha, 1) then normalize. Much faster than looping.
    """
    g = rng.standard_gamma(alpha)
    s = g.sum(axis=-1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    return g / s


class ThompsonSamplingMDPAgent:
    """PSRL-style agent: resample MDP each episode, plan via VI, act greedily."""

    def __init__(
        self,
        gamma: float = 0.95,
        prior: float = 1.0,          # uninformative Dirichlet prior — samples are diffuse
        resample_each_episode: bool = True,
        seed: int | None = None,
    ):
        self.gamma = gamma
        self.prior = float(prior)
        self.resample_each_episode = resample_each_episode
        self.rng = np.random.default_rng(seed)
        self._counts = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float64)
        self._r_sum = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self._visits = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self._q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
        self._resample_and_plan()

    def _build_alpha(self) -> np.ndarray:
        """Alpha tensor [S,A,S] for Dirichlet; terminals pinned to self-loop."""
        alpha = self._counts + self.prior
        for t in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
            alpha[t, :, :] = 1e-8
            alpha[t, :, t] = 1.0  # peaks the sample at self-loop
        return alpha

    def _r_bar(self) -> np.ndarray:
        R = self._r_sum / np.maximum(self._visits, 1.0)
        for t in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
            R[t, :] = 0.0
        return R

    def _resample_and_plan(self) -> None:
        alpha = self._build_alpha()
        P = _sample_dirichlet_batch(alpha, self.rng)
        R = self._r_bar()

        # Critical: enforce the action mask on the sampled model before VI.
        # Illegal (s,a) pairs (e.g. safe at fuel=0) have counts=0 so Dirichlet
        # samples them as P≈uniform and R=0.  Without this fix VI inflates V of
        # fuel-0 states and the greedy policy degenerates to "safe everywhere".
        for s in range(N_NON_TERMINAL):
            mask = action_mask_int(s)
            for a in range(N_ACTIONS):
                if mask[a] == 0:
                    P[s, a, :] = 0.0
                    P[s, a, TERMINAL_FAILURE] = 1.0
                    R[s, a] = -100.0

        # enforce absorbing terminals exactly
        for t in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
            P[t, :, :] = 0.0
            P[t, :, t] = 1.0
            R[t, :] = 0.0

        V, _pi = value_iteration(P, R, gamma=self.gamma)
        S, A = R.shape
        P2 = P.reshape(S * A, S)
        self._q = R + self.gamma * (P2 @ V).reshape(S, A)

    def new_episode(self) -> None:
        if self.resample_each_episode:
            self._resample_and_plan()

    def end_episode(self) -> None:
        self.new_episode()

    def act(self, state: int, mask: np.ndarray | None = None) -> int:
        m = action_mask_int(state) if mask is None else mask
        if not (m == 1).any():
            return 0
        q = self._q[state].copy()
        q[m == 0] = -np.inf
        mx = q.max()
        cands = np.where(q == mx)[0]
        return int(cands[0]) if len(cands) == 1 else int(self.rng.choice(cands))

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

    def reset(self) -> None:
        self._counts.fill(0.0)
        self._r_sum.fill(0.0)
        self._visits.fill(0.0)
        self._resample_and_plan()
