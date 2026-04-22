"""Exact discounted value iteration — vectorized."""

from __future__ import annotations

import numpy as np

from artemis.constants import N_ACTIONS, N_STATES, TERMINAL_FAILURE, TERMINAL_SUCCESS
from artemis.environment import MissionConfig, get_transition_model_and_reward


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float = 0.95,
    theta: float = 1e-8,
    max_iters: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronous, fully vectorized VI.
    Returns (V [S], pi [S]). Terminal rows in P are absorbing with R=0, so V[terminal]=0.

    Vectorization: Q = R + gamma * (P @ V) with P reshaped to (S*A, S).
    """
    S, A = R.shape
    assert P.shape == (S, A, S)
    V = np.zeros(S, dtype=np.float64)
    P2 = P.reshape(S * A, S)
    for _ in range(max_iters):
        Q = R + gamma * (P2 @ V).reshape(S, A)
        V_new = Q.max(axis=1)
        # keep terminals pinned at 0 (they are absorbing with 0 reward, but safer to enforce)
        V_new[TERMINAL_SUCCESS] = 0.0
        V_new[TERMINAL_FAILURE] = 0.0
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new
    Q = R + gamma * (P2 @ V).reshape(S, A)
    pi = Q.argmax(axis=1)
    return V, pi


def optimal_policy_for_mission(
    cfg: MissionConfig | None = None, gamma: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the oracle policy/value for a given MissionConfig."""
    P, R = get_transition_model_and_reward(cfg)
    V, pi = value_iteration(P, R, gamma=gamma)
    return V, pi, P, R


def expected_return_from_state(
    P: np.ndarray,
    R: np.ndarray,
    pi: np.ndarray,
    start_state: int,
    gamma: float = 0.95,
    theta: float = 1e-10,
    max_iters: int = 10_000,
) -> float:
    """
    Expected discounted return of following policy `pi` starting from `start_state`,
    computed by policy evaluation on the true (P,R). Vectorized.
    """
    S = R.shape[0]
    idx = np.arange(S)
    P_pi = P[idx, pi, :]          # [S, S]
    R_pi = R[idx, pi]             # [S]
    V = np.zeros(S, dtype=np.float64)
    for _ in range(max_iters):
        V_new = R_pi + gamma * (P_pi @ V)
        if np.max(np.abs(V_new - V)) < theta:
            V = V_new
            break
        V = V_new
    return float(V[start_state])
