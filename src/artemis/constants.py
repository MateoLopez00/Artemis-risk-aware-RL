"""Action indices: fixed ordering for the whole codebase."""

import numpy as np

ACTION_SAFE = 0
ACTION_RISKY = 1
ACTION_FALLBACK = 2
N_ACTIONS = 3

# State indices: 0..N_NON_TERMINAL-1 are non-terminal; 48–49 are absorbing terminals
N_NON_TERMINAL = 48
TERMINAL_SUCCESS = 48
TERMINAL_FAILURE = 49
N_STATES = 50

# Risky route base probabilities: (safe, hazard, fail) per stage 1..4 under normal hazard
RISKY_PROBS_NORMAL = np.array(
    [
        [0.75, 0.20, 0.05],
        [0.70, 0.20, 0.10],
        [0.65, 0.20, 0.15],
        [0.55, 0.25, 0.20],
    ],
    dtype=np.float64,
)

RISKY_PROBS_HAZARD = np.array(
    [
        [0.65, 0.20, 0.15],
        [0.60, 0.20, 0.20],
        [0.55, 0.20, 0.25],
        [0.45, 0.25, 0.30],
    ],
    dtype=np.float64,
)
