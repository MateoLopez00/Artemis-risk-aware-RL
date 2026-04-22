"""
Lunar mission tabular MDP — exact implementation of the project proposal.

State encoding (non-terminal, indices 0..47):
    obs = (stage-1) * 12 + fuel * 4 + hazard * 2 + fallback
    with stage in {1,2,3,4}, fuel in {0,1,2}, hazard in {0,1}, fallback in {0,1}

Terminal indices (absorbing):
    48 = TERMINAL_SUCCESS
    49 = TERMINAL_FAILURE

Rewards per spec:
    +10 stage advance
    +50 final mission success (so entering stage 5 with fuel > 0 yields +60 total)
    -5  entering hazard (transitioning normal -> hazard)
    -10 using fallback
    -100 mission failure (catastrophic or fuel would go below 0)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from artemis.constants import (
    ACTION_FALLBACK,
    ACTION_RISKY,
    ACTION_SAFE,
    N_ACTIONS,
    N_NON_TERMINAL,
    N_STATES,
    RISKY_PROBS_HAZARD,
    RISKY_PROBS_NORMAL,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
)


@dataclass
class MissionConfig:
    """
    Environment knobs from the proposal "Experiment Design":
      * risky_failure_scale  — multiply risky failure branch prob, renormalize row.
      * hazard_penalty       — reward when entering hazard (default -5 per proposal).
      * fallback_recovery_prob — probability that fallback resets hazard to normal (default 0.90).
      * start_fuel           — initial fuel level (default 2 = high).
      * seed                 — environment RNG seed.
    """

    risky_failure_scale: float = 1.0
    hazard_penalty: float = -5.0
    fallback_recovery_prob: float = 0.90
    start_fuel: int = 2
    seed: int | None = None


def decode_state(obs: int) -> tuple[int, int, int, int]:
    """Decode obs 0..47 -> (stage 1-4, fuel 0-2, hazard 0-1, fallback 0-1)."""
    if obs < 0 or obs >= N_NON_TERMINAL:
        raise ValueError(f"decode_state called with terminal/invalid obs {obs}")
    stage = obs // 12 + 1
    r = obs % 12
    fuel = r // 4
    r = r % 4
    hazard, fb = r // 2, r % 2
    return stage, fuel, hazard, fb


def encode_state(stage: int, fuel: int, hazard: int, fallback: int) -> int:
    """Inverse of decode_state."""
    if not (1 <= stage <= 4 and 0 <= fuel <= 2 and 0 <= hazard <= 1 and 0 <= fallback <= 1):
        raise ValueError(f"Invalid state tuple {(stage, fuel, hazard, fallback)}")
    return (stage - 1) * 12 + fuel * 4 + hazard * 2 + fallback


def action_mask_int(state: int) -> np.ndarray:
    """
    Legal-action mask (1=valid). Logic:
      - terminals -> all zeros
      - safe     : needs fuel >= 1
      - risky    : always allowed on non-terminal (may fail at fuel=0 if hazard branch fires)
      - fallback : needs fuel >= 1 and fallback still available
    """
    mask = np.ones(3, dtype=np.int8)
    if state < 0 or state >= N_NON_TERMINAL:
        return np.zeros(3, dtype=np.int8)
    _st, fuel, _h, fb = decode_state(state)
    if fuel < 1:
        mask[ACTION_SAFE] = 0
    if fuel < 1 or fb == 1:
        mask[ACTION_FALLBACK] = 0
    return mask


def _scale_risky_row(base: np.ndarray, fail_scale: float) -> np.ndarray:
    """Multiply failure prob by `fail_scale`, rescale the remaining two branches proportionally."""
    if fail_scale <= 0:
        raise ValueError("risky_failure_scale must be positive")
    out = base.copy()
    for i in range(4):
        s, h, f = out[i, 0], out[i, 1], out[i, 2]
        f_new = min(0.999, f * fail_scale)
        rem = 1.0 - f_new
        s_h = s + h
        if s_h <= 0:
            out[i] = [0.0, 0.0, 1.0]
            continue
        out[i, 0] = rem * (s / s_h)
        out[i, 1] = rem * (h / s_h)
        out[i, 2] = f_new
    return out


class LunarMissionEnv(gym.Env):
    """Gymnasium-style tabular env for the lunar mission."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: MissionConfig | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self._cfg = config or MissionConfig()
        if seed is not None:
            self._cfg = replace(self._cfg, seed=seed)
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._rng: np.random.Generator = np.random.default_rng(self._cfg.seed)
        self._state: int = 0
        self._hazard_penalty: float = self._cfg.hazard_penalty
        self._risky_normal = _scale_risky_row(RISKY_PROBS_NORMAL, self._cfg.risky_failure_scale)
        self._risky_hazard = _scale_risky_row(RISKY_PROBS_HAZARD, self._cfg.risky_failure_scale)

    @property
    def config(self) -> MissionConfig:
        return self._cfg

    @property
    def hazard_penalty(self) -> float:
        return self._hazard_penalty

    def get_action_mask(self, state_idx: int | None = None) -> np.ndarray:
        """Mask over [safe, risky, fallback]."""
        s = self._state if state_idx is None else int(state_idx)
        return action_mask_int(s)

    def set_state(self, state_idx: int) -> None:
        """Manually set internal state (for tests and debugging, not part of Gym API)."""
        self._state = int(state_idx)

    # ---- reward helper ----
    def _reward_components(
        self,
        old_hazard: int,
        new_hazard: int,
        new_stage: int,
        new_fuel: int,
        used_fallback: bool,
    ) -> float:
        r = 0.0
        if used_fallback:
            r -= 10.0
        r += 10.0  # stage advance
        if new_stage == 5 and new_fuel > 0:
            r += 50.0
        if new_hazard and not old_hazard:
            r += self._hazard_penalty
        return r

    # ---- API ----
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        f = int(np.clip(self._cfg.start_fuel, 0, 2))
        self._state = encode_state(1, f, 0, 0)
        return self._state, self._info(self._state)

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        if self._state >= N_NON_TERMINAL:
            return self._state, 0.0, True, False, {"need_reset": True}

        st, fuel, haz, fb = decode_state(self._state)

        a = int(action)
        if a not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}")

        mask = action_mask_int(self._state)
        if mask[a] == 0:
            # Invalid action (e.g., safe/fallback with zero fuel, or fallback already used) -> failure
            self._state = TERMINAL_FAILURE
            return self._state, -100.0, True, False, {**self._info(self._state), "success": False, "invalid_action": True}

        if a == ACTION_SAFE:
            return self._step_safe(st, fuel, haz, fb)
        if a == ACTION_RISKY:
            return self._step_risky(st, fuel, haz, fb)
        return self._step_fallback(st, fuel, haz, fb)

    # ---- step branches ----
    def _finish(
        self,
        old_h: int,
        new_h: int,
        new_s: int,
        new_f: int,
        new_fb: int,
        used_fallback: bool,
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Handle common logic: terminal success/failure vs. non-terminal transition."""
        if new_s == 5:
            if new_f > 0:
                rew = self._reward_components(old_h, new_h, new_s, new_f, used_fallback)
                self._state = TERMINAL_SUCCESS
                return self._state, rew, True, False, {**self._info(self._state), "success": True}
            self._state = TERMINAL_FAILURE
            return self._state, -100.0, True, False, {**self._info(self._state), "success": False}
        self._state = encode_state(new_s, new_f, new_h, new_fb)
        rew = self._reward_components(old_h, new_h, new_s, new_f, used_fallback)
        return self._state, rew, False, False, {**self._info(self._state), "success": None}

    def _step_safe(
        self, st: int, fuel: int, haz: int, fb: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        u = self._rng.random()
        new_h = 0 if u < 0.95 else 1
        new_f = fuel - 1
        new_s = st + 1
        return self._finish(haz, new_h, new_s, new_f, fb, used_fallback=False)

    def _step_risky(
        self, st: int, fuel: int, haz: int, fb: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        row = self._risky_hazard[st - 1] if haz else self._risky_normal[st - 1]
        u = self._rng.random()
        c0, c1 = float(row[0]), float(row[0] + row[1])
        if u < c0:
            # safe progress
            new_s, new_f, new_h = st + 1, fuel, 0
            return self._finish(haz, new_h, new_s, new_f, fb, used_fallback=False)
        if u < c1:
            # hazard progress
            if fuel < 1:
                self._state = TERMINAL_FAILURE
                return self._state, -100.0, True, False, {**self._info(self._state), "success": False}
            new_s, new_f, new_h = st + 1, fuel - 1, 1
            return self._finish(haz, new_h, new_s, new_f, fb, used_fallback=False)
        # failure
        self._state = TERMINAL_FAILURE
        return self._state, -100.0, True, False, {**self._info(self._state), "success": False}

    def _step_fallback(
        self, st: int, fuel: int, haz: int, fb: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        p_rec = float(np.clip(self._cfg.fallback_recovery_prob, 0.01, 0.99))
        u = self._rng.random()
        new_h = 0 if u < p_rec else haz
        new_f = fuel - 1
        new_s = st + 1
        return self._finish(haz, new_h, new_s, new_f, 1, used_fallback=True)

    # ---- info helper ----
    @staticmethod
    def _info(state: int) -> dict[str, Any]:
        if state >= N_NON_TERMINAL:
            return {"hazard": False, "terminal": True}
        _st, _f, h, _fb = decode_state(state)
        return {"hazard": bool(h), "terminal": False}


def get_transition_model_and_reward(
    cfg: MissionConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact tabular P(s'|s,a) [N_STATES, N_ACTIONS, N_STATES] and expected immediate reward
    R̄(s,a) [N_STATES, N_ACTIONS] built directly from the proposal spec — used for the
    oracle planner and unit tests. Terminals are absorbing with 0 reward.
    """
    cfg = cfg or MissionConfig()
    p_rec = float(np.clip(cfg.fallback_recovery_prob, 0.01, 0.99))
    hz_pen = cfg.hazard_penalty

    risky_n = _scale_risky_row(RISKY_PROBS_NORMAL, cfg.risky_failure_scale)
    risky_h = _scale_risky_row(RISKY_PROBS_HAZARD, cfg.risky_failure_scale)

    P = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float64)
    R = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    def rcomp(old_h: int, new_h: int, new_stage: int, new_fuel: int, used_fb: bool) -> float:
        rr = 0.0
        if used_fb:
            rr -= 10.0
        rr += 10.0
        if new_stage == 5 and new_fuel > 0:
            rr += 50.0
        if new_h and not old_h:
            rr += hz_pen
        return rr

    def add_outcome(
        s: int,
        a: int,
        prob: float,
        old_h: int,
        new_h: int,
        new_s: int,
        new_f: int,
        new_fb: int,
        used_fb: bool,
    ) -> None:
        if new_s == 5:
            if new_f > 0:
                P[s, a, TERMINAL_SUCCESS] += prob
                R[s, a] += prob * rcomp(old_h, new_h, 5, new_f, used_fb)
            else:
                P[s, a, TERMINAL_FAILURE] += prob
                R[s, a] += prob * (-100.0)
        else:
            sp = encode_state(new_s, new_f, new_h, new_fb)
            P[s, a, sp] += prob
            R[s, a] += prob * rcomp(old_h, new_h, new_s, new_f, used_fb)

    for s in range(N_NON_TERMINAL):
        st, fuel, h, fb = decode_state(s)
        mask = action_mask_int(s)

        for a in range(N_ACTIONS):
            if mask[a] == 0:
                # illegal action: deterministic failure (matches env behavior)
                P[s, a, TERMINAL_FAILURE] = 1.0
                R[s, a] = -100.0
                continue
            if a == ACTION_SAFE:
                add_outcome(s, a, 0.95, h, 0, st + 1, fuel - 1, fb, False)
                add_outcome(s, a, 0.05, h, 1, st + 1, fuel - 1, fb, False)
            elif a == ACTION_RISKY:
                row = risky_h[st - 1] if h else risky_n[st - 1]
                ps, ph, pf = float(row[0]), float(row[1]), float(row[2])
                # catastrophic failure branch
                P[s, a, TERMINAL_FAILURE] += pf
                R[s, a] += pf * (-100.0)
                # safe progress (fuel unchanged, hazard -> normal)
                add_outcome(s, a, ps, h, 0, st + 1, fuel, fb, False)
                # hazard progress (consumes 1 fuel)
                if fuel < 1:
                    P[s, a, TERMINAL_FAILURE] += ph
                    R[s, a] += ph * (-100.0)
                else:
                    add_outcome(s, a, ph, h, 1, st + 1, fuel - 1, fb, False)
            else:  # ACTION_FALLBACK
                add_outcome(s, a, p_rec, h, 0, st + 1, fuel - 1, 1, True)
                add_outcome(s, a, 1.0 - p_rec, h, h, st + 1, fuel - 1, 1, True)

    for t in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
        P[t, :, t] = 1.0
        R[t, :] = 0.0

    # Sanity check: all rows are proper distributions
    row_sums = P.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"P rows do not sum to 1 (max err {np.max(np.abs(row_sums - 1.0)):.2e})"
    )
    return P, R
