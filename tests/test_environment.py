import numpy as np
import pytest

from artemis.constants import (
    ACTION_FALLBACK,
    ACTION_RISKY,
    ACTION_SAFE,
    N_NON_TERMINAL,
    N_STATES,
    TERMINAL_FAILURE,
    TERMINAL_SUCCESS,
)
from artemis.environment import (
    LunarMissionEnv,
    MissionConfig,
    action_mask_int,
    decode_state,
    encode_state,
    get_transition_model_and_reward,
)
from artemis.planning import (
    expected_return_from_state,
    optimal_policy_for_mission,
    value_iteration,
)


# ---------- encoding/decoding ----------
def test_encode_decode_roundtrip():
    for s in range(1, 5):
        for f in (0, 1, 2):
            for h in (0, 1):
                for fb in (0, 1):
                    o = encode_state(s, f, h, fb)
                    assert decode_state(o) == (s, f, h, fb)


def test_encode_decode_out_of_range_raises():
    with pytest.raises(ValueError):
        decode_state(TERMINAL_SUCCESS)
    with pytest.raises(ValueError):
        encode_state(0, 0, 0, 0)


# ---------- action masks ----------
def test_action_mask_semantics():
    # fuel=2, no hazard, fallback available
    m = action_mask_int(encode_state(1, 2, 0, 0))
    assert list(m) == [1, 1, 1]
    # fuel=0 forbids safe and fallback, risky still allowed
    m = action_mask_int(encode_state(1, 0, 0, 0))
    assert list(m) == [0, 1, 0]
    # fallback already used forbids fallback
    m = action_mask_int(encode_state(1, 2, 0, 1))
    assert list(m) == [1, 1, 0]
    # terminals mask to all zeros
    assert list(action_mask_int(TERMINAL_SUCCESS)) == [0, 0, 0]
    assert list(action_mask_int(TERMINAL_FAILURE)) == [0, 0, 0]


# ---------- transition model sanity ----------
def test_transition_row_sums_and_shape():
    P, R = get_transition_model_and_reward(MissionConfig())
    assert P.shape == (N_STATES, 3, N_STATES)
    assert R.shape == (N_STATES, 3)
    row_err = np.max(np.abs(P.sum(axis=-1) - 1.0))
    assert row_err < 1e-10, f"max row-sum error {row_err}"


def test_terminals_are_absorbing():
    P, R = get_transition_model_and_reward(MissionConfig())
    for t in (TERMINAL_SUCCESS, TERMINAL_FAILURE):
        for a in range(3):
            assert P[t, a, t] == 1.0
            assert R[t, a] == 0.0


# ---------- Monte-Carlo vs. analytical P ----------
def test_monte_carlo_safe_hazard():
    env = LunarMissionEnv(MissionConfig(), seed=0)
    n = 30_000
    cnormal = 0
    for _ in range(n):
        env.reset()
        s, _, d, _, _ = env.step(ACTION_SAFE)
        assert s < N_NON_TERMINAL and not d
        _, _, haz, _ = decode_state(s)
        cnormal += 1 if haz == 0 else 0
    frac = cnormal / n
    assert 0.92 < frac < 0.98, frac


def test_monte_carlo_risky_stage1():
    env = LunarMissionEnv(MissionConfig(), seed=42)
    n = 50_000
    safe_p, hbr, f = 0, 0, 0
    for _ in range(n):
        env.reset()
        s, _, _, _, _ = env.step(ACTION_RISKY)
        if s == TERMINAL_FAILURE:
            f += 1
        else:
            st, _, hz, _ = decode_state(s)
            assert st == 2
            if hz == 0:
                safe_p += 1
            else:
                hbr += 1
    assert safe_p + hbr + f == n
    assert 0.72 < safe_p / n < 0.78
    assert 0.18 < hbr / n < 0.22
    assert 0.03 < f / n < 0.08


def test_monte_carlo_fallback_clears_hazard():
    cfg = MissionConfig(fallback_recovery_prob=0.9)
    env = LunarMissionEnv(cfg, seed=7)
    n = 20_000
    recovered = 0
    for _ in range(n):
        env.set_state(encode_state(2, 2, 1, 0))  # stage2, fuel2, hazard, fb available
        s, _, _, _, _ = env.step(ACTION_FALLBACK)
        assert s < N_NON_TERMINAL
        _, _, hz, fb = decode_state(s)
        assert fb == 1  # fallback now consumed
        if hz == 0:
            recovered += 1
    frac = recovered / n
    assert 0.88 < frac < 0.92, frac


# ---------- rewards ----------
def test_success_reward_60_from_stage4():
    env = LunarMissionEnv(MissionConfig(), seed=123)
    for _ in range(2000):
        env.set_state(encode_state(4, 1, 0, 0))
        s2, r, _, _, _ = env.step(ACTION_RISKY)
        if s2 == TERMINAL_SUCCESS:
            assert abs(r - 60.0) < 1e-9
            return
    pytest.fail("no success in 2000 tries")


def test_fuel_zero_safe_causes_failure():
    env = LunarMissionEnv(MissionConfig(), seed=0)
    env.set_state(encode_state(1, 0, 0, 0))
    s, r, d, _, _ = env.step(ACTION_SAFE)
    assert d and s == TERMINAL_FAILURE and r == -100.0


def test_fallback_already_used_causes_failure():
    env = LunarMissionEnv(MissionConfig(), seed=0)
    env.set_state(encode_state(1, 2, 0, 1))
    s, r, d, _, _ = env.step(ACTION_FALLBACK)
    assert d and s == TERMINAL_FAILURE and r == -100.0


# ---------- planning ----------
def test_value_iteration_shapes_and_terminals():
    P, R = get_transition_model_and_reward(MissionConfig())
    V, pi = value_iteration(P, R, gamma=0.95)
    assert V.shape == (N_STATES,)
    assert pi.shape == (N_STATES,)
    assert V[TERMINAL_SUCCESS] == 0 and V[TERMINAL_FAILURE] == 0


def test_oracle_policy_positive_from_start():
    V, pi, P, R = optimal_policy_for_mission(MissionConfig(), gamma=0.95)
    s0 = encode_state(1, 2, 0, 0)
    assert V[s0] > 0, "oracle value from start state should be clearly positive"
    # Policy evaluation on true MDP matches V*
    g = expected_return_from_state(P, R, pi, s0, gamma=0.95)
    assert abs(g - V[s0]) < 1e-6
