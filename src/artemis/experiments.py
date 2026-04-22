"""Training loops, metrics, and the experiment runner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from artemis.constants import N_NON_TERMINAL, TERMINAL_SUCCESS
from artemis.environment import (
    LunarMissionEnv,
    MissionConfig,
    action_mask_int,
    decode_state,
    encode_state,
    get_transition_model_and_reward,
)
from artemis.planning import expected_return_from_state, value_iteration

AgentKind = Literal["qlearning", "model_based", "ucb", "thompson"]


@dataclass
class RunConfig:
    agent: AgentKind
    n_episodes: int = 2000
    gamma: float = 0.95
    seed: int = 0
    mission: MissionConfig | None = None
    # Agent hyperparameter overrides (all optional). Keys understood per agent:
    #   qlearning   : alpha, epsilon
    #   ucb         : alpha, c_ucb
    #   model_based : epsilon, vi_every, pseudocount
    #   thompson    : prior, resample_each_episode
    agent_kwargs: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 200

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mission"] = asdict(self.mission) if self.mission else None
        return d


def _make_agent(kind: AgentKind, gamma: float, seed: int, overrides: dict[str, Any]) -> Any:
    if kind == "qlearning":
        from artemis.agents.q_learning import QLearningAgent

        return QLearningAgent(
            gamma=gamma,
            alpha=overrides.get("alpha", 0.1),
            epsilon=overrides.get("epsilon", 0.1),
            seed=seed,
        )
    if kind == "ucb":
        from artemis.agents.ucb import UCQAgent

        return UCQAgent(
            gamma=gamma,
            alpha=overrides.get("alpha", 0.1),
            c_ucb=overrides.get("c_ucb", 0.5),
            seed=seed,
        )
    if kind == "model_based":
        from artemis.agents.model_based import ModelBasedVIAgent

        return ModelBasedVIAgent(
            gamma=gamma,
            epsilon=overrides.get("epsilon", 0.2),
            vi_every=overrides.get("vi_every", 5),
            pseudocount=overrides.get("pseudocount", 0.01),
            seed=seed,
        )
    # thompson
    from artemis.agents.thompson import ThompsonSamplingMDPAgent

    return ThompsonSamplingMDPAgent(
        gamma=gamma,
        prior=overrides.get("prior", 1.0),
        resample_each_episode=overrides.get("resample_each_episode", True),
        seed=seed,
    )


def _agent_act(agent: Any, s: int, mask: np.ndarray) -> int:
    # unified: every agent exposes act(state, mask=None); passing mask is always OK
    return int(agent.act(s, mask))


def _agent_observe(
    agent: Any,
    s: int,
    a: int,
    r: float,
    s_next: int,
    done: bool,
    mask_next: np.ndarray | None,
) -> None:
    if hasattr(agent, "update"):
        agent.update(s, a, r, s_next, done, mask_next)
    if hasattr(agent, "observe"):
        agent.observe(s, a, r, s_next, done, mask_next)


def run_episode(
    env: LunarMissionEnv,
    agent: Any,
    max_steps: int = 200,
) -> tuple[float, bool, int, int]:
    """
    Run one episode. Calls `agent.new_episode()` first if defined (Thompson resamples here).
    Returns (episode_return, success, steps_in_hazard, n_steps).
    """
    if hasattr(agent, "new_episode"):
        agent.new_episode()
    s, _ = env.reset()
    g = 0.0
    steps = 0
    in_hazard_steps = 0
    while True:
        if s >= N_NON_TERMINAL:
            break
        mask = action_mask_int(s)
        a = _agent_act(agent, s, mask)
        s_next, r, done, _tr, _info = env.step(a)
        g += float(r)
        if s_next < N_NON_TERMINAL:
            _st, _fu, hzn, _ = decode_state(s_next)
            if hzn:
                in_hazard_steps += 1
        mask_next = action_mask_int(s_next) if (not done and s_next < N_NON_TERMINAL) else None
        _agent_observe(agent, s, a, float(r), s_next, bool(done), mask_next)
        s = s_next
        steps += 1
        if done or steps >= max_steps:
            break
    if hasattr(agent, "end_episode"):
        agent.end_episode()
    success = s == TERMINAL_SUCCESS
    return g, bool(success), in_hazard_steps, steps


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0 or w < 1:
        return x.copy()
    w = min(w, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def _undiscounted_oracle_mean(P: np.ndarray, R: np.ndarray, pi: np.ndarray, s0: int) -> float:
    """Expected undiscounted total reward of following pi from s0 on (P,R). Terminates because pi
    eventually reaches an absorbing terminal with prob 1 in this MDP (<= 8 steps)."""
    # Policy-induced P_pi, R_pi
    S = R.shape[0]
    idx = np.arange(S)
    P_pi = P[idx, pi, :]
    R_pi = R[idx, pi]
    # Truncated rollout expectation
    d = np.zeros(S)
    d[s0] = 1.0
    total = 0.0
    for _ in range(50):  # plenty — max path length is 8
        total += float(d @ R_pi)
        d = d @ P_pi
    return total


def run_experiment(rc: RunConfig) -> dict[str, Any]:
    mission = rc.mission or MissionConfig()
    mission = replace(mission, seed=rc.seed)
    env = LunarMissionEnv(mission)

    # Oracle planning on the TRUE MDP at the agent's discount factor
    P, R = get_transition_model_and_reward(mission)
    V_star, pi_star = value_iteration(P, R, gamma=rc.gamma)
    s0 = encode_state(1, mission.start_fuel, 0, 0)
    v_star_s0 = float(V_star[s0])
    g_star_undisc = _undiscounted_oracle_mean(P, R, pi_star, s0)

    agent = _make_agent(rc.agent, rc.gamma, rc.seed, rc.agent_kwargs)

    rets: list[float] = []
    succ: list[float] = []
    hazards: list[int] = []
    ep_steps: list[int] = []
    regret_cum: list[float] = []
    total_regret = 0.0

    for _ep in range(rc.n_episodes):
        g, ok, hsteps, nst = run_episode(env, agent, max_steps=rc.max_steps)
        rets.append(g)
        succ.append(1.0 if ok else 0.0)
        hazards.append(hsteps)
        ep_steps.append(nst)
        # Regret in undiscounted return space (matches episodic return we log)
        total_regret += g_star_undisc - g
        regret_cum.append(total_regret)

    rets_a = np.asarray(rets, dtype=np.float64)
    succ_a = np.asarray(succ, dtype=np.float64)
    window = min(100, max(10, rc.n_episodes // 10))
    return_ma = _moving_avg(rets_a, window)
    success_ma = _moving_avg(succ_a, window)
    learn_speed = -1
    if success_ma.size:
        hit = np.where(success_ma >= 0.8)[0]
        learn_speed = int(hit[0] + window) if len(hit) else -1

    last_win = rets_a[-window:] if len(rets_a) >= window else rets_a
    last_succ = succ_a[-window:] if len(succ_a) >= window else succ_a

    return {
        "v_star_s0_discounted": v_star_s0,
        "g_star_undiscounted_s0": g_star_undisc,
        "returns": rets,
        "success": succ,
        "hazard_steps_per_episode": hazards,
        "ep_steps": ep_steps,
        "return_moving_mean": return_ma.tolist(),
        "success_rate_moving": success_ma.tolist(),
        "moving_avg_window": int(window),
        "cumulative_regret": regret_cum,
        "final_mean_return_last_window": float(np.mean(last_win)),
        "final_success_rate_last_window": float(np.mean(last_succ)),
        "mean_hazard_rate": float(np.sum(hazards) / max(np.sum(ep_steps), 1)),
        "episodes_to_0p8_rolling_success": learn_speed,
        "config": rc.to_dict(),
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_sweep(
    methods: Sequence[AgentKind],
    seeds: Sequence[int],
    n_episodes: int = 1500,
    gamma: float = 0.95,
    out_dir: Path | None = None,
    mission: MissionConfig | None = None,
    agent_kwargs: dict[str, dict[str, Any]] | None = None,
    save: bool = True,
) -> list[dict[str, Any]]:
    """Run every (method, seed) combo and return the list of result dicts."""
    out_dir = out_dir or Path("results")
    akw = agent_kwargs or {}
    all_runs: list[dict[str, Any]] = []
    for m in methods:
        for se in seeds:
            rc = RunConfig(
                agent=m,
                n_episodes=n_episodes,
                gamma=gamma,
                seed=se,
                mission=mission,
                agent_kwargs=dict(akw.get(m, {})),
            )
            res = run_experiment(rc)
            res["method"] = m
            res["seed"] = int(se)
            all_runs.append(res)
            if save:
                save_json(out_dir / f"{m}_seed{se}.json", res)
    return all_runs
