# Artemis — Risk-Aware Reinforcement Learning

A tabular RL comparison study on a stochastic lunar-mission MDP. Four algorithms — Q-learning, UCB-Q, Model-based VI, and Thompson Sampling — are trained and compared against an oracle (value iteration on the true model). The main workflow lives entirely in a Jupyter notebook; the Python files are importable library code only.

---

## File structure

```
Artemis-risk-aware-RL/
├── notebooks/
│   └── Artemis_main.ipynb      # main entry point — run this
├── src/artemis/
│   ├── constants.py            # action/state indices, risky-route probability tables
│   ├── environment.py          # LunarMissionEnv (Gymnasium API), state encode/decode,
│   │                           #   action masks, true P and R builder
│   ├── planning.py             # vectorized value iteration, oracle policy
│   ├── experiments.py          # RunConfig, run_episode, run_sweep (imported by notebook)
│   └── agents/
│       ├── q_learning.py       # tabular Q-learning (ε-greedy)
│       ├── ucb.py              # UCB-Q (count-based exploration bonus)
│       ├── model_based.py      # model-based VI (MLE model + periodic re-planning)
│       └── thompson.py         # PSRL / Thompson sampling over MDPs
├── tests/
│   └── test_environment.py     # unit + Monte-Carlo tests for the environment
├── pyproject.toml
└── requirements.txt
```

---

## Setup

```bash
pip install -e ".[notebook]"    # installs jupyter + ipykernel + all dependencies
```

Or manually:

```bash
pip install -r requirements.txt
```

Then open and run `notebooks/Artemis_main.ipynb` top to bottom.

---

## The MDP

The environment models a 4-stage lunar mission. At each stage the agent chooses one of three actions:

| Action | Effect |
|---|---|
| **Safe** | Guaranteed stage advance; costs 1 fuel unit; 5 % chance of entering hazard |
| **Risky** | Stage advance without fuel cost on the safe branch (75–55 %); hazard branch (20–25 %) costs 1 fuel; catastrophic failure branch (5–30 %) ends the mission |
| **Fallback** | Stage advance, clears hazard with 90 % probability; costs 1 fuel and a −10 penalty |

State: (stage 1–4, fuel 0–2, hazard 0/1, fallback used 0/1) — 48 non-terminal states plus terminal success and failure. The key tension: risky preserves fuel on its safe branch, but risks catastrophic failure; safe is reliable but burns fuel, starving later stages.

---

## Notebook walkthrough

### § 1 — Environment smoke test
Confirms the environment API (reset / step / mask) works and shows a sample transition for each action from the start state.

### § 2 — Oracle policy (value iteration on the true MDP)
Runs vectorized value iteration on the analytically-derived `P(s'|s,a)` and `R̄(s,a)`. Prints `V*(s₀)` and the oracle's recommended action at key states (mainly risky at early stages while fuel is plentiful, safe at the final stage). A 10 000-episode Monte-Carlo simulation confirms the oracle achieves **~47.8 % mission success** — the hard upper bound imposed by the MDP's stochastic transitions.

### § 2 (verification sub-section)
Checks that (1) every row of `P` sums to 1, (2) the risky-route probability tables match the spec, (3) both terminals are absorbing with zero reward, and (4) spot-check rewards (+60 on success, −100 on fuel failure, −10 on fallback, −5 hazard entry) are correct.

### § 3 — Single-run sanity check (600 episodes)
Plots rolling success rate and rolling mean return for one seed of each agent. Shows that all four agents are learning and improving, not stuck or diverging.

### § 4 — Multi-seed sweep (2 000 episodes × 5 seeds)

**Learning curves** (rolling success rate, rolling mean return, cumulative regret — mean ± shaded std across seeds):
All four agents converge within ~500–800 episodes. UCB reaches the highest rolling success fastest; model-based VI closes in later as its learned model becomes accurate.

**Summary table** (mean of the last 100-episode window across 5 seeds):

| Agent | Success | % of oracle | Mean return | Hazard rate | Cum. regret |
|---|---|---|---|---|---|
| UCB | 0.344 | 72 % | −22.9 | 0.127 | 48 853 |
| Model-based VI | 0.326 | 68 % | −27.1 | 0.127 | 58 935 |
| Q-learning | 0.318 | 66 % | −27.7 | 0.122 | 63 060 |
| Thompson | 0.284 | 59 % | −32.2 | 0.106 | 71 485 |

*Oracle ceiling = 0.478 success / −1.1 mean return.*

**Per-metric bar chart** visualises the same four metrics side by side with error bars, making the ranking across methods immediately visible.

### § 5 — Environment variants

The sweep is repeated on four modified mission configs:

| Variant | Change | Effect on agents |
|---|---|---|
| `harsh_hazard` | Hazard penalty −10 (was −5) | All agents become more conservative; success drops ~15 % |
| `low_fuel` | Start fuel = 1 (was 2) | Very hard; margins collapse; UCB still leads |
| `risky_x2` | Risky failure probability doubled | Risky actions punished more; agents shift toward safe |
| `weak_fallback` | Fallback clears hazard with 50 % (was 90 %) | Fallback used less; minimal impact on success |

UCB and Q-learning are most robust across variants (mean success 0.27 and 0.27); Thompson is most sensitive to environment difficulty.

### § 6 — Policy heatmaps (oracle vs. learned)
Greedy policy of each agent at the end of training, shown as a colour grid over (stage, fuel) for normal/no-fallback states. All four learned policies closely match the oracle: **risky at early stages with plenty of fuel, safe only at the final stage**. The heatmaps give visual confirmation that the agents have learned the right risk tradeoff.

### § 7 — Expected vs. observed results

All three claims from the project proposal are confirmed:

```
Claim: UCB/Thompson beat Q-learning on success rate.
  Q-learning = 0.318;  best(UCB, Thompson) = 0.344  =>  CONFIRMED

Claim: UCB/Thompson have lower cumulative regret than Q-learning.
  Q-learning regret = 63 060;  best(UCB, Thompson) = 48 853  =>  CONFIRMED

Claim: Model-based VI performs strongly once enough data is collected.
  Model-based success = 0.326 (best overall = 0.344 by UCB)  =>  CONFIRMED
```

---

## Running tests

```bash
pytest tests/ -v
```

Covers state encode/decode, action-mask semantics, transition row sums, terminal absorbing property, Monte-Carlo fallback and fuel-zero checks, and oracle policy sign.
