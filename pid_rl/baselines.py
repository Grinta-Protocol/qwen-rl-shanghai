"""
PID-RL PoC — Baseline policies.

Three policies the trained model MUST beat to justify training:

1. RandomPolicy     — uniform sampling within bounds; sometimes HOLDs.
                       Lower bound: "how bad is chaos?"
2. StaticGainsPolicy — always returns current gains (hold). Equivalent to
                       "deploy and walk away". This is what's on-chain today
                       minus the monotonic-drift bug.
3. HeuristicPolicy  — rule-based port of agent/src/reasoning.ts decision
                       framework, using CORRECT bounds from config.GUARD.
                       Upper bound: "can hand-coded rules already solve it?"

All policies return a JSON string matching prompt.ParsedAction schema — they
plug directly into reward.compute_reward without changes.

evaluate_policy() rolls each policy through a set of scenarios and returns
aggregate metrics for comparison with the trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import json
import numpy as np

from config import GUARD, PID, SIM, EMERGENCY
from sim import Simulator, SystemState
from scenarios import Scenario, generate_batch
from reward import compute_reward, RewardBreakdown


# ============================================================================
# Policy interface
# ============================================================================


class Policy(Protocol):
    """A policy returns a JSON completion string given the state."""

    name: str

    def decide(
        self,
        state: SystemState,
        btc_history: np.ndarray,
        deviation_history: list[float],
    ) -> str:
        ...


# ============================================================================
# RandomPolicy
# ============================================================================


class RandomPolicy:
    name = "random"

    def __init__(self, rng: Optional[np.random.Generator] = None, hold_prob: float = 0.3) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.hold_prob = hold_prob

    def decide(
        self,
        state: SystemState,
        btc_history: np.ndarray,
        deviation_history: list[float],
    ) -> str:
        if self.rng.random() < self.hold_prob:
            return json.dumps({
                "action": "hold",
                "new_kp": state.kp,
                "new_ki": state.ki,
                "is_emergency": False,
                "reasoning": "random hold",
            })

        # Sample within delta caps AND absolute bounds
        kp_low = max(GUARD.kp_min, state.kp - GUARD.max_kp_delta)
        kp_high = min(GUARD.kp_max, state.kp + GUARD.max_kp_delta)
        ki_low = max(GUARD.ki_min, state.ki - GUARD.max_ki_delta)
        ki_high = min(GUARD.ki_max, state.ki + GUARD.max_ki_delta)

        new_kp = float(self.rng.uniform(kp_low, kp_high))
        new_ki = float(self.rng.uniform(ki_low, ki_high))

        emergency = self.rng.random() < 0.1
        action = "adjust_emergency" if emergency else "adjust"

        return json.dumps({
            "action": action,
            "new_kp": new_kp,
            "new_ki": new_ki,
            "is_emergency": emergency,
            "reasoning": "random sample",
        })


# ============================================================================
# StaticGainsPolicy
# ============================================================================


class StaticGainsPolicy:
    """Always HOLD — never tune. The 'deploy and walk away' baseline."""

    name = "static"

    def __init__(self, kp: float = PID.kp_baseline, ki: float = PID.ki_baseline) -> None:
        self.kp = kp
        self.ki = ki

    def decide(
        self,
        state: SystemState,
        btc_history: np.ndarray,
        deviation_history: list[float],
    ) -> str:
        return json.dumps({
            "action": "hold",
            "new_kp": self.kp,
            "new_ki": self.ki,
            "is_emergency": False,
            "reasoning": f"static baseline kp={self.kp}, ki={self.ki}",
        })


# ============================================================================
# HeuristicPolicy — rule-based port of reasoning.ts
# ============================================================================


class HeuristicPolicy:
    """
    Rule-based replica of agent/src/reasoning.ts decision framework, using
    the CURRENT on-chain bounds from config.GUARD (not the outdated [1.4, 2.6]).

    Framework (mirrors reasoning.ts lines 52-57):
      1. btc stable + peg stable  →  HOLD
      2. btc dropping 3-10% or dev 1-5%  →  ADJUST (pre-position kp up)
      3. btc crash >10% OR dev >=5%  →  ADJUST_EMERGENCY (aggressive)
      4. recovery (btc up, dev shrinking)  →  ADJUST (kp back toward baseline)

    KP target is a linear interpolation based on BTC drop / deviation magnitude.
    KI is conservative — only nudged during sustained deviation.
    """

    name = "heuristic"

    def __init__(self, lookback_bars: int = 5) -> None:
        self.lookback_bars = lookback_bars

    def _recent_btc_return_pct(self, btc_history: np.ndarray) -> float:
        if len(btc_history) < 2:
            return 0.0
        k = min(self.lookback_bars, len(btc_history) - 1)
        past = float(btc_history[-k - 1])
        now = float(btc_history[-1])
        if past <= 0:
            return 0.0
        return (now - past) / past * 100.0

    def _deviation_trend(self, deviation_history: list[float]) -> float:
        """Positive → deviation growing. Negative → shrinking."""
        if len(deviation_history) < 4:
            return 0.0
        recent = np.mean(np.abs(deviation_history[-2:]))
        older = np.mean(np.abs(deviation_history[-4:-2]))
        return float(recent - older)

    def _clamp_kp(self, kp: float, current_kp: float) -> float:
        kp = max(GUARD.kp_min, min(GUARD.kp_max, kp))
        if kp > current_kp + GUARD.max_kp_delta:
            kp = current_kp + GUARD.max_kp_delta
        if kp < current_kp - GUARD.max_kp_delta:
            kp = current_kp - GUARD.max_kp_delta
        return kp

    def _clamp_ki(self, ki: float, current_ki: float) -> float:
        ki = max(GUARD.ki_min, min(GUARD.ki_max, ki))
        if ki > current_ki + GUARD.max_ki_delta:
            ki = current_ki + GUARD.max_ki_delta
        if ki < current_ki - GUARD.max_ki_delta:
            ki = current_ki - GUARD.max_ki_delta
        return ki

    def decide(
        self,
        state: SystemState,
        btc_history: np.ndarray,
        deviation_history: list[float],
    ) -> str:
        btc_return_pct = self._recent_btc_return_pct(btc_history)
        abs_dev = abs(state.deviation_pct)
        dev_trend = self._deviation_trend(deviation_history)

        # Rule 3: emergency
        if btc_return_pct <= -EMERGENCY.btc_drop_pct_threshold or abs_dev >= EMERGENCY.deviation_pct_threshold:
            # Aggressive kp toward upper bound, but NOT all the way (don't burn ceiling)
            target_kp = min(GUARD.kp_max * 0.7, state.kp + GUARD.max_kp_delta)
            target_ki = state.ki + 0.001  # small ki nudge
            return json.dumps({
                "action": "adjust_emergency",
                "new_kp": self._clamp_kp(target_kp, state.kp),
                "new_ki": self._clamp_ki(target_ki, state.ki),
                "is_emergency": True,
                "reasoning": f"emergency: btc={btc_return_pct:+.1f}%, dev={state.deviation_pct:+.2f}%",
            })

        # Rule 2: btc dropping OR moderate dev → pre-position kp UP
        if btc_return_pct <= -3.0 or abs_dev >= 1.0:
            # Target kp scales with magnitude of stress
            stress = max(abs(btc_return_pct) / 10.0, abs_dev / 5.0)
            target_kp = PID.kp_baseline + stress * (GUARD.kp_max * 0.5 - PID.kp_baseline)
            target_ki = state.ki  # ki stays put
            return json.dumps({
                "action": "adjust",
                "new_kp": self._clamp_kp(target_kp, state.kp),
                "new_ki": self._clamp_ki(target_ki, state.ki),
                "is_emergency": False,
                "reasoning": f"pre-position: btc={btc_return_pct:+.1f}%, dev={state.deviation_pct:+.2f}%",
            })

        # Rule 4: recovery — kp back toward baseline if we're elevated and dev shrinking
        if state.kp > PID.kp_baseline * 1.2 and dev_trend < 0 and abs(btc_return_pct) < 2.0:
            target_kp = state.kp - 0.3  # step back toward baseline
            return json.dumps({
                "action": "adjust",
                "new_kp": self._clamp_kp(max(target_kp, PID.kp_baseline), state.kp),
                "new_ki": self._clamp_ki(state.ki, state.ki),
                "is_emergency": False,
                "reasoning": f"recovery: kp={state.kp:.2f} → lower (dev shrinking)",
            })

        # Rule 1: everything fine → HOLD
        return json.dumps({
            "action": "hold",
            "new_kp": state.kp,
            "new_ki": state.ki,
            "is_emergency": False,
            "reasoning": "stable — hold",
        })


# ============================================================================
# Evaluation harness
# ============================================================================


@dataclass
class ScenarioResult:
    scenario_type: str
    description: str
    total_reward: float
    mean_abs_deviation_pct: float
    max_abs_deviation_pct: float
    rate_std: float
    min_collat_ratio: float
    time_outside_band_pct: float
    n_actions_accepted: int
    n_actions_rejected: int
    n_parse_errors: int
    final_kp: float
    final_ki: float
    kp_range: tuple[float, float]
    monotonic_kp: bool           # True if kp never decreased in the run


@dataclass
class EvalResult:
    policy_name: str
    per_scenario: list[ScenarioResult]
    mean_reward: float = 0.0
    mean_abs_deviation: float = 0.0
    fraction_monotonic: float = 0.0
    accept_rate: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== {self.policy_name} ===",
            f"  mean_reward:              {self.mean_reward:+8.3f}",
            f"  mean_abs_deviation_pct:   {self.mean_abs_deviation:8.3f}",
            f"  fraction_monotonic:       {self.fraction_monotonic:8.2%}",
            f"  accept_rate:              {self.accept_rate:8.2%}",
        ]
        return "\n".join(lines)


def evaluate_policy(
    policy: Policy,
    scenarios: list[Scenario],
    action_every_n_steps: int = 10,
    seed: int = 42,
    verbose: bool = False,
) -> EvalResult:
    """
    Run policy on each scenario, firing a decision every `action_every_n_steps`.
    Aggregates per-step rewards (via reward.compute_reward on fresh sim clones).
    """
    per_scenario: list[ScenarioResult] = []

    for idx, scenario in enumerate(scenarios):
        sim = Simulator(
            btc_path=scenario.btc_path,
            initial_kp=scenario.initial_kp,
            initial_ki=scenario.initial_ki,
            rng=np.random.default_rng(seed=seed + idx),
        )
        deviation_history: list[float] = []
        total_reward = 0.0
        n_accepted = 0
        n_rejected = 0
        n_parse_errors = 0
        all_devs: list[float] = []
        all_rates: list[float] = []
        all_collats: list[float] = []

        n_steps = min(len(scenario.btc_path) - 1, SIM.episode_steps - 1)
        for step in range(n_steps):
            state = sim.get_state()
            deviation_history.append(state.deviation_pct)
            all_devs.append(state.deviation_pct)
            all_rates.append(state.redemption_rate)
            all_collats.append(state.collat_ratio)

            # Fire a decision at the cadence
            if step % action_every_n_steps == 0:
                btc_hist = scenario.btc_path[: sim.step_idx + 1]
                completion = policy.decide(
                    state=state,
                    btc_history=btc_hist,
                    deviation_history=deviation_history,
                )
                br: RewardBreakdown = compute_reward(completion, sim)
                total_reward += br.total
                if br.parse_error:
                    n_parse_errors += 1
                else:
                    # Actually commit the action on the real sim (reward used a clone)
                    import json as _json
                    try:
                        parsed = _json.loads(_extract_json_simple(completion) or "{}")
                        if parsed.get("action") != "hold":
                            result = sim.apply_action(
                                requested_kp=float(parsed.get("new_kp", state.kp)),
                                requested_ki=float(parsed.get("new_ki", state.ki)),
                                is_emergency=bool(parsed.get("is_emergency", False)),
                            )
                            if result.accepted:
                                n_accepted += 1
                            else:
                                n_rejected += 1
                    except (ValueError, TypeError):
                        n_parse_errors += 1

            sim.step()

        # Terminal metrics
        metrics = sim.get_history_metrics(
            [sim.get_state()] if not all_devs else
            # Rebuild a fake history with arrays we captured
            [
                SystemState(
                    t=0, market_price=0, redemption_price=0, btc_price=0,
                    kp=0, ki=0, last_proportional=0, last_integral=0,
                    redemption_rate=r, collat_ratio=c,
                    guard_update_count=0, steps_since_last_update=0,
                    deviation_pct=d, collateral_drop_pct=0,
                )
                for d, r, c in zip(all_devs, all_rates, all_collats)
            ]
        )

        kps = [g[0] for g in sim.gain_history]
        monotonic_kp = all(kps[i + 1] >= kps[i] for i in range(len(kps) - 1))

        per_scenario.append(ScenarioResult(
            scenario_type=scenario.scenario_type,
            description=scenario.description,
            total_reward=total_reward,
            mean_abs_deviation_pct=metrics["mean_abs_deviation_pct"],
            max_abs_deviation_pct=metrics["max_abs_deviation_pct"],
            rate_std=metrics["rate_std"],
            min_collat_ratio=metrics["min_collat_ratio"],
            time_outside_band_pct=metrics["time_outside_band_pct"],
            n_actions_accepted=n_accepted,
            n_actions_rejected=n_rejected,
            n_parse_errors=n_parse_errors,
            final_kp=sim.pid.state.kp,
            final_ki=sim.pid.state.ki,
            kp_range=(min(kps), max(kps)),
            monotonic_kp=monotonic_kp and len(kps) > 1,
        ))

        if verbose:
            print(f"  [{idx:3d}] {scenario.scenario_type:12s} "
                  f"reward={total_reward:+7.2f} "
                  f"dev={metrics['mean_abs_deviation_pct']:.3f}% "
                  f"kp=[{min(kps):.2f},{max(kps):.2f}] "
                  f"acc/rej/err={n_accepted}/{n_rejected}/{n_parse_errors}")

    result = EvalResult(policy_name=policy.name, per_scenario=per_scenario)
    if per_scenario:
        result.mean_reward = float(np.mean([r.total_reward for r in per_scenario]))
        result.mean_abs_deviation = float(np.mean([r.mean_abs_deviation_pct for r in per_scenario]))
        result.fraction_monotonic = float(np.mean([r.monotonic_kp for r in per_scenario]))
        total_actions = sum(r.n_actions_accepted + r.n_actions_rejected for r in per_scenario)
        total_accepted = sum(r.n_actions_accepted for r in per_scenario)
        result.accept_rate = (total_accepted / total_actions) if total_actions > 0 else 0.0
    return result


def _extract_json_simple(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# ============================================================================
# Smoke test
# ============================================================================


def _smoke_test() -> None:
    print("=" * 60)
    print("SMOKE TEST — 3 baselines × 12 scenarios (all types)")
    print("=" * 60)
    scenarios = generate_batch(n=12, seed=42)

    rng = np.random.default_rng(seed=99)
    policies: list[Policy] = [
        RandomPolicy(rng=rng),
        StaticGainsPolicy(),
        HeuristicPolicy(),
    ]

    results: list[EvalResult] = []
    for pol in policies:
        print(f"\n--- evaluating {pol.name} ---")
        res = evaluate_policy(pol, scenarios, action_every_n_steps=10, verbose=True)
        results.append(res)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(r.summary())

    print()
    print("ORDERING (higher reward = better):")
    ranked = sorted(results, key=lambda r: -r.mean_reward)
    for i, r in enumerate(ranked):
        print(f"  #{i + 1}  {r.policy_name:10s}  mean_reward={r.mean_reward:+8.3f}")

    print()
    print("OK — baseline smoke test done.")


if __name__ == "__main__":
    _smoke_test()
