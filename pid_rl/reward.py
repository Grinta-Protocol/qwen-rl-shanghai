"""
PID-RL PoC — Reward function.

The reward is computed OVER A WINDOW post-action: we apply the proposed gains,
step the Simulator forward REWARD.reward_window_steps, then aggregate metrics.

Formula (from config.RewardWeights):
    reward =
        - α * mean(|deviation_pct|)        # primary: system health
        - β * std(redemption_rate)          # rate predictability
        - γ * ||Δgains||                    # action regularization
        - δ * bounds_violation              # guardrail violations
        - ε * monotonic_drift_penalty       # prevent the kp-ceiling bug
        + bonus_justified_emergency         # correct ADJUST_EMERGENCY
        - penalty_false_alarm               # incorrect ADJUST_EMERGENCY
    + malformed_output_penalty              # if parse fails

All components are captured in RewardBreakdown for debugging + reward-shaping.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import REWARD, EMERGENCY, SIM
from sim import Simulator, SystemState
from prompt import ParsedAction, ParseError, parse_output


# ============================================================================
# Breakdown container — useful for logging and reward-shaping iteration
# ============================================================================


@dataclass
class RewardBreakdown:
    total: float = 0.0
    deviation_term: float = 0.0          # -α * mean|dev|
    rate_std_term: float = 0.0           # -β * std(rate)
    action_mag_term: float = 0.0         # -γ * ||Δgains||
    bounds_term: float = 0.0             # -δ * bounds_violation
    monotonic_term: float = 0.0          # -ε * monotonic_penalty
    emergency_term: float = 0.0          # +bonus or -penalty
    malformed_term: float = 0.0          # -10 if parse error
    metrics: dict = field(default_factory=dict)
    parse_error: Optional[str] = None


# ============================================================================
# Helpers
# ============================================================================


def _is_actual_emergency(
    state_before: SystemState,
    state_after_window: list[SystemState],
    btc_path: np.ndarray,
    step_idx: int,
    lookback_bars: int = 5,
) -> bool:
    """
    An emergency IS justified if any of:
      - pre-action deviation above threshold
      - recent BTC move (last `lookback_bars`) drops >= threshold
      - post-action window reveals the emergency was impending

    We use RECENT BTC move, not baseline comparison — a scenario starting at
    a low BTC price is not automatically an emergency.
    """
    if abs(state_before.deviation_pct) >= EMERGENCY.deviation_pct_threshold:
        return True

    lookback = min(lookback_bars, step_idx)
    if lookback >= 1:
        past = float(btc_path[step_idx - lookback])
        now = float(btc_path[step_idx])
        if past > 0:
            recent_return_pct = (now - past) / past * 100.0
            if recent_return_pct <= -EMERGENCY.btc_drop_pct_threshold:
                return True

    max_dev_after = max((abs(s.deviation_pct) for s in state_after_window), default=0.0)
    if max_dev_after >= EMERGENCY.deviation_pct_threshold:
        return True
    return False


def _monotonic_drift_penalty(gain_history: list[tuple[float, float]]) -> float:
    """
    Continuous drift signal in [0, 1]. Scans the tail of gain history for the
    longest run of same-direction kp moves. Penalty = (run_length / window)².

    Rationale: a binary "5 in a row" flag is too coarse — the agent gets zero
    signal at run=4 and max signal at run=5. Quadratic gives gradient:
      run=2 → 0.04 × ε   (noise)
      run=3 → 0.36 × ε
      run=4 → 0.64 × ε
      run=5 → 1.00 × ε   (full penalty at the configured window)

    Zero deltas (holds) break the streak — you can't drift by standing still.
    """
    window = REWARD.monotonic_window
    if len(gain_history) < 2:
        return 0.0

    # All kp deltas, newest last
    diffs = [gain_history[i + 1][0] - gain_history[i][0] for i in range(len(gain_history) - 1)]
    if not diffs:
        return 0.0

    # Longest trailing same-direction run (ignoring zeros which reset the streak)
    run = 0
    sign = 0
    for d in reversed(diffs):
        if abs(d) < 1e-9:
            break
        cur_sign = 1 if d > 0 else -1
        if sign == 0:
            sign = cur_sign
            run = 1
        elif cur_sign == sign:
            run += 1
        else:
            break

    ratio = min(run, window) / window
    return ratio * ratio


# ============================================================================
# Core reward computation
# ============================================================================


def compute_reward(
    completion: str,
    sim: Simulator,
    reward_window_steps: Optional[int] = None,
) -> RewardBreakdown:
    """
    Evaluate one completion against one Simulator state.

    IMPORTANT: this DEEPCOPIES the sim so repeated evaluation of different
    completions from the same state gives independent rewards (needed for GRPO
    groups). The passed `sim` is NOT modified.
    """
    if reward_window_steps is None:
        reward_window_steps = SIM.reward_window_steps

    breakdown = RewardBreakdown()

    # 1. Parse ----------------------------------------------------------------
    parsed = parse_output(completion)
    if isinstance(parsed, ParseError):
        breakdown.malformed_term = REWARD.malformed_output_penalty
        breakdown.total = REWARD.malformed_output_penalty
        breakdown.parse_error = parsed.reason
        return breakdown

    assert isinstance(parsed, ParsedAction)

    # 2. Clone sim, snapshot pre-state ---------------------------------------
    sim_clone = copy.deepcopy(sim)
    state_before = sim_clone.get_state()

    # 3. Apply action (guard runs here) --------------------------------------
    # For "hold", we still call apply_action with current gains — action_magnitude
    # will be 0 and guard will accept (cheap), keeping logic uniform.
    if parsed.action == "hold":
        req_kp, req_ki = state_before.kp, state_before.ki
    else:
        req_kp, req_ki = parsed.new_kp, parsed.new_ki

    action_result = sim_clone.apply_action(
        requested_kp=req_kp,
        requested_ki=req_ki,
        is_emergency=parsed.is_emergency,
    )

    # 4. Roll forward the reward window --------------------------------------
    history = sim_clone.run_forward(n_steps=reward_window_steps)
    metrics = sim_clone.get_history_metrics(history)
    breakdown.metrics = metrics

    # 5. Components ----------------------------------------------------------
    # (a) deviation (primary)
    breakdown.deviation_term = -REWARD.alpha_deviation * metrics["mean_abs_deviation_pct"]

    # (b) rate stability
    # rate_std is in per-second units; magnify to be visible in reward
    breakdown.rate_std_term = -REWARD.beta_rate_std * metrics["rate_std"] * 1e6

    # (c) action magnitude — only penalize if action != hold
    if parsed.action != "hold":
        breakdown.action_mag_term = -REWARD.gamma_action_mag * action_result.action_magnitude

    # (d) bounds violations (absolute + delta + cooldown mismatch)
    breakdown.bounds_term = -REWARD.delta_bounds_violation * action_result.bounds_violation

    # (e) monotonic drift
    monotonic_p = _monotonic_drift_penalty(sim_clone.gain_history)
    breakdown.monotonic_term = -REWARD.epsilon_monotonic * monotonic_p

    # (f) emergency classification
    actual_emergency = _is_actual_emergency(
        state_before=state_before,
        state_after_window=history,
        btc_path=sim_clone.btc_path,
        step_idx=sim.step_idx,
    )
    declared_emergency = parsed.is_emergency or parsed.action == "adjust_emergency"
    if declared_emergency and actual_emergency:
        breakdown.emergency_term = REWARD.bonus_justified_emergency
    elif declared_emergency and not actual_emergency:
        breakdown.emergency_term = -REWARD.penalty_false_alarm
    # Missed emergency (not declared but actual) is implicitly punished by the
    # deviation term — post-action deviation will be high.

    breakdown.total = (
        breakdown.deviation_term
        + breakdown.rate_std_term
        + breakdown.action_mag_term
        + breakdown.bounds_term
        + breakdown.monotonic_term
        + breakdown.emergency_term
    )
    return breakdown


# ============================================================================
# GRPO-shaped wrapper (Unsloth/TRL interface)
# ============================================================================


def pid_reward_func(
    prompts: list,
    completions: list,
    sims: Optional[list[Simulator]] = None,
    **kwargs,
) -> list[float]:
    """
    GRPO-shaped reward function.

    `sims[i]` must correspond to `completions[i]` — the training loop is
    responsible for building the sim list (usually: one sim per prompt, broadcast
    to each of the G generations in the group).

    Returns a flat list of floats (one per completion).
    """
    if sims is None or len(sims) != len(completions):
        raise ValueError(
            f"pid_reward_func requires sims list matching completions length "
            f"(got sims={len(sims) if sims else 0}, completions={len(completions)})"
        )

    rewards: list[float] = []
    for completion, sim in zip(completions, sims):
        # Unsloth may wrap completion as list of message dicts — normalize
        text = _normalize_completion(completion)
        rewards.append(compute_reward(text, sim).total)
    return rewards


def _normalize_completion(c) -> str:
    """Unsloth/TRL sometimes passes a chat-formatted list of dicts."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for m in c:
            if isinstance(m, dict) and "content" in m:
                parts.append(str(m["content"]))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(c)


# ============================================================================
# Smoke test
# ============================================================================


def _smoke_test() -> None:
    from sim import Simulator
    from scenarios import _gen_flash_crash, _gen_stable

    rng = np.random.default_rng(seed=42)

    print("=" * 60)
    print("SMOKE TEST — happy-path reward on flash crash")
    print("=" * 60)
    scenario = _gen_flash_crash(rng, SIM.episode_steps)
    sim = Simulator(
        btc_path=scenario.btc_path,
        initial_kp=scenario.initial_kp,
        initial_ki=scenario.initial_ki,
        rng=np.random.default_rng(seed=7),
    )
    sim.run_forward(n_steps=50)

    # A reasonable action: increase kp during crash
    good = '{"action":"adjust","new_kp":3.5,"new_ki":0.004,"is_emergency":false,"reasoning":"BTC drop, strengthen response"}'
    br = compute_reward(good, sim)
    _print_breakdown("good adjust", br)

    hold = '{"action":"hold","new_kp":2.0,"new_ki":0.002,"is_emergency":false,"reasoning":"monitor"}'
    br_hold = compute_reward(hold, sim)
    _print_breakdown("hold", br_hold)

    # Out-of-bounds action
    bad = '{"action":"adjust","new_kp":15.0,"new_ki":0.5,"is_emergency":true,"reasoning":"panic"}'
    br_bad = compute_reward(bad, sim)
    _print_breakdown("bounds-violating", br_bad)

    # Malformed
    br_malformed = compute_reward("sorry I don't know", sim)
    _print_breakdown("malformed", br_malformed)

    print()
    print("=" * 60)
    print("SMOKE TEST — false alarm emergency on stable scenario")
    print("=" * 60)
    stable = _gen_stable(rng, SIM.episode_steps)
    sim_s = Simulator(
        btc_path=stable.btc_path,
        initial_kp=stable.initial_kp,
        initial_ki=stable.initial_ki,
        rng=np.random.default_rng(seed=7),
    )
    sim_s.run_forward(n_steps=50)
    false_alarm = '{"action":"adjust_emergency","new_kp":4.0,"new_ki":0.005,"is_emergency":true,"reasoning":"everything on fire"}'
    br_fa = compute_reward(false_alarm, sim_s)
    _print_breakdown("false alarm on stable", br_fa)

    print()
    print("=" * 60)
    print("SMOKE TEST — pid_reward_func batched (GRPO shape)")
    print("=" * 60)
    sim_copy = copy.deepcopy(sim)
    completions = [good, hold, bad, "garbage"]
    sims = [sim_copy] * len(completions)
    rewards = pid_reward_func(prompts=[""] * len(completions), completions=completions, sims=sims)
    for c, r in zip(["good", "hold", "bad", "garbage"], rewards):
        print(f"  {c:10s} reward={r:+8.4f}")

    print()
    print("OK — reward smoke tests done.")


def _print_breakdown(label: str, br: RewardBreakdown) -> None:
    print(f"\n[{label}]  total={br.total:+8.4f}")
    if br.parse_error:
        print(f"    parse_error: {br.parse_error}")
        return
    print(f"    deviation:     {br.deviation_term:+8.4f}")
    print(f"    rate_std:      {br.rate_std_term:+8.4f}")
    print(f"    action_mag:    {br.action_mag_term:+8.4f}")
    print(f"    bounds:        {br.bounds_term:+8.4f}")
    print(f"    monotonic:     {br.monotonic_term:+8.4f}")
    print(f"    emergency:     {br.emergency_term:+8.4f}")
    for k, v in br.metrics.items():
        print(f"    {k:25s} {v:.4f}")


if __name__ == "__main__":
    _smoke_test()
