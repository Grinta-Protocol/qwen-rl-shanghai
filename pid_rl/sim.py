"""
PID-RL PoC — Simulator

Pure-Python port of the PID controller logic from
/mnt/c/Users/henry/desktop/pid/src/pid_controller.cairo

Plus a simple market model and collateral accounting for the reward.

Units convention (see config.py):
    proportional     : dimensionless relative deviation (e.g. 0.05 = 5%)
    integral         : dev-seconds
    kp, ki           : human decimals (2.0, 0.002)
    pi_output        : WAD-equivalent mixed units (matches contract)
    rate_per_second  : dimensionless per-second rate factor (1.0 = no change)
    prices           : USD floats

The float math mirrors the Cairo integer math with the RAY/WAD scaling:
in Cairo, `rate = RAY + pi_output_wad` where RAY=1e27 and pi_output is in
WAD (1e18). That's equivalent to `rate_per_second = 1 + pi_output/1e9`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import PID, MARKET, SIM, GUARD, RAY_WAD_RATIO


# ============================================================================
# State containers
# ============================================================================


@dataclass
class PIDState:
    """Matches the DeviationObservation + ControllerGains storage in Cairo."""
    kp: float
    ki: float
    last_proportional: float = 0.0
    last_integral: float = 0.0
    last_deviation_timestamp: float = 0.0  # sim time (seconds) of last update


@dataclass
class SystemState:
    """
    Maps 1:1 to agent/src/monitor.ts ProtocolState.
    This is what the RL agent sees (after feature engineering in prompt.py).
    """
    t: float                      # sim time (seconds)
    market_price: float           # GRIT/USD
    redemption_price: float       # GRIT target
    btc_price: float              # collateral price
    kp: float
    ki: float
    last_proportional: float
    last_integral: float
    redemption_rate: float        # per-second rate factor (1.0 = no change)
    collat_ratio: float
    guard_update_count: int
    steps_since_last_update: int
    deviation_pct: float          # peg deviation in %
    collateral_drop_pct: float    # BTC drop vs baseline in %


# ============================================================================
# PID Controller (port of pid_controller.cairo)
# ============================================================================


class PIDController:
    """Pure-Python PID, faithful to pid_controller.cairo."""

    def __init__(self, kp: float, ki: float) -> None:
        self.state = PIDState(kp=kp, ki=ki)

    def set_gains(self, kp: float, ki: float) -> None:
        self.state.kp = kp
        self.state.ki = ki

    def _proportional_term(self, market_price: float, redemption_price: float) -> float:
        """
        (redemption - market) / redemption — dimensionless.
        Positive when market < redemption (peg below target).
        """
        if redemption_price <= 0:
            return 0.0
        return (redemption_price - market_price) / redemption_price

    def _next_integral(
        self, proportional: float, time_elapsed: float
    ) -> tuple[float, float]:
        """
        Trapezoidal integration with per-second leak, matching Cairo.

        Returns (new_integral, new_time_adjusted).
        """
        avg_deviation = (proportional + self.state.last_proportional) / 2.0
        new_time_adjusted = avg_deviation * time_elapsed

        leak_factor = PID.per_second_leak ** time_elapsed
        leaked_integral = self.state.last_integral * leak_factor

        new_integral = leaked_integral + new_time_adjusted
        return new_integral, new_time_adjusted

    def _pi_output(self, proportional: float, integral: float) -> float:
        """kp*P + ki*I in WAD-equivalent units (matches Cairo semantics)."""
        return self.state.kp * proportional + self.state.ki * integral

    def _breaks_noise_barrier(self, pi_output: float, redemption_price: float) -> bool:
        """
        |pi_output| >= (2 - noise) * redemption_price - redemption_price
                     = (1 - noise) * redemption_price
        With noise=0.995 → threshold = 0.005 * redemption_price.
        """
        if pi_output == 0.0:
            return False
        threshold = (1.0 - PID.noise_barrier) * redemption_price
        return abs(pi_output) >= threshold

    def _bounded_rate(self, pi_output: float) -> float:
        """
        Bound pi_output to feedback limits, then convert to per-second rate.
        rate = 1 + pi_output / RAY_WAD_RATIO, clamped to [min_floor, max].
        """
        bounded = max(PID.feedback_lower_wad, min(PID.feedback_upper_wad, pi_output))
        rate = 1.0 + bounded / RAY_WAD_RATIO
        return max(PID.min_rate_per_second, min(PID.max_rate_per_second, rate))

    def compute_rate(
        self,
        market_price: float,
        redemption_price: float,
        current_time: float,
    ) -> float:
        """
        Main entry point — mirrors IPIDController.compute_rate in Cairo.
        Returns per-second rate factor (1.0 = no change).

        Note: the Cairo version enforces a cooldown. Here we don't — the
        caller (Simulator) controls timing via integral_period_s.
        """
        time_elapsed = (
            current_time - self.state.last_deviation_timestamp
            if self.state.last_deviation_timestamp > 0
            else 0.0
        )

        proportional = self._proportional_term(market_price, redemption_price)
        integral, _ = self._next_integral(proportional, time_elapsed)

        # Commit state (matches _update_deviation in Cairo)
        self.state.last_deviation_timestamp = current_time
        self.state.last_proportional = proportional
        self.state.last_integral = integral

        pi_output = self._pi_output(proportional, integral)

        if self._breaks_noise_barrier(pi_output, redemption_price):
            return self._bounded_rate(pi_output)
        return 1.0  # below noise barrier → no change


# ============================================================================
# Market Model
# ============================================================================


class MarketModel:
    """
    Simple dynamics:
        d(market_price)/dt = arb_speed * (redemption_price - market_price)
                             - btc_sensitivity * btc_return
                             + noise

    BTC return is the exogenous driver — comes from the scenario.
    Arbitrage pulls the market toward the redemption price with time constant
    1/arb_speed. BTC crashes drag the market down (confidence shock).
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def step(
        self,
        market_price: float,
        redemption_price: float,
        btc_price_prev: float,
        btc_price_now: float,
        dt: float,
    ) -> float:
        if btc_price_prev <= 0:
            btc_return = 0.0
        else:
            btc_return = (btc_price_now - btc_price_prev) / btc_price_prev

        arb_pull = MARKET.arb_speed * (redemption_price - market_price) * dt
        # BTC crash (negative return) drags market DOWN (same direction).
        btc_drag = MARKET.btc_sensitivity * btc_return * market_price
        noise = self.rng.normal(0.0, MARKET.market_noise_std) * market_price

        return max(1e-6, market_price + arb_pull + btc_drag + noise)


# ============================================================================
# Simulator — combines everything
# ============================================================================


@dataclass
class ActionResult:
    """Returned by Simulator.apply_action — enough for reward.py to compute."""
    accepted: bool                    # True if guard would accept
    bounds_violation: float           # 0 if valid, else magnitude of violation
    action_magnitude: float           # ||Δgains||
    requested_kp: float
    requested_ki: float
    applied_kp: float
    applied_ki: float


class Simulator:
    """
    Orchestrator:
      - Steps forward in time using BTC path (exogenous)
      - Market price evolves per MarketModel
      - Every integral_period_s, PID recomputes rate
      - Redemption price drifts per rate between PID calls
      - Agent actions apply gains subject to guard bounds
    """

    def __init__(
        self,
        btc_path: np.ndarray,                    # shape (T,) — BTC prices
        initial_kp: float = PID.kp_baseline,
        initial_ki: float = PID.ki_baseline,
        initial_market_price: float = 1.0,
        initial_redemption_price: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.btc_path = btc_path

        self.pid = PIDController(kp=initial_kp, ki=initial_ki)
        self.market = MarketModel(rng=rng)

        self.t: float = 0.0
        self.step_idx: int = 0
        self.market_price = initial_market_price
        self.redemption_price = initial_redemption_price
        self.redemption_rate = 1.0  # per-second
        self.last_pid_compute_t: float = 0.0

        self.guard_update_count: int = 0
        self.steps_since_last_update: int = 0
        self.guard_stopped: bool = False

        # Track gain history for monotonic-drift detection
        self.gain_history: list[tuple[float, float]] = [(initial_kp, initial_ki)]

    # ------------------------------------------------------------------ reset
    def reset(self) -> SystemState:
        """Return to t=0 with initial values. Re-seeds RNG is caller's choice."""
        self.t = 0.0
        self.step_idx = 0
        self.market_price = 1.0
        self.redemption_price = 1.0
        self.redemption_rate = 1.0
        self.last_pid_compute_t = 0.0
        self.guard_update_count = 0
        self.steps_since_last_update = 0
        self.guard_stopped = False

        baseline_kp, baseline_ki = self.gain_history[0]
        self.pid = PIDController(kp=baseline_kp, ki=baseline_ki)
        self.gain_history = [(baseline_kp, baseline_ki)]
        return self.get_state()

    # ------------------------------------------------------------------- step
    def step(self) -> SystemState:
        """Advance one dt. No agent action — just physics."""
        if self.step_idx >= len(self.btc_path) - 1:
            return self.get_state()

        dt = SIM.dt_s
        btc_prev = float(self.btc_path[self.step_idx])
        btc_now = float(self.btc_path[self.step_idx + 1])

        # 1. Market evolves
        self.market_price = self.market.step(
            market_price=self.market_price,
            redemption_price=self.redemption_price,
            btc_price_prev=btc_prev,
            btc_price_now=btc_now,
            dt=dt,
        )

        # 2. PID recomputes if integral period elapsed
        if self.t - self.last_pid_compute_t >= PID.integral_period_s:
            self.redemption_rate = self.pid.compute_rate(
                market_price=self.market_price,
                redemption_price=self.redemption_price,
                current_time=self.t,
            )
            self.last_pid_compute_t = self.t

        # 3. Redemption price drifts per rate
        #    rate is per-second → apply ^dt for the elapsed interval
        self.redemption_price *= self.redemption_rate ** dt

        # 4. Advance clock
        self.t += dt
        self.step_idx += 1
        self.steps_since_last_update += 1

        return self.get_state()

    # --------------------------------------------------------- apply_action
    def apply_action(
        self,
        requested_kp: float,
        requested_ki: float,
        is_emergency: bool = False,
    ) -> ActionResult:
        """
        Apply an agent proposal through the ParameterGuard logic.

        Returns ActionResult with bounds_violation=0 if accepted, else
        the violation magnitude, plus what was actually applied.
        """
        current_kp = self.pid.state.kp
        current_ki = self.pid.state.ki

        action_magnitude = abs(requested_kp - current_kp) + abs(
            requested_ki - current_ki
        )

        # Bounds checks
        bounds_violation = 0.0
        if requested_kp < GUARD.kp_min:
            bounds_violation += GUARD.kp_min - requested_kp
        if requested_kp > GUARD.kp_max:
            bounds_violation += requested_kp - GUARD.kp_max
        if requested_ki < GUARD.ki_min:
            bounds_violation += GUARD.ki_min - requested_ki
        if requested_ki > GUARD.ki_max:
            bounds_violation += requested_ki - GUARD.ki_max

        # Delta caps
        if abs(requested_kp - current_kp) > GUARD.max_kp_delta:
            bounds_violation += abs(requested_kp - current_kp) - GUARD.max_kp_delta
        if abs(requested_ki - current_ki) > GUARD.max_ki_delta:
            bounds_violation += abs(requested_ki - current_ki) - GUARD.max_ki_delta

        # Cooldown
        cooldown = (
            GUARD.cooldown_emergency_s if is_emergency else GUARD.cooldown_normal_s
        )
        cooldown_steps = max(1, int(cooldown / SIM.dt_s))
        cooldown_ok = self.steps_since_last_update >= cooldown_steps

        # Budget
        budget_ok = self.guard_update_count < GUARD.max_updates

        if (
            bounds_violation == 0.0
            and cooldown_ok
            and budget_ok
            and not self.guard_stopped
        ):
            # Accepted
            self.pid.set_gains(requested_kp, requested_ki)
            self.guard_update_count += 1
            self.steps_since_last_update = 0
            self.gain_history.append((requested_kp, requested_ki))
            return ActionResult(
                accepted=True,
                bounds_violation=0.0,
                action_magnitude=action_magnitude,
                requested_kp=requested_kp,
                requested_ki=requested_ki,
                applied_kp=requested_kp,
                applied_ki=requested_ki,
            )

        # Rejected — state unchanged
        return ActionResult(
            accepted=False,
            bounds_violation=bounds_violation,
            action_magnitude=action_magnitude,
            requested_kp=requested_kp,
            requested_ki=requested_ki,
            applied_kp=current_kp,
            applied_ki=current_ki,
        )

    # -------------------------------------------------------- run_forward
    def run_forward(self, n_steps: int) -> list[SystemState]:
        """Step forward n_steps without any agent action. Returns history."""
        history = []
        for _ in range(n_steps):
            state = self.step()
            history.append(state)
            if self.step_idx >= len(self.btc_path) - 1:
                break
        return history

    # -------------------------------------------------------------- getters
    def get_state(self) -> SystemState:
        btc = float(self.btc_path[min(self.step_idx, len(self.btc_path) - 1)])
        deviation_pct = (
            (self.redemption_price - self.market_price) / self.redemption_price * 100
            if self.redemption_price > 0
            else 0.0
        )
        collat_drop_pct = (
            (MARKET.btc_baseline_usd - btc) / MARKET.btc_baseline_usd * 100
            if MARKET.btc_baseline_usd > 0
            else 0.0
        )
        # Simple collat ratio: (btc_value / baseline_btc_value) * initial_ratio,
        # scaled by redemption_price drift (debt grows if redemption rises)
        collat_ratio = (
            MARKET.initial_collat_ratio
            * (btc / MARKET.btc_baseline_usd)
            / max(self.redemption_price, 1e-6)
        )

        return SystemState(
            t=self.t,
            market_price=self.market_price,
            redemption_price=self.redemption_price,
            btc_price=btc,
            kp=self.pid.state.kp,
            ki=self.pid.state.ki,
            last_proportional=self.pid.state.last_proportional,
            last_integral=self.pid.state.last_integral,
            redemption_rate=self.redemption_rate,
            collat_ratio=collat_ratio,
            guard_update_count=self.guard_update_count,
            steps_since_last_update=self.steps_since_last_update,
            deviation_pct=deviation_pct,
            collateral_drop_pct=collat_drop_pct,
        )

    def get_history_metrics(self, history: list[SystemState]) -> dict:
        """Summarize a history window — used by reward.py."""
        if not history:
            return {
                "mean_abs_deviation_pct": 0.0,
                "max_abs_deviation_pct": 0.0,
                "rate_std": 0.0,
                "min_collat_ratio": 0.0,
                "time_outside_band_pct": 0.0,
            }
        devs = np.array([h.deviation_pct for h in history])
        rates = np.array([h.redemption_rate for h in history])
        collats = np.array([h.collat_ratio for h in history])
        return {
            "mean_abs_deviation_pct": float(np.mean(np.abs(devs))),
            "max_abs_deviation_pct": float(np.max(np.abs(devs))),
            "rate_std": float(np.std(rates)),
            "min_collat_ratio": float(np.min(collats)),
            "time_outside_band_pct": float(np.mean(np.abs(devs) > 1.0) * 100),
        }


# ============================================================================
# Smoke test (run: python sim.py)
# ============================================================================


def _smoke_test() -> None:
    """Quick sanity check: random BTC path, no agent action, print state."""
    rng = np.random.default_rng(seed=42)
    # Synthetic BTC path: starts at 60k, random walk with drift
    n_steps = 200
    log_returns = rng.normal(loc=-0.0005, scale=0.015, size=n_steps)  # slight drift down
    btc_path = 60_000.0 * np.exp(np.cumsum(log_returns))

    sim = Simulator(btc_path=btc_path, rng=rng)
    history = sim.run_forward(n_steps=n_steps - 1)
    metrics = sim.get_history_metrics(history)

    print("=" * 60)
    print("SMOKE TEST — no agent action, baseline gains")
    print("=" * 60)
    print(f"BTC path: {btc_path[0]:.0f} → {btc_path[-1]:.0f}")
    print(f"Market price final:    ${sim.market_price:.4f}")
    print(f"Redemption price final: ${sim.redemption_price:.4f}")
    print(f"Rate final:            {sim.redemption_rate:.9f} per second")
    print(f"KP / KI:               {sim.pid.state.kp} / {sim.pid.state.ki}")
    print(f"Guard updates used:    {sim.guard_update_count}")
    print("Metrics over run:")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}")

    print()
    print("=" * 60)
    print("SMOKE TEST — with a random valid action mid-episode")
    print("=" * 60)
    sim2 = Simulator(btc_path=btc_path, rng=np.random.default_rng(seed=7))
    sim2.run_forward(n_steps=50)
    result = sim2.apply_action(requested_kp=3.5, requested_ki=0.005)
    print(f"Action accepted:    {result.accepted}")
    print(f"Bounds violation:   {result.bounds_violation}")
    print(f"Action magnitude:   {result.action_magnitude:.4f}")
    print(f"Applied (kp, ki):   ({result.applied_kp}, {result.applied_ki})")
    sim2.run_forward(n_steps=50)
    print(f"Post-action deviation_pct: {sim2.get_state().deviation_pct:.4f}%")

    print()
    print("=" * 60)
    print("SMOKE TEST — action that violates bounds")
    print("=" * 60)
    sim3 = Simulator(btc_path=btc_path)
    result = sim3.apply_action(requested_kp=15.0, requested_ki=0.5)  # both out of bounds
    print(f"Action accepted:    {result.accepted}  (expected False)")
    print(f"Bounds violation:   {result.bounds_violation:.4f}  (expected > 0)")

    print()
    print("OK — smoke tests done.")


if __name__ == "__main__":
    _smoke_test()
