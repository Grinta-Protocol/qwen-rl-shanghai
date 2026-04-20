"""
PID-RL PoC — Scenario Generator

Synthetic BTC price paths for training + real historical crashes for validation.

Synthetic generators produce 200-step paths (1-hour granularity to match SIM.dt_s).
Each scenario type tests a different market regime the agent must handle.

Validation uses yfinance to grab real BTC crash windows. yfinance is optional
(install with `pip install yfinance`). If unavailable, we fall back to
synthetic crashes labeled "real_*".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config import MARKET, SIM, SCENARIO_WEIGHTS


# ============================================================================
# Scenario container
# ============================================================================


@dataclass
class Scenario:
    """A single training/eval sample."""
    btc_path: np.ndarray           # shape (T,) — BTC prices in USD
    initial_kp: float
    initial_ki: float
    scenario_type: str             # one of SCENARIO_WEIGHTS keys or "real_*"
    description: str               # human-readable for prompts / debugging

    def __post_init__(self) -> None:
        assert self.btc_path.ndim == 1, "btc_path must be 1D"
        assert len(self.btc_path) >= 2, "btc_path must have at least 2 points"
        assert np.all(self.btc_path > 0), "btc_path must be strictly positive"


# ============================================================================
# Synthetic generators — one per scenario_type
# ============================================================================


def _gbm_path(
    rng: np.random.Generator,
    n: int,
    start: float,
    drift_per_step: float,
    vol_per_step: float,
) -> np.ndarray:
    """Geometric Brownian motion path."""
    log_returns = rng.normal(loc=drift_per_step, scale=vol_per_step, size=n - 1)
    returns = np.concatenate([[0.0], log_returns])
    return start * np.exp(np.cumsum(returns))


def _inject_jump(
    path: np.ndarray, step: int, magnitude: float, recover_fraction: float = 0.0
) -> np.ndarray:
    """
    Apply a multiplicative jump at `step`, optionally recovering part of it
    over the next 10 steps. magnitude = -0.15 means -15% jump.
    """
    path = path.copy()
    if step >= len(path):
        return path
    path[step:] *= (1.0 + magnitude)

    if recover_fraction > 0 and step < len(path) - 10:
        recovery_target = -magnitude * recover_fraction
        ramp_end = min(len(path), step + 10)
        ramp_len = ramp_end - step
        ramp = np.linspace(0, recovery_target, ramp_len)
        path[step:ramp_end] *= (1.0 + ramp)

    return path


def _gen_stable(rng: np.random.Generator, n: int) -> Scenario:
    start = rng.uniform(50_000, 70_000)
    path = _gbm_path(rng, n, start, drift_per_step=0.0, vol_per_step=0.005)
    return Scenario(
        btc_path=path,
        initial_kp=2.0,
        initial_ki=0.002,
        scenario_type="stable",
        description=f"Stable regime starting at ${start:,.0f}, low volatility.",
    )


def _gen_crash(rng: np.random.Generator, n: int) -> Scenario:
    start = rng.uniform(50_000, 70_000)
    path = _gbm_path(rng, n, start, drift_per_step=-0.002, vol_per_step=0.015)
    # Inject 1-2 meaningful crashes
    n_crashes = rng.integers(1, 3)
    for _ in range(int(n_crashes)):
        crash_step = int(rng.integers(n // 4, 3 * n // 4))
        mag = rng.uniform(-0.20, -0.08)
        path = _inject_jump(path, crash_step, mag)
    return Scenario(
        btc_path=path,
        initial_kp=2.0,
        initial_ki=0.002,
        scenario_type="crash",
        description=f"Crash regime starting at ${start:,.0f} with sustained downward pressure.",
    )


def _gen_flash_crash(rng: np.random.Generator, n: int) -> Scenario:
    start = rng.uniform(55_000, 70_000)
    path = _gbm_path(rng, n, start, drift_per_step=0.0, vol_per_step=0.008)
    crash_step = int(rng.integers(n // 3, 2 * n // 3))
    mag = rng.uniform(-0.25, -0.15)
    path = _inject_jump(path, crash_step, mag, recover_fraction=0.8)
    return Scenario(
        btc_path=path,
        initial_kp=2.0,
        initial_ki=0.002,
        scenario_type="flash_crash",
        description=f"Flash crash at ${start:,.0f}: sharp {mag*100:.0f}% drop, 80% recovery.",
    )


def _gen_pump(rng: np.random.Generator, n: int) -> Scenario:
    start = rng.uniform(40_000, 60_000)
    path = _gbm_path(rng, n, start, drift_per_step=0.003, vol_per_step=0.012)
    # Occasional upward jumps
    for _ in range(int(rng.integers(1, 3))):
        step = int(rng.integers(n // 4, 3 * n // 4))
        mag = rng.uniform(0.05, 0.15)
        path = _inject_jump(path, step, mag)
    return Scenario(
        btc_path=path,
        initial_kp=2.0,
        initial_ki=0.002,
        scenario_type="pump",
        description=f"Pump regime starting at ${start:,.0f} with upward jumps.",
    )


def _gen_volatile(rng: np.random.Generator, n: int) -> Scenario:
    start = rng.uniform(45_000, 70_000)
    path = _gbm_path(rng, n, start, drift_per_step=0.0, vol_per_step=0.03)
    return Scenario(
        btc_path=path,
        initial_kp=2.0,
        initial_ki=0.002,
        scenario_type="volatile",
        description=f"High-volatility choppy regime at ${start:,.0f}, no clear drift.",
    )


def _gen_recovery(rng: np.random.Generator, n: int) -> Scenario:
    # Start from a post-crash low and recover
    crash_start = rng.uniform(60_000, 70_000)
    low = crash_start * rng.uniform(0.5, 0.7)
    path = _gbm_path(rng, n, low, drift_per_step=0.004, vol_per_step=0.02)
    return Scenario(
        btc_path=path,
        initial_kp=2.0,
        initial_ki=0.002,
        scenario_type="recovery",
        description=f"Recovery regime from post-crash low ${low:,.0f}.",
    )


_GENERATORS: dict[str, Callable[[np.random.Generator, int], Scenario]] = {
    "stable": _gen_stable,
    "crash": _gen_crash,
    "flash_crash": _gen_flash_crash,
    "pump": _gen_pump,
    "volatile": _gen_volatile,
    "recovery": _gen_recovery,
}


# ============================================================================
# Sampling API — used by the training loop
# ============================================================================


def sample_scenario(
    rng: Optional[np.random.Generator] = None,
    n_steps: Optional[int] = None,
) -> Scenario:
    """
    Sample one scenario according to SCENARIO_WEIGHTS distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_steps is None:
        n_steps = SIM.episode_steps
    types = list(SCENARIO_WEIGHTS.keys())
    weights = np.array(list(SCENARIO_WEIGHTS.values()))
    weights = weights / weights.sum()
    chosen = rng.choice(types, p=weights)
    return _GENERATORS[chosen](rng, n_steps)


def generate_batch(
    n: int, seed: int = 42, n_steps: Optional[int] = None
) -> list[Scenario]:
    """Generate a reproducible batch of n scenarios."""
    rng = np.random.default_rng(seed=seed)
    return [sample_scenario(rng, n_steps) for _ in range(n)]


# ============================================================================
# Real BTC historical data (validation)
# ============================================================================


# Known crash windows for validation — (start_date, end_date, label)
# Windows extended to 30+ days. yfinance hourly only goes 730 days back, so
# older crashes (pre-2024-04) will fall to DAILY bars (~30 points) and younger
# crashes keep hourly (~720 points capped to episode_steps=200). This is an
# acknowledged limitation — the older-crash daily resolution still reveals
# policy behavior over the crash+recovery arc, just with fewer decision points.
REAL_CRASH_WINDOWS: list[tuple[str, str, str]] = [
    ("2020-03-01", "2020-04-05", "COVID crash March 2020"),
    ("2021-05-10", "2021-06-15", "China ban May 2021"),
    ("2022-05-05", "2022-06-10", "LUNA collapse May 2022"),
    ("2022-11-05", "2022-12-10", "FTX collapse Nov 2022"),
    ("2024-08-01", "2024-09-05", "Yen carry trade Aug 2024"),
]


def load_real_crashes(use_hourly: bool = True) -> list[Scenario]:
    """
    Download BTC crash windows via yfinance. Returns scenarios.

    Hourly data in yfinance is limited to ~730 days back, so older crashes
    fall back to daily bars (still usable but coarser).

    If yfinance is unavailable or download fails, returns empty list and
    prints a warning — training won't block on this.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[scenarios] yfinance not installed — skipping real crash loading.")
        print("             install with: pip install yfinance")
        return []

    scenarios: list[Scenario] = []
    for start, end, label in REAL_CRASH_WINDOWS:
        try:
            interval = "1h" if use_hourly else "1d"
            df = yf.download(
                "BTC-USD", start=start, end=end, interval=interval, progress=False
            )
            if df is None or len(df) < 10:
                # Hourly might not be available for old dates; retry daily
                df = yf.download(
                    "BTC-USD", start=start, end=end, interval="1d", progress=False
                )
            if df is None or len(df) < 10:
                print(f"[scenarios] Skipping {label}: insufficient data.")
                continue
            close = df["Close"].to_numpy().flatten().astype(float)
            close = close[np.isfinite(close) & (close > 0)]
            if len(close) < 10:
                continue
            scenarios.append(
                Scenario(
                    btc_path=close,
                    initial_kp=2.0,
                    initial_ki=0.002,
                    scenario_type=f"real_{label.lower().replace(' ', '_')}",
                    description=f"Real BTC window: {label} ({start} to {end})",
                )
            )
        except Exception as e:
            print(f"[scenarios] Failed to download {label}: {e}")
            continue

    print(f"[scenarios] Loaded {len(scenarios)} real crash scenarios.")
    return scenarios


# ============================================================================
# Smoke test
# ============================================================================


def _smoke_test() -> None:
    rng = np.random.default_rng(seed=42)

    print("=" * 60)
    print("SMOKE TEST — one scenario of each type")
    print("=" * 60)
    for name, gen in _GENERATORS.items():
        scenario = gen(rng, SIM.episode_steps)
        path = scenario.btc_path
        min_price = float(path.min())
        max_price = float(path.max())
        total_return = (float(path[-1]) - float(path[0])) / float(path[0]) * 100
        print(
            f"  {name:14s}  "
            f"start=${path[0]:>8,.0f}  "
            f"end=${path[-1]:>8,.0f}  "
            f"range=[${min_price:>8,.0f}, ${max_price:>8,.0f}]  "
            f"total={total_return:+6.1f}%"
        )

    print()
    print("=" * 60)
    print("SMOKE TEST — batch sampling respects distribution")
    print("=" * 60)
    batch = generate_batch(n=1000, seed=123)
    from collections import Counter

    counts = Counter(s.scenario_type for s in batch)
    for t, expected_w in SCENARIO_WEIGHTS.items():
        observed = counts.get(t, 0) / len(batch)
        print(f"  {t:14s}  expected={expected_w:.2f}  observed={observed:.3f}")

    print()
    print("=" * 60)
    print("SMOKE TEST — feed scenario into Simulator")
    print("=" * 60)
    from sim import Simulator

    scenario = _gen_flash_crash(rng, SIM.episode_steps)
    sim = Simulator(
        btc_path=scenario.btc_path,
        initial_kp=scenario.initial_kp,
        initial_ki=scenario.initial_ki,
        rng=np.random.default_rng(seed=7),
    )
    history = sim.run_forward(n_steps=SIM.episode_steps - 1)
    metrics = sim.get_history_metrics(history)
    print(f"Scenario: {scenario.description}")
    print(f"BTC path: {scenario.btc_path[0]:.0f} → {scenario.btc_path[-1]:.0f}")
    print(f"Final market/redemption: ${sim.market_price:.4f} / ${sim.redemption_price:.4f}")
    for k, v in metrics.items():
        print(f"  {k:30s} {v:.4f}")

    print()
    print("OK — scenarios smoke tests done.")


if __name__ == "__main__":
    _smoke_test()
