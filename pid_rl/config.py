"""
PID-RL PoC — Configuration

Single source of truth for contract bounds, PID parameters, reward weights,
and training hyperparameters. Values verified against deployed contracts
at /mnt/c/Users/henry/desktop/pid/ (v10.1, on-chain as of 2026-04-19).

Units convention in sim.py:
  - prices in USD floats (e.g. 1.0 = $1.00)
  - kp, ki as human decimals (e.g. 2.0, 0.002)
  - proportional = dimensionless relative deviation
  - integral = dev-seconds (dimensionless * s)
  - pi_output_wad = mixed units as per Cairo: kp*prop + ki*integral
  - rate_per_second = 1 + pi_output_wad / RAY_WAD_RATIO (see sim.py)
"""

from dataclasses import dataclass


# ============================================================================
# Unit constants (match Cairo contract)
# ============================================================================

WAD = 1e18
RAY = 1e27
RAY_WAD_RATIO = RAY / WAD  # 1e9 — used to scale WAD pi_output to RAY rate delta


# ============================================================================
# ParameterGuard bounds (deployed_v10_1.json — on-chain verified 2026-04-19)
# ============================================================================

@dataclass(frozen=True)
class GuardBounds:
    kp_min: float = 0.1
    kp_max: float = 10.0
    ki_min: float = 0.0
    ki_max: float = 0.1
    max_kp_delta: float = 2.0
    max_ki_delta: float = 0.2
    cooldown_normal_s: int = 5
    cooldown_emergency_s: int = 3
    max_updates: int = 1000


GUARD = GuardBounds()


# ============================================================================
# PID deploy-time parameters (deploy_sepolia.sh)
# ============================================================================

@dataclass(frozen=True)
class PIDParams:
    # Initial gains on deployment (baseline for Static policy)
    kp_baseline: float = 2.0
    ki_baseline: float = 0.002

    # Noise barrier: 0.995 WAD → dead zone of ~0.5% around target
    noise_barrier: float = 0.995

    # Min seconds between PID computations (integral period)
    integral_period_s: int = 3600

    # Feedback bounds (RAY in contract); converted to WAD-scale here
    # FEEDBACK_UPPER = 1e27 = 1 RAY = 1e9 WAD → pi_output capped at ±1e9 WAD
    feedback_upper_wad: float = 1e9
    feedback_lower_wad: float = -1e9

    # Per-second leak applied to integral (RAY in contract)
    # 0.999999732582142 RAY/s → integral decays ~0.1% per hour
    per_second_leak: float = 0.999999732582142

    # Rate floor (MIN_RATE_FLOOR in contract): prevents rate from going negative
    # Translated to float: ~0.99999993 per second
    min_rate_per_second: float = 0.99999993

    # Practical max rate ceiling (very large, effectively unbounded)
    max_rate_per_second: float = 2.0


PID = PIDParams()


# ============================================================================
# Market model (our simulator — NOT in the contract)
# ============================================================================

@dataclass(frozen=True)
class MarketParams:
    # Arbitrage speed: how fast market price converges to redemption price
    # units: per second. 2e-5/s → ~7% convergence in 1 hour
    arb_speed: float = 2e-5

    # BTC sensitivity: confidence shock from BTC moves drags market same direction
    # 0.3 means a -10% BTC move pulls market down 3%
    btc_sensitivity: float = 0.3

    # Idiosyncratic noise (std per step); calibrated for 1-hour dt
    market_noise_std: float = 1e-3

    # Initial collateral ratio (collateral_value / debt_value)
    initial_collat_ratio: float = 2.0

    # Liquidation threshold for the collat ratio
    liquidation_threshold: float = 1.3

    # BTC baseline for collat computation (matches agent monitor.ts BTC_BASELINE)
    btc_baseline_usd: float = 60_000.0


MARKET = MarketParams()


# ============================================================================
# Simulation parameters
# ============================================================================

@dataclass(frozen=True)
class SimParams:
    # Seconds per simulation step = PID integral period.
    # Each sim step triggers one PID compute → gains changes take effect
    # on the next step (realistic and fast for RL).
    dt_s: float = 3600.0

    # Steps per episode (200 hours = ~8 days of sim time)
    episode_steps: int = 200

    # Steps evaluated AFTER each agent action (reward window)
    reward_window_steps: int = 20

    # Agent can propose every step; guard cooldown (5s) is trivially satisfied
    # since dt=3600s >> 5s.
    action_cooldown_steps: int = 1


SIM = SimParams()


# ============================================================================
# Reward weights
# ============================================================================

@dataclass(frozen=True)
class RewardWeights:
    # Term 1: penalize mean absolute deviation post-action (PRIMARY)
    # Bumped from 1.0 → 3.0 so that reducing dev actually outweighs the
    # action_magnitude cost of moving gains.
    alpha_deviation: float = 3.0

    # Term 2: penalize redemption-rate volatility (predictability)
    beta_rate_std: float = 0.5

    # Term 3: regularize action magnitude (avoid gratuitous changes)
    # Lowered 0.2 → 0.1 so legitimate tuning isn't eaten by the penalty.
    gamma_action_mag: float = 0.1

    # Term 4: heavy penalty if action violates guard bounds
    delta_bounds_violation: float = 10.0

    # Term 5: penalty for monotonic drift (the PROJECT's raison d'être — the
    # on-chain agent is stuck at kp=10.0 ceiling). Bumped 0.3 → 2.0 AND made
    # continuous/quadratic in _monotonic_drift_penalty.
    epsilon_monotonic: float = 2.0
    monotonic_window: int = 5  # length of streak that yields max penalty

    # Term 6: bonus for correctly-declared emergencies (precision)
    # Bumped 2.0 → 5.0 to meet DoD: emergency precision > 70%.
    bonus_justified_emergency: float = 5.0

    # Term 7: penalty for false-alarm emergencies (symmetric with bonus)
    penalty_false_alarm: float = 5.0

    # Output validation penalty (malformed JSON)
    malformed_output_penalty: float = -10.0


REWARD = RewardWeights()


# ============================================================================
# Emergency detection thresholds (for reward classification)
# ============================================================================

@dataclass(frozen=True)
class EmergencyThresholds:
    # Deviation magnitude that counts as "actual emergency" post-action
    deviation_pct_threshold: float = 5.0

    # BTC drop magnitude that counts as "actual emergency"
    btc_drop_pct_threshold: float = 10.0


EMERGENCY = EmergencyThresholds()


# ============================================================================
# Training hyperparameters (for train_grpo.ipynb)
# ============================================================================

@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_rank: int = 8
    batch_size: int = 4
    num_generations: int = 4  # GRPO group size
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    learning_rate: float = 5e-6
    max_steps: int = 500
    warmup_ratio: float = 0.1
    kl_coef: float = 0.04
    seed: int = 42


TRAIN = TrainConfig()


# ============================================================================
# Scenario distribution (for scenarios.py)
# ============================================================================

SCENARIO_WEIGHTS: dict = {
    "stable": 0.15,
    "crash": 0.30,
    "flash_crash": 0.15,
    "pump": 0.10,
    "volatile": 0.20,
    "recovery": 0.10,
}
