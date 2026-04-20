"""
PID-RL PoC — Prompt builder + output parser.

build_prompt() feeds the LLM everything it needs to decide: current state,
recent BTC/deviation history, and the guardrail bounds. Bounds come from
config.py (single source of truth) — NOT from the old TypeScript reasoning.ts
which still has outdated [1.4, 2.6] bounds.

parse_output() accepts the LLM completion string and returns either a valid
action dict or a `ParseError` telling reward.py to apply malformed_output_penalty.

Output schema:
    {
      "action": "hold" | "adjust" | "adjust_emergency",
      "new_kp": float,
      "new_ki": float,
      "is_emergency": bool,
      "reasoning": string
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import GUARD, PID
from sim import SystemState


# ============================================================================
# Parsed action container
# ============================================================================


@dataclass
class ParsedAction:
    action: str              # "hold" | "adjust" | "adjust_emergency"
    new_kp: float
    new_ki: float
    is_emergency: bool
    reasoning: str
    raw: str                 # original completion (for debugging / logging)


@dataclass
class ParseError:
    reason: str              # short diagnostic
    raw: str


# ============================================================================
# Prompt construction
# ============================================================================


SYSTEM_PROMPT = f"""You are an autonomous governance agent for a BTC-collateralized stablecoin (GRIT). You tune the on-chain PID controller that sets the redemption rate.

MECHANICS (must understand):
- redemption_rate > 1 → redemption_price rises over time → debt grows → incentive to buy GRIT → price up
- redemption_rate < 1 → redemption_price falls → debt shrinks → incentive to sell GRIT → price down
- The PID computes rate from (redemption_price - market_price)/redemption_price using gains kp (proportional) and ki (integral).
- Higher kp = faster reaction to deviation. Higher ki = stronger memory of past deviation.

YOUR JOB: propose (new_kp, new_ki) to keep the system HEALTHY — prevent liquidations during BTC crashes, avoid overshoot during pumps. You are NOT trying to "correct the peg" directly; you adjust how aggressively the PID corrects the rate.

HARD BOUNDS (enforced on-chain by ParameterGuard — violating them wastes an action):
- kp ∈ [{GUARD.kp_min}, {GUARD.kp_max}]
- ki ∈ [{GUARD.ki_min}, {GUARD.ki_max}]
- |new_kp - current_kp| ≤ {GUARD.max_kp_delta}
- |new_ki - current_ki| ≤ {GUARD.max_ki_delta}
- cooldown: {GUARD.cooldown_normal_s}s normal / {GUARD.cooldown_emergency_s}s emergency

DEPLOY BASELINE: kp={PID.kp_baseline}, ki={PID.ki_baseline}. Return to baseline when conditions normalize.

DECISION FRAMEWORK:
- HOLD: deviation < 1% AND BTC stable → no change
- ADJUST: deviation 1-5% OR moderate BTC move (±5-10%) → tune gains, is_emergency=false
- ADJUST_EMERGENCY: deviation > 5% OR BTC drop > 10% → tune aggressively, is_emergency=true

CRITICAL ANTI-PATTERNS (the previous agent got stuck doing these — avoid):
- Monotonic drift: don't push kp up every round. If BTC PUMPS, DECREASE kp. Symmetry matters.
- Gratuitous changes: if state is fine, HOLD — don't burn guard budget.
- Ignoring ki: ki accumulates error. If integral is already large, don't raise ki further.

OUTPUT — respond with ONE valid JSON object, nothing else. No markdown fences, no prose before/after:
{{"action":"hold|adjust|adjust_emergency","new_kp":<float>,"new_ki":<float>,"is_emergency":<bool>,"reasoning":"<one short sentence>"}}"""


def _fmt_history(arr: np.ndarray, n: int = 8, decimals: int = 4) -> str:
    """Format the last n entries as a compact list."""
    tail = arr[-n:] if len(arr) >= n else arr
    return "[" + ", ".join(f"{float(x):.{decimals}f}" for x in tail) + "]"


def build_prompt(
    state: SystemState,
    btc_history: np.ndarray,
    deviation_history: Optional[list[float]] = None,
    window_high_lookback: int = 24,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the LLM.

    btc_history: recent BTC prices (1-hour bars), most recent last.
    deviation_history: recent deviation_pct values (hourly), most recent last.
    """
    btc_history = np.asarray(btc_history, dtype=float)

    # Rolling features
    if len(btc_history) >= 2:
        btc_returns = np.diff(btc_history) / btc_history[:-1]
    else:
        btc_returns = np.array([0.0])
    btc_vol = float(np.std(btc_returns[-24:])) if len(btc_returns) >= 2 else 0.0
    lookback = min(window_high_lookback, len(btc_history))
    window_high = float(np.max(btc_history[-lookback:])) if lookback > 0 else state.btc_price
    drawdown_pct = (
        (state.btc_price - window_high) / window_high * 100.0
        if window_high > 0 else 0.0
    )

    dev_hist_str = (
        _fmt_history(np.asarray(deviation_history), n=8, decimals=3)
        if deviation_history else "[]"
    )

    user_prompt = f"""STATE:
- market_price_usd: {state.market_price:.6f}
- redemption_price_usd: {state.redemption_price:.6f}
- deviation_pct: {state.deviation_pct:+.3f}%
- btc_price_usd: {state.btc_price:,.2f}
- collateral_drop_pct: {state.collateral_drop_pct:+.2f}%  (vs baseline {60000:,})
- collat_ratio: {state.collat_ratio:.3f}  (liquidation < 1.30)
- current_kp: {state.kp:.4f}
- current_ki: {state.ki:.6f}
- last_proportional: {state.last_proportional:+.6f}
- last_integral: {state.last_integral:+.6f}
- redemption_rate_per_s: {state.redemption_rate:.9f}
- guard_updates_used: {state.guard_update_count} / {GUARD.max_updates}
- steps_since_last_update: {state.steps_since_last_update}

HISTORY (last 24h):
- btc_returns: {_fmt_history(btc_returns, n=8, decimals=4)}
- btc_volatility_24h: {btc_vol:.4f}
- btc_drawdown_from_window_high_pct: {drawdown_pct:+.2f}%
- deviation_history: {dev_hist_str}

Decide. Respond with ONE JSON object exactly matching the schema."""

    return SYSTEM_PROMPT, user_prompt


# ============================================================================
# Output parsing
# ============================================================================


_ALLOWED_ACTIONS = {"hold", "adjust", "adjust_emergency"}


def _extract_json(text: str) -> Optional[str]:
    """Find the first balanced JSON object in text. Returns None if none found."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)

    # Balanced-brace scan (the regex can't handle nested braces reliably)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_output(completion: str) -> ParsedAction | ParseError:
    """
    Parse an LLM completion into a ParsedAction or ParseError.

    We're LENIENT about surrounding prose/fences (common with instruct models)
    but STRICT about the schema itself — GRPO needs a clean signal.
    """
    if not completion or not isinstance(completion, str):
        return ParseError(reason="empty_or_non_string", raw=str(completion))

    js = _extract_json(completion)
    if js is None:
        return ParseError(reason="no_json_object_found", raw=completion)

    try:
        data = json.loads(js)
    except json.JSONDecodeError as e:
        return ParseError(reason=f"json_decode_error:{e.msg}", raw=completion)

    if not isinstance(data, dict):
        return ParseError(reason="json_not_object", raw=completion)

    # Required fields
    for key in ("action", "new_kp", "new_ki", "is_emergency"):
        if key not in data:
            return ParseError(reason=f"missing_field:{key}", raw=completion)

    action = data["action"]
    if not isinstance(action, str) or action.lower() not in _ALLOWED_ACTIONS:
        return ParseError(reason=f"invalid_action:{action!r}", raw=completion)
    action = action.lower()

    try:
        new_kp = float(data["new_kp"])
        new_ki = float(data["new_ki"])
    except (TypeError, ValueError):
        return ParseError(
            reason=f"non_numeric_gains:kp={data['new_kp']!r},ki={data['new_ki']!r}",
            raw=completion,
        )

    if not (np.isfinite(new_kp) and np.isfinite(new_ki)):
        return ParseError(reason="non_finite_gains", raw=completion)

    is_emergency = data["is_emergency"]
    if not isinstance(is_emergency, bool):
        # Accept common string variants
        if isinstance(is_emergency, str):
            is_emergency = is_emergency.strip().lower() in ("true", "1", "yes")
        else:
            return ParseError(
                reason=f"invalid_is_emergency:{is_emergency!r}", raw=completion
            )

    reasoning = str(data.get("reasoning", "")).strip()

    return ParsedAction(
        action=action,
        new_kp=new_kp,
        new_ki=new_ki,
        is_emergency=is_emergency,
        reasoning=reasoning,
        raw=completion,
    )


# ============================================================================
# Smoke test
# ============================================================================


def _smoke_test() -> None:
    from sim import Simulator
    from scenarios import _gen_flash_crash
    from config import SIM

    rng = np.random.default_rng(seed=42)
    scenario = _gen_flash_crash(rng, SIM.episode_steps)
    sim = Simulator(
        btc_path=scenario.btc_path,
        initial_kp=scenario.initial_kp,
        initial_ki=scenario.initial_ki,
        rng=np.random.default_rng(seed=7),
    )
    sim.run_forward(n_steps=30)
    state = sim.get_state()

    sys_p, user_p = build_prompt(
        state=state,
        btc_history=scenario.btc_path[: sim.step_idx + 1],
        deviation_history=[0.1, 0.3, 0.5, -0.2, 1.2, 2.5, 3.1, 4.0],
    )

    print("=" * 60)
    print("SMOKE TEST — build_prompt")
    print("=" * 60)
    print("[SYSTEM]")
    print(sys_p)
    print()
    print("[USER]")
    print(user_p)

    print()
    print("=" * 60)
    print("SMOKE TEST — parse_output happy path")
    print("=" * 60)
    good = '{"action":"adjust","new_kp":3.5,"new_ki":0.003,"is_emergency":false,"reasoning":"BTC down 8%, increase kp"}'
    result = parse_output(good)
    print(result)

    print()
    print("=" * 60)
    print("SMOKE TEST — parse_output with prose + fences (common case)")
    print("=" * 60)
    messy = """Here's my decision:
```json
{"action":"adjust_emergency","new_kp":4.0,"new_ki":0.005,"is_emergency":true,"reasoning":"flash crash detected"}
```
Hope that helps!"""
    print(parse_output(messy))

    print()
    print("=" * 60)
    print("SMOKE TEST — parse_output error cases")
    print("=" * 60)
    for bad in [
        "",
        "no json here",
        '{"action":"panic","new_kp":2,"new_ki":0.002,"is_emergency":false}',
        '{"action":"hold","new_kp":"not_a_number","new_ki":0.002,"is_emergency":false}',
        '{"action":"hold","new_kp":2}',  # missing fields
        '{"action":"hold","new_kp":NaN,"new_ki":0.002,"is_emergency":false}',
    ]:
        r = parse_output(bad)
        label = type(r).__name__
        detail = r.reason if isinstance(r, ParseError) else "(ok)"
        print(f"  {label:12s} {detail}")

    print()
    print("OK — prompt smoke tests done.")


if __name__ == "__main__":
    _smoke_test()
