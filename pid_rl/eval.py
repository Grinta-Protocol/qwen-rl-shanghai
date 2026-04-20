"""
PID-RL PoC — Evaluation harness.

Compares any number of policies on two benchmarks:

  1. SYNTHETIC HOLDOUT — fresh scenarios from scenarios.generate_batch with a
     held-out seed (not used during training). Tests the distribution the
     model was trained on.

  2. REAL CRASHES — BTC historical windows via yfinance (COVID, LUNA, FTX,
     Yen carry, China ban). Tests generalization. This is the ULTIMATE
     test for the DoD.

Output:
  - Per-policy aggregate table (mean reward, dev, accept rate, monotonic %)
  - Per-scenario-type breakdown (crash vs stable vs pump etc.)
  - Per-real-crash breakdown

Designed to be called from the training notebook (after the trained model
is wrapped into a `TrainedModelPolicy`) or standalone (baselines only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scenarios import Scenario, generate_batch, load_real_crashes
from baselines import (
    Policy,
    RandomPolicy,
    StaticGainsPolicy,
    HeuristicPolicy,
    EvalResult,
    evaluate_policy,
)


# ============================================================================
# Aggregation helpers
# ============================================================================


@dataclass
class ComparisonRow:
    """One row of the side-by-side comparison table."""
    policy: str
    bench: str
    n: int
    mean_reward: float
    mean_abs_dev: float
    max_abs_dev_worst: float     # worst-case (max of max_abs_dev_pct across scenarios)
    fraction_monotonic: float
    accept_rate: float
    parse_error_rate: float
    mean_emergency_calls: float  # avg number of adjust_emergency per scenario


def _build_row(policy_name: str, bench: str, result: EvalResult) -> ComparisonRow:
    scenarios_n = len(result.per_scenario)
    if scenarios_n == 0:
        return ComparisonRow(
            policy=policy_name, bench=bench, n=0, mean_reward=0, mean_abs_dev=0,
            max_abs_dev_worst=0, fraction_monotonic=0, accept_rate=0,
            parse_error_rate=0, mean_emergency_calls=0,
        )
    total_errors = sum(s.n_parse_errors for s in result.per_scenario)
    total_decisions = sum(
        s.n_actions_accepted + s.n_actions_rejected + s.n_parse_errors
        for s in result.per_scenario
    )
    return ComparisonRow(
        policy=policy_name,
        bench=bench,
        n=scenarios_n,
        mean_reward=result.mean_reward,
        mean_abs_dev=result.mean_abs_deviation,
        max_abs_dev_worst=max((s.max_abs_deviation_pct for s in result.per_scenario), default=0.0),
        fraction_monotonic=result.fraction_monotonic,
        accept_rate=result.accept_rate,
        parse_error_rate=(total_errors / total_decisions) if total_decisions > 0 else 0.0,
        mean_emergency_calls=0.0,  # filled from ScenarioResult if we add counter; stub for now
    )


def _print_comparison_table(rows: list[ComparisonRow]) -> None:
    header = (
        f"{'policy':12s}  {'bench':16s}  {'n':>3s}  {'reward':>9s}  "
        f"{'dev%':>6s}  {'worst%':>7s}  {'mono%':>6s}  {'acc%':>6s}  {'err%':>5s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.policy:12s}  {r.bench:16s}  {r.n:>3d}  "
            f"{r.mean_reward:>+9.2f}  {r.mean_abs_dev:>6.3f}  "
            f"{r.max_abs_dev_worst:>7.3f}  "
            f"{r.fraction_monotonic * 100:>5.1f}  "
            f"{r.accept_rate * 100:>5.1f}  "
            f"{r.parse_error_rate * 100:>4.1f}"
        )


def _print_per_type_breakdown(results_by_policy: dict[str, EvalResult]) -> None:
    """Group scenario results by scenario_type, compare policies per type."""
    # Collect types
    types = set()
    for result in results_by_policy.values():
        for s in result.per_scenario:
            # Collapse real_* into a single "real" bucket for this view
            t = "real" if s.scenario_type.startswith("real_") else s.scenario_type
            types.add(t)
    if not types:
        return

    header = f"\n{'type':14s}  " + "  ".join(f"{name:>10s}" for name in results_by_policy.keys())
    print(header)
    print("-" * len(header))

    for t in sorted(types):
        cells = []
        for _pname, result in results_by_policy.items():
            matching = [
                s for s in result.per_scenario
                if (s.scenario_type == t) or (t == "real" and s.scenario_type.startswith("real_"))
            ]
            if matching:
                mean = float(np.mean([s.total_reward for s in matching]))
                cells.append(f"{mean:>+10.2f}")
            else:
                cells.append(f"{'--':>10s}")
        print(f"{t:14s}  " + "  ".join(cells))


def _print_real_crashes(results_by_policy: dict[str, EvalResult]) -> None:
    """One line per real crash, showing reward per policy."""
    # Collect real crash labels in order
    labels = []
    for result in results_by_policy.values():
        for s in result.per_scenario:
            if s.scenario_type.startswith("real_"):
                lbl = s.scenario_type.replace("real_", "")
                if lbl not in labels:
                    labels.append(lbl)
    if not labels:
        print("\n(no real crash scenarios — yfinance unavailable?)")
        return

    header = f"\n{'crash':40s}  " + "  ".join(f"{name:>10s}" for name in results_by_policy.keys())
    print(header)
    print("-" * len(header))
    for lbl in labels:
        cells = []
        for result in results_by_policy.values():
            matched = next(
                (s for s in result.per_scenario if s.scenario_type == f"real_{lbl}"), None
            )
            cells.append(f"{matched.total_reward:>+10.2f}" if matched else f"{'--':>10s}")
        # Cap label length for the table
        shown = lbl[:38] + ".." if len(lbl) > 40 else lbl
        print(f"{shown:40s}  " + "  ".join(cells))


# ============================================================================
# Main evaluation driver
# ============================================================================


def _target_cadence(scenario: Scenario, target_decisions: int = 20) -> int:
    """
    Pick action_every_n_steps so each scenario sees ~`target_decisions` decisions.
    Short daily-bar real crashes would otherwise collapse to 1-2 decisions under
    the synthetic default of 10.
    """
    episode_len = min(len(scenario.btc_path), 200) - 1
    return max(1, episode_len // target_decisions)


def run_full_eval(
    policies: list[Policy],
    n_synthetic_holdout: int = 24,
    holdout_seed: int = 9999,                 # different from training seed
    action_every_n_steps: int = 10,
    include_real: bool = True,
    verbose: bool = False,
) -> dict[str, dict[str, EvalResult]]:
    """
    Returns: {policy_name: {"synthetic": EvalResult, "real": EvalResult}}

    Note: synthetic uses a fixed cadence (default 10 → 20 decisions per 200-step
    episode). Real crashes use ADAPTIVE cadence per scenario to hit ~20 decisions
    each — otherwise short daily-bar crashes collapse to 1-2 decisions and all
    policies tie trivially.
    """
    # Load benchmarks
    print(f"[eval] Generating {n_synthetic_holdout} synthetic holdout scenarios "
          f"(seed={holdout_seed}) ...")
    synth = generate_batch(n=n_synthetic_holdout, seed=holdout_seed)

    real: list[Scenario] = []
    if include_real:
        print("[eval] Loading real crash windows via yfinance ...")
        real = load_real_crashes(use_hourly=True)

    all_results: dict[str, dict[str, EvalResult]] = {}
    rows: list[ComparisonRow] = []

    for pol in policies:
        print(f"\n[eval] ===== policy: {pol.name} =====")
        print(f"  [eval] synthetic holdout ({len(synth)}) ...")
        res_synth = evaluate_policy(
            pol, synth, action_every_n_steps=action_every_n_steps, verbose=verbose
        )
        rows.append(_build_row(pol.name, "synthetic", res_synth))

        if real:
            # Per-scenario adaptive cadence for real crashes
            real_results = []
            for sc in real:
                cadence = _target_cadence(sc)
                sub = evaluate_policy(
                    pol, [sc], action_every_n_steps=cadence, verbose=verbose
                )
                real_results.extend(sub.per_scenario)
            # Build combined EvalResult manually
            res_real = EvalResult(policy_name=pol.name, per_scenario=real_results)
            if real_results:
                res_real.mean_reward = float(np.mean([r.total_reward for r in real_results]))
                res_real.mean_abs_deviation = float(
                    np.mean([r.mean_abs_deviation_pct for r in real_results])
                )
                res_real.fraction_monotonic = float(
                    np.mean([r.monotonic_kp for r in real_results])
                )
                total_acts = sum(r.n_actions_accepted + r.n_actions_rejected for r in real_results)
                total_acc = sum(r.n_actions_accepted for r in real_results)
                res_real.accept_rate = (total_acc / total_acts) if total_acts > 0 else 0.0
            print(f"  [eval] real crashes ({len(real)}) done.")
            rows.append(_build_row(pol.name, "real", res_real))
        else:
            res_real = EvalResult(policy_name=pol.name, per_scenario=[])

        all_results[pol.name] = {"synthetic": res_synth, "real": res_real}

    # Aggregate report
    print()
    print("=" * 78)
    print("FULL EVAL — aggregate comparison")
    print("=" * 78)
    _print_comparison_table(rows)

    print()
    print("=" * 78)
    print("PER-SCENARIO-TYPE REWARD (synthetic holdout)")
    print("=" * 78)
    _print_per_type_breakdown({p: r["synthetic"] for p, r in all_results.items()})

    if any(r["real"].per_scenario for r in all_results.values()):
        print()
        print("=" * 78)
        print("PER-REAL-CRASH REWARD")
        print("=" * 78)
        _print_real_crashes({p: r["real"] for p, r in all_results.items()})

    # DoD checklist
    print()
    print("=" * 78)
    print("DEFINITION OF DONE — checklist vs PLAN.md")
    print("=" * 78)
    _print_dod_checklist(all_results)

    return all_results


def _print_dod_checklist(all_results: dict[str, dict[str, EvalResult]]) -> None:
    """
    Checks DoD items from PLAN.md. This is a STRICT eval — meant for the
    trained model. Baselines will naturally fail some items (e.g. static
    never hits emergency precision since it never declares emergency).
    """
    def fmt_check(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    baseline_static_reward = None
    baseline_heuristic_reward = None
    if "static" in all_results:
        baseline_static_reward = all_results["static"]["synthetic"].mean_reward
    if "heuristic" in all_results:
        baseline_heuristic_reward = all_results["heuristic"]["synthetic"].mean_reward

    for policy_name, results in all_results.items():
        if policy_name in ("static", "heuristic", "random"):
            continue  # checklist applies to the trained model, not baselines

        synth = results["synthetic"]
        real = results["real"]

        beats_static = (
            baseline_static_reward is None
            or synth.mean_reward > baseline_static_reward
        )
        beats_heuristic = (
            baseline_heuristic_reward is None
            or synth.mean_reward >= baseline_heuristic_reward
        )
        no_monotonic = synth.fraction_monotonic < 0.10

        total_decisions = sum(
            s.n_actions_accepted + s.n_actions_rejected + s.n_parse_errors
            for s in synth.per_scenario
        )
        parse_success_rate = (
            (total_decisions - sum(s.n_parse_errors for s in synth.per_scenario))
            / total_decisions
            if total_decisions > 0 else 0.0
        )
        parse_ok = parse_success_rate > 0.98

        print(f"\n  policy: {policy_name}")
        print(f"    [{fmt_check(parse_ok)}] valid JSON rate > 98%: {parse_success_rate:.1%}")
        print(f"    [{fmt_check(beats_static)}] beats static ({synth.mean_reward:+.2f} vs "
              f"{baseline_static_reward:+.2f})" if baseline_static_reward else
              f"    [--] beats static (no static baseline run)")
        print(f"    [{fmt_check(beats_heuristic)}] meets heuristic "
              f"({synth.mean_reward:+.2f} vs {baseline_heuristic_reward:+.2f})"
              if baseline_heuristic_reward else
              f"    [--] meets heuristic (no heuristic baseline run)")
        print(f"    [{fmt_check(no_monotonic)}] no monotonic drift "
              f"(fraction={synth.fraction_monotonic:.1%})")

        if real.per_scenario:
            real_beats_static = (
                "static" not in all_results
                or real.mean_reward > all_results["static"]["real"].mean_reward
            )
            print(f"    [{fmt_check(real_beats_static)}] real crashes: beats static "
                  f"({real.mean_reward:+.2f})")


# ============================================================================
# Standalone: baselines-only eval
# ============================================================================


def main(n_synthetic: int = 24, include_real: bool = True, verbose: bool = False) -> None:
    """Run the three baselines on both benchmarks. No trained model here."""
    rng = np.random.default_rng(seed=123)
    policies: list[Policy] = [
        StaticGainsPolicy(),
        HeuristicPolicy(),
        RandomPolicy(rng=rng),
    ]
    run_full_eval(
        policies=policies,
        n_synthetic_holdout=n_synthetic,
        include_real=include_real,
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24, help="synthetic holdout count")
    parser.add_argument("--no-real", action="store_true", help="skip yfinance real crashes")
    parser.add_argument("--verbose", action="store_true", help="per-scenario output")
    args = parser.parse_args()
    main(n_synthetic=args.n, include_real=not args.no_real, verbose=args.verbose)
