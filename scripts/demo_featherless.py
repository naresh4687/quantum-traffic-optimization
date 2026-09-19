"""Demo: explain REAL saved project results with Featherless, or with the deterministic fallback.

Run:  python scripts/demo_featherless.py [--results-dir results] [--scenario ns_heavy]

1. Loads real saved results (Phase 3-6 files) and builds structured analysis contexts.
2. Tries Featherless if FEATHERLESS_API_KEY and FEATHERLESS_MODEL are set (one request per explanation).
3. Otherwise, or on any failure, uses the deterministic fallback built from the same numbers.
4. Prints each explanation and whether it came from "featherless" or "deterministic_fallback".

Featherless only EXPLAINS results that the deterministic simulator and optimizers already produced. It
never chooses signals, changes QUBO/QAOA parameters or touches the emergency controller. No demonstration
data is made up: every number comes from results/. The API key is never printed.
"""

from __future__ import annotations

import argparse
import sys

from qtraffic.ai import (
    Explainer, FeatherlessConfig, explain_emergency, explain_environment, explain_optimization, explain_qaoa,
    explain_signal_choice, load_controller_context, load_emergency_context, load_qaoa_context,
)


def show(title: str, result) -> None:
    print("=" * 78)
    print(f"{title}")
    reason = f" (reason: {result.fallback_reason})" if result.fallback_reason else ""
    print(f"source: {result.source}{reason}   featherless_available: {result.available}"
          + (f"   model: {result.model}" if result.model else ""))
    print("-" * 78)
    print(result.explanation)
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--scenario", default="ns_heavy", help="scenario for the optimization / QAOA / signal explanations")
    ap.add_argument("--controller", default="qubo_static")
    ap.add_argument("--emergency-scenario", default="balanced_medium")
    ap.add_argument("--environment-scenario", default="ew_heavy")
    args = ap.parse_args()

    config = FeatherlessConfig.from_env()
    print("Featherless configuration (key never shown):", config.public_summary())
    if config.unavailable_reason():
        print(f"-> Featherless not used: {config.unavailable_reason()}. Using the deterministic fallback.\n")
    explainer = Explainer(config)

    try:
        opt_ctx = load_controller_context(args.results_dir, args.scenario, args.controller, seed=0)
        qaoa_ctx = load_qaoa_context(args.results_dir, args.scenario, "qaoa_p1", seed=0)
        emg_ctx = load_emergency_context(args.results_dir, args.emergency_scenario, "fixed")
        env_ctx = load_controller_context(args.results_dir, args.environment_scenario, "adaptive")
    except (FileNotFoundError, LookupError) as exc:
        print(f"Could not load saved results ({type(exc).__name__}): {exc}. Run the earlier phase scripts first.")
        return 1

    show(f"Optimization: {args.controller} in '{args.scenario}' (seed 0)", explain_optimization(opt_ctx, explainer))
    show(f"QAOA: qaoa_p1 in '{args.scenario}' (seed 0)", explain_qaoa(qaoa_ctx, explainer))
    show(f"Emergency corridor: '{args.emergency_scenario}'", explain_emergency(emg_ctx, explainer))
    show(f"Environmental proxy: adaptive in '{args.environment_scenario}'", explain_environment(env_ctx, explainer))
    show("Signal choice: what does the pipeline say for I3? (explains a computed plan; chooses nothing)",
         explain_signal_choice(opt_ctx, "I3", explainer))
    print("=" * 78)
    print(f"API requests attempted: {explainer.requests_made} | numbers above come from saved simulator results;")
    print("Featherless text (if any) is explanation only. Environmental values are a simulation proxy, not measurements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
