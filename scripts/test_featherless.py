"""OPTIONAL live smoke test for the Featherless integration. NOT part of pytest.

Run:  python scripts/test_featherless.py

Requires FEATHERLESS_API_KEY and FEATHERLESS_MODEL. It makes ONE small request with a tiny, clearly
synthetic structured context (test data, not project results), using the configured model, and prints
success or failure. The API key is never printed. Without a key it says the smoke test was skipped.
"""

from __future__ import annotations

import sys

from qtraffic.ai import Explainer, FeatherlessConfig, TrafficAnalysisContext


def main() -> int:
    config = FeatherlessConfig.from_env()
    print("Configuration (key never shown):", config.public_summary())
    if not config.has_api_key:
        print("SKIPPED: FEATHERLESS_API_KEY is not set, so no live request was made.")
        return 0
    if not config.has_model:
        print("SKIPPED: FEATHERLESS_MODEL is not set. Set it to the model supplied for the project; no model is assumed.")
        return 0

    context = TrafficAnalysisContext(  # tiny SYNTHETIC context: smoke-test data only
        scenario="smoke_test_synthetic", controller="fixed",
        traffic={"vehicles_admitted": 100.0, "throughput_vehicles_per_hour": 4000.0, "waiting_seconds_per_admitted": 60.0,
                 "average_queue_vehicles": 5.0, "max_queue_vehicles": 12.0})
    explainer = Explainer(config)
    result = explainer.explain("comparison", context)
    if result.source == "featherless":
        print(f"SUCCESS: received and validated an explanation from model {result.model!r} ({len(result.explanation.split())} words).")
        print(result.explanation[:400])
        return 0
    print(f"FAILURE: request did not produce a valid explanation. reason={result.fallback_reason} detail={result.error_detail}")
    print("The application itself is unaffected: it would use the deterministic fallback.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
