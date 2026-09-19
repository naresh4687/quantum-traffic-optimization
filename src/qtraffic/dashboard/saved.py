"""Loading of the project's SAVED experiment results for the dashboard (Phases 4-6).

Pure data code: no Streamlit. Missing or unreadable files never raise: they become entries in ``warnings`` and
empty tables, so the dashboard degrades gracefully instead of crashing. Numbers are parsed from the files as
they are; nothing is filled in.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..signals import SignalPlan


def _parse(value):
    if value is None or value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value) if value.lstrip("-").isdigit() else float(value)
    except (ValueError, AttributeError):
        return value


@dataclass
class SavedResults:
    controller_summary: list = field(default_factory=list)  # Phase 6: scenario x controller means over seeds
    controller_rows: list = field(default_factory=list)  # Phase 6: per seed
    sensitivity: list = field(default_factory=list)  # Phase 6: proxy vs coefficient sets
    qaoa: list = field(default_factory=list)  # Phase 4: QAOA vs exact per scenario x variant
    qaoa_runs: list = field(default_factory=list)  # Phase 4: simulator results incl. plans
    emergency_summary: list = field(default_factory=list)  # Phase 6: corridor A/B means
    emergency_comparison: list = field(default_factory=list)  # Phase 6: corridor A/B per seed
    configuration: dict = field(default_factory=dict)  # Phase 6 configuration.json
    verification: dict = field(default_factory=dict)  # Phase 6 verification block
    test_count: dict | None = None
    warnings: list = field(default_factory=list)

    @property
    def has_phase6(self) -> bool:
        return bool(self.controller_summary)

    @property
    def has_qaoa(self) -> bool:
        return bool(self.qaoa)

    def summary_row(self, scenario: str, controller_key: str) -> dict | None:
        name = "qaoa_p1_saved" if controller_key == "qaoa_p1" else controller_key
        return next((r for r in self.controller_summary if r["scenario"] == scenario and r["controller"] == name), None)

    def qaoa_row(self, scenario: str, variant: str = "qaoa_p1", seed: int = 0) -> dict | None:
        return next((r for r in self.qaoa if r["case"] == scenario and r["variant"] == variant and r["seed"] == seed), None)

    def qaoa_plans(self, scenario: str, seed: int = 0) -> dict | None:
        """The saved QAOA p=1 best-sampled feasible plans (Phase 4 exists for seed 0)."""
        for r in self.qaoa_runs:
            if r["case"] == scenario and r["seed"] == seed and r["controller"] == "qaoa_p1" and r.get("plan_I1"):
                return {f"I{i}": SignalPlan.from_ns(int(str(r[f"plan_I{i}"]).split("/")[0].removeprefix("NS"))) for i in range(1, 7)}
        return None

    def qaoa_seeds(self, scenario: str) -> list[int]:
        return sorted({r["seed"] for r in self.qaoa_runs if r["case"] == scenario and r["controller"] == "qaoa_p1" and r.get("plan_I1")})


def _read_csv(path: Path, warnings: list) -> list:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(fh)]
    except (OSError, csv.Error, UnicodeDecodeError):
        warnings.append(f"saved results not found: {path.parent.name}/{path.name}")
        return []


def _read_json(path: Path, warnings: list):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        warnings.append(f"saved results not found: {path.parent.name}/{path.name}")
        return None


def load_saved_results(results_dir: str | Path) -> SavedResults:
    d = Path(results_dir)
    w: list = []
    summary = _read_json(d / "phase6" / "summary.json", w) or {}
    config = _read_json(d / "phase6" / "configuration.json", w) or {}
    test_count = None
    try:
        test_count = json.loads((d / "dashboard" / "test_count.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass  # optional
    return SavedResults(
        controller_summary=_read_csv(d / "phase6" / "controller_summary.csv", w),
        controller_rows=_read_csv(d / "phase6" / "controller_comparison.csv", w),
        sensitivity=_read_csv(d / "phase6" / "sensitivity.csv", w),
        qaoa=_read_csv(d / "phase4" / "qaoa_solutions.csv", w),
        qaoa_runs=_read_csv(d / "phase4" / "qaoa_runs.csv", w),
        emergency_summary=list(summary.get("emergency_summary", [])),
        emergency_comparison=_read_csv(d / "phase6" / "emergency_comparison.csv", w),
        configuration=config, verification=dict(summary.get("verification", {})), test_count=test_count, warnings=w)
