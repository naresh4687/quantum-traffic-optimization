"""Record the number of passing tests for the dashboard header chip (results/dashboard/test_count.json).

Runs the whole suite; the file is written only if every test passes, so the chip can never show a failing count.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    match = re.search(r"(\d+) passed", tail)
    if proc.returncode != 0 or not match or re.search(r"\d+ (failed|error)", tail):
        print("tests did not all pass; nothing written:", tail)
        return 1
    out = ROOT / "results" / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_count.json").write_text(json.dumps({"tests_passed": int(match.group(1))}, indent=2) + "\n", encoding="utf-8")
    print("recorded:", tail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
