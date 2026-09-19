# Reproducibility

Everything in this repository is deterministic: the same inputs give the same numbers. Demand is seeded per `(seed, cycle)`, QAOA
sampling uses a fixed Aer seed, and no result depends on wall-clock time.

## What was validated

- **Fresh environment.** In a new Python 3.12.10 virtual environment created outside the repository (Windows 11, PowerShell), `pip install -r
  requirements.txt` followed by `pip install -e .` gave an importable `qtraffic` package and `python -m pytest -q` reported 452 passed.
  `pip check` reported no broken requirements. The resolved versions were numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, networkx 3.6.1,
  qiskit 2.5.2, qiskit-aer 0.17.2, streamlit 1.64.0, plotly 7.1.0, openai 2.54.0 and pytest 9.1.1. In that environment `pip install -r
  requirements.txt` **alone** was not enough: test collection failed with `ModuleNotFoundError: qtraffic`, which is why `pip install -e .` is
  a required step.
- **Not a real clone.** That environment was built from a copy of the repository's working tree, not from a `git clone` of GitHub, so a
  fresh GitHub clone has **not** been validated. The commands below are the ones that were run.
- **Current repository state.** The same commands were re-run in the project's own Python 3.12.10 environment: the requirements resolve,
  `pip install -e .` succeeds, `pip check` is clean, `qtraffic` imports from a directory outside the repository, the test suite reports
  452 passed, and `streamlit run app.py` starts and answers the Streamlit health check.
- Only Windows with PowerShell was used. Other operating systems have not been tried; only the environment-activation command differs there.

## Setup

```powershell
git clone https://github.com/naresh4687/quantum-traffic-optimization.git
cd quantum-traffic-optimization
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
streamlit run app.py
```

- `pip install -r requirements.txt` installs the full tested environment. Installing takes a few minutes (Qiskit and Streamlit are large).
- `pip install -e .` installs the `qtraffic` package from `src/`. It is required: the tests, `app.py` and the scripts import `qtraffic`,
  and one test starts a fresh interpreter that must be able to import it.
- `python -m pytest -q` should report **452 passed** (about 20 to 35 s; 6 warnings from Qiskit/SciPy, no skips).
- `streamlit run app.py` (from the repository root) starts the dashboard at `http://localhost:8501`. It needs no API key, no network and
  no extra files: it reads the saved results in `results/` and runs the simulator live. `python -m streamlit run app.py` is equivalent.
  By default Streamlit listens on all network interfaces; add `--server.address localhost` to keep the demo local.
- The demo preset is described in `docs/demo_preset.md`.

`pyproject.toml` describes the installable package. Its core dependencies (numpy, networkx) are enough for the simulator, controllers,
emergency corridor and metrics; the optional groups `optimization` (Qiskit, Aer, SciPy), `dashboard` (Streamlit, Plotly), `ai` (openai)
and `dev` (pytest) carry the version ranges that `requirements.txt` pins, so `python -m pip install -e ".[all]"` is an alternative to
the two install commands above. `requirements.txt` is the environment that was tested.

## Experiments

Every experiment script writes to `results/<phase>/` by default and needs no network access, no Featherless key and no special
hardware (QAOA runs on the classical Qiskit Aer simulator, on the CPU). Timings were measured on the development machine.
The saved files can differ from a re-run only in timing and version fields (`*_seconds`, library versions); every result number
reproduces exactly, including the raw QAOA sample counts. To compare without touching the tracked files, write to another folder with
`--out`.

| Area | Script and command | Purpose | Writes | Time |
|---|---|---|---|---|
| simulator | `python scripts/demo_phase1.py [--level low\|medium\|high] [--seed N] [--cycles N]` | 6-intersection grid with fixed control; prints determinism, conservation and propagation checks | nothing | 1 s |
| adaptive controller | `python scripts/run_phase2_experiments.py [--cycles 120] [--seeds 0 1 2 3 4] [--out results/phase2]` | Fixed vs Adaptive on identical demand (plus a 2-cycle-dwell ablation) | `results/phase2/` | a few minutes |
| QUBO | `python scripts/run_phase3_qubo.py [--seeds ...] [--cycles 60] [--out results/phase3]` | exact-QUBO controllers vs Fixed and Adaptive; energy vs waiting over all 729 assignments | `results/phase3/` | about 4 min |
| QUBO ablation | `python scripts/run_phase3_qubo.py --horizon-plan reference --out results/phase3_ablation_reference_plan` | earlier objective variant kept as an ablation | that folder | about 4 min |
| QUBO audit | `python scripts/audit_qubo.py [--out results/phase3/audit.json]` | penalty, one-hot, full 2^18 scan and repair check for 25 scenario-seed states | audit file | 6 s |
| QAOA | `python scripts/run_phase4_qaoa.py [--seed 0] [--cycles 60] [--out results/phase4] [--phase3 results/phase3]` | QAOA p=1, p=2 and the warm-start diagnostic on Qiskit Aer vs the exact optimum | `results/phase4/` | about 1 min |
| QAOA audit | `python scripts/audit_qaoa.py [--out results/phase4/audit.json] [--dir results/phase4]` | recomputes energies and probabilities from the saved counts; checks that no exact optimum reaches the optimizer | audit file | 16 s |
| emergency corridor | `python scripts/run_phase5_emergency.py [--out results/phase5]` | corridor vs no-override A/B on identical demand | `results/phase5/` | 20 s |
| metrics / environmental | `python scripts/run_phase6_metrics.py [--out results/phase6] [--results results]` | unified metrics and the waiting-based proxy; reads the saved Phase 3 to 5 results | `results/phase6/` | 10 s |
| Featherless | `python scripts/demo_featherless.py [--results-dir results] [--scenario ns_heavy]` | explains real saved results; local fallback without a key | nothing | 1 s |
| Featherless (live, optional) | `python scripts/test_featherless.py` | one real request with a tiny synthetic context; skips without a key | nothing | needs `FEATHERLESS_API_KEY` and `FEATHERLESS_MODEL`, and network |
| dashboard | `streamlit run app.py` | control-room dashboard | nothing | - |
| test count chip | `python scripts/record_test_count.py` | runs the suite and, only if every test passes, records the count shown in the dashboard header | `results/dashboard/test_count.json` | about 20 s |

Order: Phase 3 reads nothing; Phase 4 reads Phase 3; Phase 6 reads Phases 3 to 5. Phase 2 is independent. Only the Featherless live
smoke test needs the network, and only when a key and model are configured.

## What "reproduced" means here

- The saved Phase 2 to 6 result files and the Phase 3 ablation folder (39 files) were regenerated into a separate folder and compared
  field by field: all matched apart from timing and version fields. The Phase 5 and 6 scripts and the two audits were also run in the
  fresh environment described above (Phase 5 and 6 outputs matched the saved files).
- The QAOA numbers depend on the pinned Qiskit 2.5.x and Qiskit Aer 0.17.x behaviour; another Aer version could change sampling.
- The environmental values are simulation proxies (see `docs/metrics_and_environmental_model.md`).
