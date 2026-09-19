r"""QAOA for the signal-plan QUBO, run as a real parameterised circuit on Qiskit Aer.

Pipeline:  QUBO -> Ising Hamiltonian -> QAOA circuit -> Aer statevector / shots
           -> measured bitstrings -> QUBO energy + one-hot check -> comparison with exact.

This module never sees the exact optimum while running QAOA: ``run_qaoa`` takes only the
QUBO. The optimum enters only in ``compare_with_exact`` (evaluation), and the existing
``QUBO.energy`` stays the authoritative energy of every measured bitstring.

QUBO -> Ising
-------------
The QUBO is ``E(x) = offset + sum_{u<=v} Q_uv x_u x_v`` (``x_u^2 = x_u``, so ``Q_uu`` is
linear). Substitute ``x_u = (1 - Z_u) / 2``. On the computational basis state ``|x>`` the
Pauli ``Z_u`` has eigenvalue ``z_u = 1 - 2 x_u`` (``|0> -> +1``, ``|1> -> -1``)::

    Q_uu x_u        = Q_uu (1 - z_u) / 2
    Q_uv x_u x_v    = Q_uv (1 - z_u - z_v + z_u z_v) / 4          (u < v)

so ``E = H_C(z) = c + sum_u h_u z_u + sum_{u<v} J_uv z_u z_v`` with::

    h_u  = -Q_uu / 2  -  (1/4) sum_{v != u} Q_uv        J_uv = Q_uv / 4
    c    = offset + sum_u Q_uu / 2 + sum_{u<v} Q_uv / 4

This is an identity on every bit string (feasible or not), so QUBO and Ising energies
agree exactly up to rounding. The constant ``c`` shifts all energies equally: it does not
change which state is lowest and only a global phase in ``exp(-i gamma H_C)``, so it is
dropped from the circuit and added back when reporting energies.

Scaling. QUBO energies are ~1e4 (they contain the one-hot penalty), so a raw ``gamma``
would need to be ~1e-4. The circuit uses ``H~ = (H_C - c) / s`` with ``s`` the largest
absolute Ising coefficient (a fixed, deterministic positive number). ``E = c + s * H~``
is strictly increasing in ``H~``, so the energy ordering of all 2^n states is unchanged.
QAOA angles ``gamma`` refer to ``H~``.

Circuit (n = 18 qubits, one per QUBO variable, qubit u <-> variable u)
---------------------------------------------------------------------
``|+>^n`` then, for layers l = 1..p::

    U_C(gamma_l) = exp(-i gamma_l H~)      = prod RZ(2 gamma_l h~_u) prod RZZ(2 gamma_l J~_uv)
    U_M(beta_l)  = exp(-i beta_l sum X_u)  = prod RX(2 beta_l)

(``RZ(t) = exp(-i t Z / 2)``, ``RZZ(t) = exp(-i t ZZ / 2)``, ``RX(t) = exp(-i t X / 2)``.)
State: ``U_M(beta_p) U_C(gamma_p) ... U_M(beta_1) U_C(gamma_1) |+>^n``. Angles are stored as
``[gamma_1..gamma_p, beta_1..beta_p]``.

Bit order. Qiskit is little-endian: qubit 0 is the RIGHTMOST character of a count key and
bit 0 of a statevector index. ``key_to_bits`` reverses the key so that ``bits[u]`` is the
QUBO variable ``u``. Tests prepare known assignments to catch a reversed-order bug.

Expectation values are computed from the Aer statevector as ``<H~> = sum_x |psi_x|^2 H~(x)``
where ``H~(x)`` is the diagonal built from the Ising terms only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit
import qiskit_aer
from qiskit_aer import AerSimulator

from ..signals import SignalPlan
from .exact import TIE_TOL, ExactSolution, enumerate_feasible
from .qubo import QUBO


# -- Ising ------------------------------------------------------------------------------
@dataclass(frozen=True)
class IsingHamiltonian:
    """``H_C(z) = constant + sum_u h[u] z_u + sum_{u<v} J[(u,v)] z_u z_v``, ``z_u = 1 - 2 x_u``."""

    n: int
    h: np.ndarray
    J: Mapping[tuple[int, int], float]
    constant: float

    @property
    def scale(self) -> float:
        """Largest absolute coefficient (1.0 if the Hamiltonian is constant)."""
        biggest = max([float(np.abs(self.h).max(initial=0.0))] + [abs(c) for c in self.J.values()])
        return biggest if biggest > 0 else 1.0

    def _spins(self, bits: Sequence[int]) -> np.ndarray:
        arr = np.asarray(bits)
        if arr.shape != (self.n,) or not np.all((arr == 0) | (arr == 1)):
            raise ValueError(f"expected {self.n} bits, each 0 or 1")
        return 1.0 - 2.0 * arr.astype(float)

    def energy(self, bits: Sequence[int]) -> float:
        """Ising energy of the computational basis state ``|bits>``."""
        z = self._spins(bits)
        return float(self.constant + self.h @ z + sum(c * z[u] * z[v] for (u, v), c in self.J.items()))

    def diagonal(self, include_constant: bool = True, normalized: bool = False) -> np.ndarray:
        """Energy of every basis state, indexed by the integer whose bit u is ``x_u``.

        Built only from the Hamiltonian's terms. ``normalized`` gives ``H~ = (H - c) / s``.
        """
        idx = np.arange(2**self.n)
        z = 1.0 - 2.0 * ((idx[:, None] >> np.arange(self.n)[None, :]) & 1).astype(np.float64)
        diag = z @ self.h
        for (u, v), c in self.J.items():
            diag += c * z[:, u] * z[:, v]
        if normalized:
            return diag / self.scale
        return diag + (self.constant if include_constant else 0.0)

    def to_pauli_op(self, normalized: bool = True) -> SparsePauliOp:
        """Qiskit operator (without the constant) with explicit qubit indices."""
        s = self.scale if normalized else 1.0
        terms = [("Z", [u], float(self.h[u]) / s) for u in range(self.n) if self.h[u] != 0]
        terms += [("ZZ", [u, v], c / s) for (u, v), c in self.J.items() if c != 0]
        return SparsePauliOp.from_sparse_list(terms, num_qubits=self.n)


def qubo_to_ising(qubo: QUBO) -> IsingHamiltonian:
    """Convert with ``x = (1 - Z)/2`` (see module docstring); independent of ``QUBO.to_ising``."""
    n = qubo.variables.n_variables
    h = np.zeros(n)
    J: dict[tuple[int, int], float] = {}
    constant = qubo.offset
    for (u, v), q in qubo.coefficients.items():
        if u == v:
            constant += q / 2.0
            h[u] -= q / 2.0
        else:
            constant += q / 4.0
            h[u] -= q / 4.0
            h[v] -= q / 4.0
            J[(u, v)] = J.get((u, v), 0.0) + q / 4.0
    return IsingHamiltonian(n, h, J, constant)


# -- bit order ------------------------------------------------------------------------------
def key_to_bits(key: str, n: int) -> tuple[int, ...]:
    """Qiskit count key (qubit n-1 ... qubit 0) -> ``bits[u]`` = value of QUBO variable u."""
    key = key.replace(" ", "")
    if len(key) != n or set(key) - {"0", "1"}:
        raise ValueError(f"bad count key {key!r} for {n} qubits")
    return tuple(int(c) for c in reversed(key))


def bits_to_key(bits: Sequence[int]) -> str:
    """Inverse of ``key_to_bits`` (Qiskit little-endian string)."""
    return "".join(str(int(b)) for b in reversed(tuple(bits)))


def bits_to_index(bits: Sequence[int]) -> int:
    """Statevector index whose binary digit u is ``bits[u]``."""
    return sum(int(b) << u for u, b in enumerate(bits))


def variable_order_string(bits: Sequence[int]) -> str:
    """``x_0 x_1 ... x_{n-1}`` left to right (used in the stored result files)."""
    return "".join(str(int(b)) for b in bits)


# -- circuit ---------------------------------------------------------------------------------
def split_angles(angles: Sequence[float], p: int) -> tuple[np.ndarray, np.ndarray]:
    """``[gamma_1..gamma_p, beta_1..beta_p]`` -> (gammas, betas)."""
    arr = np.asarray(angles, dtype=float)
    if p < 1 or arr.shape != (2 * p,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"need {2 * p} finite angles for p={p}")
    return arr[:p], arr[p:]


def tqa_initial_angles(p: int, dt: float = 0.75) -> np.ndarray:
    """Deterministic initial angles from a Trotterised-annealing ramp (no randomness).

    ``gamma_l = ((l - 1/2)/p) dt``, ``beta_l = (1 - (l - 1/2)/p) dt`` for ``l = 1..p``.
    """
    if p < 1 or not dt > 0:
        raise ValueError("p must be >= 1 and dt > 0")
    ramp = (np.arange(1, p + 1) - 0.5) / p
    return np.concatenate([ramp * dt, (1.0 - ramp) * dt])


@dataclass(frozen=True)
class QAOACircuit:
    circuit: QuantumCircuit
    gammas: ParameterVector
    betas: ParameterVector
    p: int

    def bind(self, angles: Sequence[float]) -> QuantumCircuit:
        g, b = split_angles(angles, self.p)
        values = {self.gammas[i]: float(g[i]) for i in range(self.p)}
        values.update({self.betas[i]: float(b[i]) for i in range(self.p)})
        return self.circuit.assign_parameters(values)


def build_mixer_layer(circuit: QuantumCircuit, beta: Parameter, n: int) -> None:
    """``exp(-i beta sum_u X_u)``: one ``RX(2 beta)`` per qubit."""
    for u in range(n):
        circuit.rx(2 * beta, u)


def build_cost_layer(circuit: QuantumCircuit, gamma: Parameter, ham: IsingHamiltonian) -> None:
    """``exp(-i gamma H~)`` for the normalised, constant-free Ising Hamiltonian."""
    s = ham.scale
    for u in range(ham.n):
        if ham.h[u] != 0:
            circuit.rz(2 * gamma * float(ham.h[u] / s), u)
    for (u, v), c in sorted(ham.J.items()):
        if c != 0:
            circuit.rzz(2 * gamma * float(c / s), u, v)


def build_qaoa_circuit(ham: IsingHamiltonian, p: int) -> QAOACircuit:
    """``|+>^n`` followed by ``p`` alternating cost and mixer layers (parameterised)."""
    if isinstance(p, bool) or not isinstance(p, int) or p < 1:
        raise ValueError("p must be an integer >= 1")
    gammas, betas = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(ham.n, name=f"qaoa_p{p}")
    qc.h(range(ham.n))
    for layer in range(p):
        build_cost_layer(qc, gammas[layer], ham)
        build_mixer_layer(qc, betas[layer], ham.n)
    return QAOACircuit(qc, gammas, betas, p)


# -- configuration and results ---------------------------------------------------------------
@dataclass(frozen=True)
class QAOAConfig:
    """One configuration, used unchanged for every case (only ``p`` differs by run)."""

    p: int = 1
    shots: int = 8192
    seed: int = 0  # Aer sampling seed
    optimizer: str = "COBYLA"
    maxiter: int = 200
    rhobeg: float = 0.2  # initial COBYLA step in angle space
    tol: float = 1e-4
    tqa_dt: float = 0.75  # deterministic initial angles, see ``tqa_initial_angles``

    def __post_init__(self) -> None:
        if isinstance(self.p, bool) or not isinstance(self.p, int) or self.p < 1:
            raise ValueError("p must be an integer >= 1")
        if self.shots < 1 or self.maxiter < 1 or not self.rhobeg > 0 or not self.tol > 0:
            raise ValueError("shots, maxiter, rhobeg and tol must be positive")
        if self.optimizer != "COBYLA":
            raise ValueError("only the fixed COBYLA configuration is supported")

    def with_p(self, p: int) -> QAOAConfig:
        return replace(self, p=p)


@dataclass(frozen=True)
class QAOAResult:
    config: QAOAConfig
    n_qubits: int
    scale: float
    constant: float
    initial_angles: tuple[float, ...]
    gammas: tuple[float, ...]
    betas: tuple[float, ...]
    optimizer_success: bool
    optimizer_message: str
    n_iterations: int
    n_function_evals: int
    objective_trace: tuple[float, ...]  # <H~> at each evaluation
    initial_expectation: float  # <H_C> in QUBO energy units at the initial angles
    expectation: float  # <H_C> in QUBO energy units at the optimised angles
    feasible_probability: float  # exact (statevector) probability of a one-hot string
    shots: int
    counts: Mapping[str, int]  # variable-order bitstring x_0..x_{n-1} -> count
    sampled_energy: Mapping[str, float]  # QUBO energy of each sampled bitstring
    n_unique_sampled: int
    n_unique_feasible: int
    feasible_rate: float  # sampled fraction of shots that are one-hot
    best_sampled_energy: float  # over all samples (feasible or not)
    best_sampled_bits: tuple[int, ...]
    best_feasible_bits: tuple[int, ...] | None
    best_feasible_energy: float | None
    best_feasible_plans: Mapping[str, SignalPlan] | None
    logical_depth: int
    logical_ops: Mapping[str, int]
    basis_depth: int
    basis_ops: Mapping[str, int]
    optimize_seconds: float
    sample_seconds: float
    total_seconds: float
    qiskit_version: str = qiskit.__version__
    aer_version: str = qiskit_aer.__version__
    probabilities: np.ndarray = field(default=None, repr=False, compare=False)  # |psi_x|^2, index = bits


@dataclass(frozen=True)
class ExactComparison:
    exact_energy: float
    exact_plans: Mapping[str, SignalPlan]
    exact_n_optimal: int
    qaoa_energy: float | None  # best sampled FEASIBLE energy
    energy_gap: float | None  # qaoa_energy - exact_energy (>= 0)
    approximation_ratio: float | None  # (qaoa - exact) / |exact|; 0 = exact optimum
    optimum_over_qaoa: float | None  # exact / qaoa, in (0, 1] for positive energies
    found_optimum: bool
    optimum_probability_sampled: float  # fraction of shots that are an exact-optimal assignment
    optimum_probability_exact: float  # statevector probability of an exact-optimal assignment
    uniform_optimum_probability: float  # baseline: n_optimal / 2^n for uniform sampling
    uniform_feasible_probability: float  # baseline: 3^6 / 2^n


# -- evaluating measured bitstrings -----------------------------------------------------------
@dataclass(frozen=True)
class SampleSummary:
    """Every measured bitstring, scored by the existing QUBO evaluator."""

    counts: Mapping[str, int]  # variable-order bitstring x_0..x_{n-1} -> count
    energies: Mapping[str, float]  # QUBO energy of each measured bitstring
    feasible: Mapping[str, bool]  # one-hot at every intersection
    shots: int
    n_unique: int
    n_unique_feasible: int
    feasible_rate: float  # fraction of shots that are one-hot
    best_sampled_energy: float  # over all samples, feasible or not
    best_sampled_bits: tuple[int, ...]
    best_feasible_energy: float | None
    best_feasible_bits: tuple[int, ...] | None


def evaluate_samples(qubo: QUBO, raw_counts: Mapping[str, int]) -> SampleSummary:
    """Decode Qiskit count keys (little-endian) and score each with ``QUBO.energy``."""
    n = qubo.variables.n_variables
    counts: dict[str, int] = {}
    energies: dict[str, float] = {}
    feasible: dict[str, bool] = {}
    best_all: tuple[float, tuple[int, ...]] | None = None
    best_ok: tuple[float, tuple[int, ...]] | None = None
    ok_shots = 0
    for key, count in sorted(raw_counts.items()):
        bits = key_to_bits(key, n)
        name = variable_order_string(bits)
        counts[name] = counts.get(name, 0) + count
        energy = qubo.energy(bits)  # the authoritative QUBO evaluator
        energies[name] = energy
        feasible[name] = qubo.is_feasible(bits)
        if best_all is None or energy < best_all[0]:
            best_all = (energy, bits)
        if feasible[name]:
            ok_shots += count
            if best_ok is None or energy < best_ok[0]:
                best_ok = (energy, bits)
    if best_all is None:
        raise ValueError("no samples to evaluate")
    shots = sum(counts.values())
    return SampleSummary(
        counts=counts, energies=energies, feasible=feasible, shots=shots, n_unique=len(counts),
        n_unique_feasible=sum(feasible.values()), feasible_rate=ok_shots / shots,
        best_sampled_energy=best_all[0], best_sampled_bits=best_all[1],
        best_feasible_energy=None if best_ok is None else best_ok[0],
        best_feasible_bits=None if best_ok is None else best_ok[1],
    )


# -- engine ------------------------------------------------------------------------------------
class QAOAEngine:
    """Holds the Hamiltonian, its diagonal, the parameterised circuit and the Aer backend."""

    def __init__(self, qubo: QUBO, p: int, backend: AerSimulator | None = None):
        self.qubo = qubo
        self.ham = qubo_to_ising(qubo)
        self.p = p
        self.backend = backend or AerSimulator(method="statevector")
        self.diag_normalized = self.ham.diagonal(normalized=True)  # H~ for every basis state
        self.parameterised = build_qaoa_circuit(self.ham, p)
        sv = self.parameterised.circuit.copy()
        sv.save_statevector()
        self._sv_circuit = transpile(sv, self.backend, optimization_level=0)
        self.n_evaluations = 0

    def _statevector_circuit(self, angles: Sequence[float]) -> QuantumCircuit:
        g, b = split_angles(angles, self.p)
        values = {self.parameterised.gammas[i]: float(g[i]) for i in range(self.p)}
        values.update({self.parameterised.betas[i]: float(b[i]) for i in range(self.p)})
        return self._sv_circuit.assign_parameters(values)

    def statevector(self, angles: Sequence[float]) -> np.ndarray:
        result = self.backend.run(self._statevector_circuit(angles)).result()
        return np.asarray(result.get_statevector())

    def probabilities(self, angles: Sequence[float]) -> np.ndarray:
        psi = self.statevector(angles)
        probs = np.abs(psi) ** 2
        return probs / probs.sum()

    def expectation_normalized(self, angles: Sequence[float]) -> float:
        """``<H~>`` for the QAOA state at ``angles`` (the optimiser's objective)."""
        self.n_evaluations += 1
        return float(self.probabilities(angles) @ self.diag_normalized)

    def to_energy(self, expectation_normalized: float) -> float:
        """``<H~>`` -> ``<H_C>`` in QUBO energy units (adds the constant back)."""
        return self.ham.constant + self.ham.scale * expectation_normalized

    def sample(self, angles: Sequence[float], shots: int, seed: int) -> dict[str, int]:
        """Measure all qubits; returns Qiskit count keys (little-endian) -> counts."""
        qc = self.parameterised.bind(angles)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=0)
        return dict(self.backend.run(tqc, shots=shots, seed_simulator=seed).result().get_counts())


def _circuit_stats(engine: QAOAEngine, angles: Sequence[float]) -> tuple[int, dict, int, dict]:
    logical = engine.parameterised.circuit
    basis = transpile(engine.parameterised.bind(angles), basis_gates=["rz", "sx", "x", "cx"],
                      optimization_level=0)
    return (logical.depth(), dict(logical.count_ops()), basis.depth(), dict(basis.count_ops()))


# -- running QAOA ---------------------------------------------------------------------------------
def run_qaoa(qubo: QUBO, config: QAOAConfig | None = None,
             initial_angles: Sequence[float] | None = None) -> QAOAResult:
    """Optimise the QAOA angles on ``<H_C>`` with SciPy, then sample the optimised circuit.

    Uses only the QUBO. The optimiser minimises the expectation value; nothing about the
    exact optimum is available here. ``initial_angles`` (default: the deterministic
    ``tqa_initial_angles``) lets a caller warm-start, e.g. from a shallower run.
    """
    cfg = config or QAOAConfig()
    t0 = time.perf_counter()
    engine = QAOAEngine(qubo, cfg.p)
    n = engine.ham.n

    x0 = tqa_initial_angles(cfg.p, cfg.tqa_dt) if initial_angles is None else np.asarray(initial_angles, dtype=float)
    split_angles(x0, cfg.p)  # validates the length
    trace: list[float] = []

    def objective(angles: np.ndarray) -> float:
        value = engine.expectation_normalized(angles)
        trace.append(value)
        return value

    opt = minimize(objective, x0, method=cfg.optimizer,
                   options={"maxiter": cfg.maxiter, "rhobeg": cfg.rhobeg}, tol=cfg.tol)
    angles = np.asarray(opt.x, dtype=float)
    t_opt = time.perf_counter()

    probs = engine.probabilities(angles)
    idx = np.arange(2**n)
    bits_matrix = ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    per_node = bits_matrix.reshape(len(idx), len(qubo.variables.nodes), -1).sum(axis=2)
    feasible_probability = float(probs[(per_node == 1).all(axis=1)].sum())

    summary = evaluate_samples(qubo, engine.sample(angles, cfg.shots, cfg.seed))
    t_end = time.perf_counter()

    ld, lops, bd, bops = _circuit_stats(engine, angles)
    g, b = split_angles(angles, cfg.p)
    return QAOAResult(
        config=cfg, n_qubits=n, scale=engine.ham.scale, constant=engine.ham.constant,
        initial_angles=tuple(float(v) for v in x0),
        gammas=tuple(float(v) for v in g), betas=tuple(float(v) for v in b),
        optimizer_success=bool(opt.success), optimizer_message=str(opt.message),
        n_iterations=int(getattr(opt, "nit", len(trace))), n_function_evals=len(trace),
        objective_trace=tuple(trace),
        initial_expectation=engine.to_energy(trace[0]), expectation=engine.to_energy(engine.expectation_normalized(angles)),
        feasible_probability=feasible_probability, shots=summary.shots, counts=summary.counts,
        sampled_energy=summary.energies, n_unique_sampled=summary.n_unique,
        n_unique_feasible=summary.n_unique_feasible, feasible_rate=summary.feasible_rate,
        best_sampled_energy=summary.best_sampled_energy, best_sampled_bits=summary.best_sampled_bits,
        best_feasible_bits=summary.best_feasible_bits,
        best_feasible_energy=summary.best_feasible_energy,
        best_feasible_plans=None if summary.best_feasible_bits is None else qubo.decode(summary.best_feasible_bits),
        logical_depth=ld, logical_ops=lops, basis_depth=bd, basis_ops=bops,
        optimize_seconds=t_opt - t0, sample_seconds=t_end - t_opt, total_seconds=t_end - t0,
        probabilities=probs,
    )


def compare_with_exact(result: QAOAResult, qubo: QUBO, exact: ExactSolution) -> ExactComparison:
    """Evaluation only: compare a finished QAOA run with the exact solver's optimum.

    ``approximation_ratio = (E_qaoa - E_exact) / |E_exact|`` with both energies the
    authoritative QUBO energies (which include the QUBO offset). For a one-hot string the
    penalty terms and the offset cancel exactly, so a feasible energy is the surrogate
    traffic cost itself (positive, in vehicle-seconds): the ratio is not distorted by any
    arbitrary constant. 0 means QAOA's best feasible sample is optimal.
    """
    tol = TIE_TOL * max(1.0, abs(exact.energy))
    optimal_energy = exact.energy + tol
    p_sampled = sum(c for name, c in result.counts.items()
                    if result.sampled_energy[name] <= optimal_energy
                    and qubo.is_feasible([int(ch) for ch in name])) / result.shots
    # exact probability: every one-hot string whose QUBO energy equals the optimum
    space = enumerate_feasible(qubo)
    optimal_rows = np.flatnonzero(space.energies <= optimal_energy)
    indices = [bits_to_index(space.assignments[r]) for r in optimal_rows]
    p_exact = float(result.probabilities[indices].sum())
    n = result.n_qubits
    e_q = result.best_feasible_energy
    return ExactComparison(
        exact_energy=exact.energy, exact_plans=exact.plans, exact_n_optimal=exact.n_optimal,
        qaoa_energy=e_q,
        energy_gap=None if e_q is None else e_q - exact.energy,
        approximation_ratio=None if e_q is None else (e_q - exact.energy) / abs(exact.energy),
        optimum_over_qaoa=None if e_q is None else exact.energy / e_q,
        found_optimum=e_q is not None and e_q <= optimal_energy,
        optimum_probability_sampled=p_sampled, optimum_probability_exact=p_exact,
        uniform_optimum_probability=len(optimal_rows) / 2**n,
        uniform_feasible_probability=len(space.energies) / 2**n,
    )
