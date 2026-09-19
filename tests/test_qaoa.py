"""Tests for the QUBO -> Ising -> QAOA pipeline (real circuits, run on Qiskit Aer)."""

import inspect

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator
from scipy.linalg import expm

from qtraffic import Approach, Heading, grid_network
from qtraffic.optimization import (
    PLANS, QUBO, QAOAConfig, QAOAEngine, TrafficState, VariableMap, build_qaoa_circuit, build_qubo,
    compare_with_exact, evaluate_samples, key_to_bits, qubo_from_traffic, qubo_to_ising, run_qaoa,
    solve_exact, tqa_initial_angles,
)
from qtraffic.optimization.qaoa import (
    bits_to_index, bits_to_key, build_cost_layer, build_mixer_layer, split_angles,
    variable_order_string,
)

NET = grid_network(2, 3)
NODES = NET.nodes
BACKEND = AerSimulator(method="statevector")


def A(node, heading):
    return Approach(node, Heading[heading])


def real_qubo():
    q = {a: 0.0 for a in NET.approaches}
    t = {a: 0.0 for a in NET.approaches}
    q.update({A("I1", "S"): 25, A("I2", "S"): 25, A("I1", "E"): 30, A("I2", "E"): 34,
              A("I3", "E"): 38, A("I4", "N"): 22, A("I5", "W"): 12})
    t.update({A("I4", "S"): 15, A("I5", "S"): 8})
    return build_qubo(TrafficState(NET, q, t, {}))


def small_qubo(seed=0):
    """Two intersections = 6 qubits: same structure, fast enough for exhaustive checks."""
    rng = np.random.default_rng(seed)
    traffic = {(u, v): float(rng.uniform(0, 500)) for u in range(6) for v in range(u, 6)
               if u // 3 != v // 3 or u == v}
    return qubo_from_traffic(VariableMap(["I1", "I2"]), traffic)


def prepared_key(bits, backend=BACKEND, shots=32):
    """Prepare |bits> with X gates on qubit u <=> variable u, measure, return Qiskit keys."""
    n = len(bits)
    qc = QuantumCircuit(n)
    for u, b in enumerate(bits):
        if b:
            qc.x(u)
    qc.measure_all()
    return backend.run(transpile(qc, backend), shots=shots, seed_simulator=1).result().get_counts()


def assignment_bits(choice):
    bits = [0] * 18
    for i, node in enumerate(NODES):
        bits[3 * i + choice[node]] = 1
    return tuple(bits)


# -- 1. QUBO -> Ising ---------------------------------------------------------------------------
def test_ising_coefficients_match_a_hand_derivation():
    """E = 1 + 4 x0 + 8 x0 x1  ->  h0 = -2 - 2 = -4, h1 = -2, J01 = 2, c = 1 + 2 + 2 = 5."""
    qubo = QUBO(VariableMap(["I1", "I2"]), {}, {(0, 0): 4.0, (0, 1): 8.0}, 1.0, 1.0)
    ham = qubo_to_ising(qubo)
    assert ham.n == 6
    assert ham.h[0] == -4.0 and ham.h[1] == -2.0 and not ham.h[2:].any()
    assert ham.J == {(0, 1): 2.0} and ham.constant == 5.0
    x = np.array([1, 1, 0, 0, 0, 0])  # E = 1 + 4 + 8 = 13
    assert qubo.energy(x) == 13.0 and ham.energy(x) == 13.0
    assert ham.energy([0] * 6) == 1.0 == qubo.energy([0] * 6)


def test_ising_conversion_agrees_with_the_existing_qubo_to_ising():
    qubo = real_qubo()
    ham, other = qubo_to_ising(qubo), qubo.to_ising()
    assert ham.constant == pytest.approx(other.offset)
    assert np.allclose(ham.h, other.h)
    assert set(ham.J) == set(other.J) and all(ham.J[k] == pytest.approx(other.J[k]) for k in ham.J)


# -- 2. energy equivalence on random valid AND invalid bitstrings ---------------------------------
def test_ising_energy_equals_qubo_energy_for_random_bitstrings():
    qubo = real_qubo()
    ham = qubo_to_ising(qubo)
    rng = np.random.default_rng(0)
    checked_invalid = checked_valid = 0
    for _ in range(400):
        x = rng.integers(0, 2, 18)
        assert ham.energy(x) == pytest.approx(qubo.energy(x), rel=1e-9, abs=1e-6)
        checked_invalid += not qubo.is_feasible(x)
    for _ in range(400):
        choice = {n: int(rng.integers(0, 3)) for n in NODES}
        x = assignment_bits(choice)
        assert qubo.is_feasible(x)
        assert ham.energy(x) == pytest.approx(qubo.energy(x), rel=1e-9, abs=1e-6)
        checked_valid += 1
    assert checked_invalid > 300 and checked_valid == 400


def test_ising_diagonal_reproduces_the_qubo_energy_of_all_2_pow_18_strings_and_its_order():
    qubo = real_qubo()
    ham = qubo_to_ising(qubo)
    diag = ham.diagonal()
    idx = np.arange(2**18)
    bits = ((idx[:, None] >> np.arange(18)[None, :]) & 1).astype(float)
    energies = qubo.offset + ((bits @ qubo.matrix()) * bits).sum(axis=1)
    assert np.allclose(diag, energies, rtol=1e-9, atol=1e-6)
    assert int(np.argmin(diag)) == int(np.argmin(energies))
    order_a, order_b = np.argsort(diag, kind="stable"), np.argsort(energies, kind="stable")
    assert np.allclose(diag[order_a], energies[order_b])  # identical sorted spectrum
    # the normalised, constant-free H~ is a positive affine image: same ordering
    norm = ham.diagonal(normalized=True)
    assert np.allclose(ham.constant + ham.scale * norm, diag, rtol=1e-9, atol=1e-6)
    assert ham.scale > 0


# -- 3-5. circuits ------------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [1, 2])
def test_18_qubit_circuit_has_the_expected_structure(p):
    ham = qubo_to_ising(real_qubo())
    qaoa = build_qaoa_circuit(ham, p)
    qc = qaoa.circuit
    ops = dict(qc.count_ops())
    assert qc.num_qubits == 18 and qc.num_parameters == 2 * p
    assert ops["h"] == 18  # |+>^18
    assert ops["rx"] == 18 * p  # X mixer, one per qubit per layer
    assert ops["rzz"] == p * sum(1 for c in ham.J.values() if c != 0)
    assert ops["rz"] == p * int(np.count_nonzero(ham.h))
    assert len(ham.J) == 81  # 18 intra-intersection + 63 neighbour couplings
    assert "measure" not in ops


def test_invalid_layer_counts_are_rejected():
    ham = qubo_to_ising(small_qubo())
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            build_qaoa_circuit(ham, bad)
    with pytest.raises(ValueError):
        split_angles([0.1, 0.2, 0.3], 1)
    with pytest.raises(ValueError):
        split_angles([0.1, float("nan")], 1)


# -- 6. mixer and cost layers are the intended unitaries ------------------------------------------------
def test_x_mixer_layer_is_exp_minus_i_beta_sum_x():
    from qiskit.circuit import Parameter

    beta = Parameter("b")
    qc = QuantumCircuit(3)
    build_mixer_layer(qc, beta, 3)
    value = 0.37
    got = Operator(qc.assign_parameters({beta: value})).data
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I = np.eye(2, dtype=complex)
    ops = []
    for q in range(3):  # qubit q acts on tensor position (2 - q): little-endian
        mats = [X if k == q else I for k in (2, 1, 0)]
        ops.append(np.kron(np.kron(mats[0], mats[1]), mats[2]))
    expected = expm(-1j * value * sum(ops))
    assert np.allclose(got, expected)


def test_cost_layer_is_exp_minus_i_gamma_normalised_hamiltonian():
    from qiskit.circuit import Parameter

    ham = qubo_to_ising(small_qubo(3))
    gamma = Parameter("g")
    qc = QuantumCircuit(ham.n)
    build_cost_layer(qc, gamma, ham)
    value = 0.21
    got = Operator(qc.assign_parameters({gamma: value})).data
    diag = ham.diagonal(normalized=True)  # H~ on every basis state (index bit u = qubit u)
    assert np.allclose(got, np.diag(np.exp(-1j * value * diag)))


def test_qaoa_state_matches_qiskits_own_qaoa_ansatz():
    """Independent implementation: qiskit.circuit.library.QAOAAnsatz on the same H~."""
    ham = qubo_to_ising(small_qubo(4))
    for p in (1, 2):
        ours = build_qaoa_circuit(ham, p)
        angles = np.array([0.31, 0.17, 0.42, 0.26][: 2 * p])
        gammas, betas = split_angles(angles, p)
        theirs = QAOAAnsatz(cost_operator=ham.to_pauli_op(normalized=True), reps=p)
        values = {}
        for prm in theirs.parameters:  # named beta[i] / gamma[i]
            name = prm.name
            idx = int(name[name.index("[") + 1 : -1])
            values[prm] = float(betas[idx] if name.startswith("β") or name.startswith("beta") else gammas[idx])
        a = Statevector.from_instruction(ours.bind(angles))
        b = Statevector.from_instruction(theirs.assign_parameters(values))
        assert abs(abs(np.vdot(a.data, b.data)) - 1.0) < 1e-9  # equal up to global phase


# -- 7. deterministic initial parameters ----------------------------------------------------------------
def test_initial_angles_are_deterministic_and_documented():
    assert list(tqa_initial_angles(1)) == pytest.approx([0.375, 0.375])
    assert list(tqa_initial_angles(2)) == pytest.approx([0.1875, 0.5625, 0.5625, 0.1875])
    assert np.array_equal(tqa_initial_angles(3), tqa_initial_angles(3))
    g, b = split_angles(tqa_initial_angles(4), 4)
    assert np.all(np.diff(g) > 0) and np.all(np.diff(b) < 0)  # ramp up / ramp down
    with pytest.raises(ValueError):
        tqa_initial_angles(0)


# -- expectation value ---------------------------------------------------------------------------------
def test_engine_expectation_matches_qiskit_statevector_expectation():
    qubo = small_qubo(5)
    engine = QAOAEngine(qubo, p=2)
    angles = np.array([0.3, 0.2, 0.45, 0.25])
    ours = engine.expectation_normalized(angles)
    state = Statevector.from_instruction(engine.parameterised.bind(angles))
    theirs = state.expectation_value(engine.ham.to_pauli_op(normalized=True)).real
    assert ours == pytest.approx(theirs, abs=1e-9)
    raw = engine.to_energy(ours)  # in QUBO units: sum_x p(x) E(x)
    probs = np.abs(state.data) ** 2
    energies = np.array([qubo.energy([(i >> u) & 1 for u in range(6)]) for i in range(64)])
    assert raw == pytest.approx(float(probs @ energies), rel=1e-9)
    assert engine.probabilities(angles).sum() == pytest.approx(1.0)


# -- 9. Qiskit bit ordering -----------------------------------------------------------------------------
def test_count_key_is_little_endian_and_decoded_to_variable_order():
    counts = prepared_key([1] + [0] * 17)  # only qubit 0 set
    (key,) = counts
    assert key == "0" * 17 + "1"  # qubit 0 is the RIGHTMOST character
    bits = key_to_bits(key, 18)
    assert bits[0] == 1 and sum(bits) == 1
    assert bits_to_key(bits) == key and variable_order_string(bits) == "1" + "0" * 17
    with pytest.raises(ValueError):
        key_to_bits("101", 18)
    with pytest.raises(ValueError):
        key_to_bits("2" * 18, 18)


def test_a_reversed_bit_order_would_be_caught():
    """Prepare a known valid assignment, measure it, and decode it to the intended plans."""
    qubo = real_qubo()
    choice = {"I1": 2, "I2": 0, "I3": 1, "I4": 2, "I5": 0, "I6": 0}
    bits = assignment_bits(choice)
    (key,) = prepared_key(bits)
    decoded = qubo.decode(key_to_bits(key, 18))
    assert decoded == {n: PLANS[choice[n]] for n in NODES}
    reversed_bits = tuple(reversed(key_to_bits(key, 18)))  # what a reversed-order bug would give
    assert reversed_bits != bits and (not qubo.is_feasible(reversed_bits) or qubo.decode(reversed_bits) != decoded)
    # statevector path: the amplitude index of |bits> is bits_to_index(bits), and its
    # Hamiltonian diagonal entry is the QUBO energy of exactly those bits
    ham = qubo_to_ising(qubo)
    qc = QuantumCircuit(18)
    for u, b in enumerate(bits):
        if b:
            qc.x(u)
    sv = Statevector.from_instruction(qc)
    assert int(np.argmax(np.abs(sv.data))) == bits_to_index(bits)
    assert ham.diagonal()[bits_to_index(bits)] == pytest.approx(qubo.energy(bits), rel=1e-9)


def test_engine_sampling_path_uses_the_same_ordering():
    qubo = small_qubo(6)
    engine = QAOAEngine(qubo, p=1)
    counts = engine.sample(np.array([0.0, 0.0]), shots=2000, seed=3)  # gamma = beta = 0: |+>^n
    summary = evaluate_samples(qubo, counts)
    for name, c in summary.counts.items():
        bits = [int(ch) for ch in name]
        assert summary.energies[name] == pytest.approx(qubo.energy(bits))  # scored in variable order
    assert summary.shots == 2000


# -- 10-12. decoding, invalid samples, evaluation by the QUBO evaluator -------------------------------------
def test_valid_and_invalid_measured_bitstrings_are_identified():
    qubo = real_qubo()
    good = assignment_bits({n: 1 for n in NODES})
    two_on = list(good); two_on[0] = 1  # I1 has two plans on
    none_on = list(good); none_on[3:6] = [0, 0, 0]  # I2 has none
    raw = {}
    for bits, c in ((good, 60), (tuple(two_on), 25), (tuple(none_on), 15)):
        ((key, _),) = prepared_key(bits).items()
        raw[key] = raw.get(key, 0) + c
    summary = evaluate_samples(qubo, raw)
    assert summary.shots == 100 and summary.n_unique == 3 and summary.n_unique_feasible == 1
    assert summary.feasible_rate == pytest.approx(0.60)
    assert summary.feasible[variable_order_string(good)]
    assert not summary.feasible[variable_order_string(two_on)]
    assert not summary.feasible[variable_order_string(none_on)]
    assert summary.best_feasible_bits == good
    assert summary.best_feasible_energy == pytest.approx(qubo.energy(good))
    for name, e in summary.energies.items():
        assert e == pytest.approx(qubo.energy([int(ch) for ch in name]))
    assert qubo.decode(summary.best_feasible_bits) == {n: PLANS[1] for n in NODES}


def test_no_feasible_sample_is_reported_not_hidden():
    qubo = real_qubo()
    ((key, _),) = prepared_key([0] * 18).items()
    summary = evaluate_samples(qubo, {key: 10})
    assert summary.best_feasible_bits is None and summary.best_feasible_energy is None
    assert summary.feasible_rate == 0.0 and summary.best_sampled_bits == (0,) * 18


# -- 8, 12, 13. running QAOA -----------------------------------------------------------------------------------
def test_qaoa_run_is_deterministic_for_a_fixed_configuration():
    qubo = small_qubo(7)
    cfg = QAOAConfig(p=2, shots=1024, seed=5, maxiter=40)
    a, b = run_qaoa(qubo, cfg), run_qaoa(qubo, cfg)
    assert a.gammas == b.gammas and a.betas == b.betas
    assert a.counts == b.counts and a.objective_trace == b.objective_trace
    assert a.best_feasible_bits == b.best_feasible_bits


def test_qaoa_on_the_real_18_qubit_qubo_is_deterministic_and_evaluated_by_the_qubo_evaluator():
    qubo = real_qubo()
    cfg = QAOAConfig(p=1, shots=2048, seed=0, maxiter=30)
    a, b = run_qaoa(qubo, cfg), run_qaoa(qubo, cfg)
    assert a.n_qubits == 18 and a.gammas == b.gammas and a.counts == b.counts
    assert sum(a.counts.values()) == 2048
    for name, energy in a.sampled_energy.items():  # every sample scored by QUBO.energy
        assert energy == pytest.approx(qubo.energy([int(ch) for ch in name]))
    assert a.best_feasible_bits is not None and qubo.is_feasible(a.best_feasible_bits)
    assert a.best_feasible_energy == pytest.approx(qubo.energy(a.best_feasible_bits))
    assert qubo.decode(a.best_feasible_bits) == a.best_feasible_plans
    assert 0.0 <= a.feasible_probability <= 1.0 and 0.0 <= a.feasible_rate <= 1.0
    assert a.probabilities.sum() == pytest.approx(1.0)
    nonzero_j = sum(1 for c in qubo_to_ising(qubo).J.values() if c != 0)  # zero couplings emit no gate
    assert a.logical_ops["rzz"] == nonzero_j and a.basis_ops["cx"] == 2 * nonzero_j
    assert a.qiskit_version and a.aer_version


def test_optimiser_minimises_the_expectation_value_and_never_receives_the_optimum():
    qubo = real_qubo()
    result = run_qaoa(qubo, QAOAConfig(p=1, shots=512, maxiter=40))
    assert result.expectation <= result.initial_expectation + 1e-9  # COBYLA improved <H>
    # the trace is <H~>; converted back it starts at the initial expectation in QUBO units
    assert result.constant + result.scale * result.objective_trace[0] == pytest.approx(result.initial_expectation)
    assert result.constant + result.scale * min(result.objective_trace) <= result.initial_expectation + 1e-9
    params = inspect.signature(run_qaoa).parameters
    assert set(params) == {"qubo", "config", "initial_angles"}  # no exact-solution argument
    assert result.n_function_evals == len(result.objective_trace)


def test_zero_angles_give_the_uniform_superposition():
    """gamma = beta = 0: |+>^n, so each of the 729 valid strings has probability 2^-18."""
    qubo = real_qubo()
    engine = QAOAEngine(qubo, p=1)
    probs = engine.probabilities(np.zeros(2))
    assert np.allclose(probs, 1 / 2**18)
    assert engine.to_energy(engine.expectation_normalized(np.zeros(2))) == pytest.approx(
        float(engine.ham.diagonal().mean()))


def test_exact_optimum_comparison_is_consistent():
    qubo = real_qubo()
    exact = solve_exact(qubo)
    result = run_qaoa(qubo, QAOAConfig(p=1, shots=4096, maxiter=60))
    cmp = compare_with_exact(result, qubo, exact)
    assert cmp.exact_energy == exact.energy and cmp.exact_plans == exact.plans
    assert cmp.energy_gap >= -1e-9  # nothing feasible can beat the exact minimum
    assert cmp.approximation_ratio == pytest.approx(cmp.energy_gap / abs(exact.energy))
    assert cmp.optimum_over_qaoa == pytest.approx(exact.energy / cmp.qaoa_energy)
    assert cmp.found_optimum == (cmp.energy_gap <= 1e-6)
    assert 0.0 <= cmp.optimum_probability_sampled <= 1.0 and 0.0 <= cmp.optimum_probability_exact <= 1.0
    assert cmp.uniform_optimum_probability == pytest.approx(exact.n_optimal / 2**18)
    assert cmp.uniform_feasible_probability == pytest.approx(729 / 2**18)
    if cmp.found_optimum:
        assert cmp.approximation_ratio == pytest.approx(0.0, abs=1e-9)


def test_optimum_probability_is_the_statevector_mass_on_optimal_assignments():
    qubo = small_qubo(8)
    exact = solve_exact(qubo)
    result = run_qaoa(qubo, QAOAConfig(p=1, shots=1024, maxiter=30))
    cmp = compare_with_exact(result, qubo, exact)
    probs = result.probabilities
    optimal = [i for i in range(64)
               if qubo.is_feasible([(i >> u) & 1 for u in range(6)])
               and qubo.energy([(i >> u) & 1 for u in range(6)]) <= exact.energy + 1e-6]
    assert cmp.optimum_probability_exact == pytest.approx(float(probs[optimal].sum()))
    assert len(optimal) == exact.n_optimal


def test_config_validation_and_variants():
    cfg = QAOAConfig()
    assert (cfg.p, cfg.shots, cfg.optimizer, cfg.maxiter) == (1, 8192, "COBYLA", 200)
    assert cfg.with_p(2).p == 2 and cfg.with_p(2).maxiter == cfg.maxiter  # same config, only p differs
    for bad in ({"p": 0}, {"shots": 0}, {"maxiter": 0}, {"optimizer": "SLSQP"}, {"rhobeg": 0}):
        with pytest.raises(ValueError):
            QAOAConfig(**bad)
    with pytest.raises(ValueError):
        run_qaoa(small_qubo(), QAOAConfig(p=2, maxiter=5), initial_angles=[0.1, 0.2])  # needs 4 angles


def test_warm_start_from_a_shallower_run_never_starts_worse():
    """Zero angles in the extra layer is the identity, so p=2 can start exactly at the p=1 optimum."""
    qubo = small_qubo(9)
    p1 = run_qaoa(qubo, QAOAConfig(p=1, shots=256, maxiter=30))
    warm = [p1.gammas[0], 0.0, p1.betas[0], 0.0]
    p2 = run_qaoa(qubo, QAOAConfig(p=2, shots=256, maxiter=30), initial_angles=warm)
    assert p2.initial_expectation == pytest.approx(p1.expectation, rel=1e-9)
    assert p2.expectation <= p1.expectation + 1e-6
