# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from itertools import permutations
from random import seed

import networkx as nx
import pytest
from compiler_config.config import CompilerConfig, Qasm2Optimizations, Tket
from docplex.mp.model import Model
from numpy import array, random
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator as QuantumInstance
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)
from qiskit_algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (
    ADMMOptimizer,
    ADMMParameters,
    CobylaOptimizer,
    GroverOptimizer,
    MinimumEigenOptimizer,
)
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

from qat import qatconfig
from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.backends.realtime_chip_simulator import qutip_available
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.integrations.qasm import Qasm2Parser
from qat.purr.integrations.qiskit import QatBackend
from qat.purr.qat import execute_qasm, execute_qasm_with_metrics

from tests.unit.utils.qasm_qir import get_qasm2

qiskitconfig = qatconfig.SIMULATION.QISKIT


class TestQiskitSimulator:
    @pytest.mark.parametrize("strict_placement", [True, False])
    @pytest.mark.parametrize("qi0, qi1, valid", [(0, 1, True), (0, 2, False)])
    def test_coupling_enforcement_on_qasm_hardware(self, strict_placement, qi0, qi1, valid):
        coupling_map = [(0, 1), (1, 2)]
        hardware = get_default_qiskit_hardware(
            qubit_count=3, strict_placement=strict_placement, connectivity=coupling_map
        )
        builder = hardware.create_builder()
        builder.had(hardware.get_qubit(qi0))
        builder.cnot(hardware.get_qubit(qi0), hardware.get_qubit(qi1))
        builder.measure(hardware.get_qubit(qi0))
        builder.measure(hardware.get_qubit(qi1))
        runtime = hardware.create_runtime()
        if not valid and strict_placement:
            with pytest.raises(RuntimeError):
                runtime.execute(builder)
        else:
            results = runtime.execute(builder)
            assert len(results) > 0

    @pytest.mark.parametrize("with_seed", [False, True])
    def test_coupled_qasm_hardware(self, with_seed):
        """Includes a regression test with a seed that has caused problems in the past."""
        if with_seed:
            seed(454)
        hardware = get_default_qiskit_hardware(35)
        builder = self.parse_and_apply_optimiziations(hardware, "15qb.qasm")
        runtime = hardware.create_runtime()
        results = runtime.execute(builder)
        assert len(results) > 0

    def test_bitflip_noise_model(self):
        # Example error probabilities
        p_reset = 0.03
        p_meas = 0.1
        p_gate1 = 0.05

        # QuantumError objects
        error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
        error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
        error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
        error_gate2 = error_gate1.tensor(error_gate1)

        # Add errors to noise model
        noise_bit_flip = NoiseModel()
        noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
        noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
        noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
        noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

        qubit_count = 2
        hardware = get_default_qiskit_hardware(
            qubit_count=qubit_count, noise_model=noise_bit_flip
        )
        builder = hardware.create_builder()
        builder.had(hardware.get_qubit(0))
        builder.cnot(hardware.get_qubit(0), hardware.get_qubit(1))
        builder.measure(hardware.get_qubit(0))
        builder.measure(hardware.get_qubit(1))

        runtime = hardware.create_runtime()
        results = runtime.execute(builder)
        print(results)

    def test_thermal_relaxation_noise_model(self):
        # T1 and T2 values for qubits 0-3
        T1s = random.normal(
            50e3, 10e3, 4
        )  # Sampled from normal distribution mean 50 microsec
        T2s = random.normal(
            70e3, 10e3, 4
        )  # Sampled from normal distribution mean 50 microsec

        # Truncate random T2s <= T1s
        T2s = array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

        # Instruction times (in nanoseconds)
        time_u1 = 0  # virtual gate
        time_u2 = 50  # (single X90 pulse)
        time_u3 = 100  # (two X90 pulses)
        time_cx = 300
        time_reset = 1000  # 1 microsecond
        time_measure = 1000  # 1 microsecond

        # QuantumError objects
        errors_reset = [
            thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)
        ]
        errors_measure = [
            thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)
        ]
        errors_u1 = [thermal_relaxation_error(t1, t2, time_u1) for t1, t2 in zip(T1s, T2s)]
        errors_u2 = [thermal_relaxation_error(t1, t2, time_u2) for t1, t2 in zip(T1s, T2s)]
        errors_u3 = [thermal_relaxation_error(t1, t2, time_u3) for t1, t2 in zip(T1s, T2s)]
        errors_cx = [
            [
                thermal_relaxation_error(t1a, t2a, time_cx).expand(
                    thermal_relaxation_error(t1b, t2b, time_cx)
                )
                for t1a, t2a in zip(T1s, T2s)
            ]
            for t1b, t2b in zip(T1s, T2s)
        ]

        # Add errors to noise model
        noise_thermal = NoiseModel()
        for j in range(4):
            noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
            noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
            noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
            noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
            noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(4):
                noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

        qubit_count = 2
        hardware = get_default_qiskit_hardware(
            qubit_count=qubit_count, noise_model=noise_thermal
        )
        builder = hardware.create_builder()
        builder.had(hardware.get_qubit(0))
        builder.cnot(hardware.get_qubit(0), hardware.get_qubit(1))
        builder.measure(hardware.get_qubit(0))
        builder.measure(hardware.get_qubit(1))

        runtime = hardware.create_runtime()
        results = runtime.execute(builder)
        print(results)

    def test_depolarising_noise_model(self):
        prob_1 = 0.001  # 1-qubit gate
        prob_2 = 0.1  # 2-qubit gate

        # Depolarizing quantum errors
        error_1 = depolarizing_error(prob_1, 1)
        error_2 = depolarizing_error(prob_2, 2)

        # Add errors to noise model
        noise_depo = NoiseModel()
        noise_depo.add_all_qubit_quantum_error(error_1, ["u1", "u2", "u3"])
        noise_depo.add_all_qubit_quantum_error(error_2, ["cx"])

        qubit_count = 2
        hardware = get_default_qiskit_hardware(
            qubit_count=qubit_count, noise_model=noise_depo
        )
        builder = hardware.create_builder()
        builder.had(hardware.get_qubit(0))
        builder.cnot(hardware.get_qubit(0), hardware.get_qubit(1))
        builder.measure(hardware.get_qubit(0))
        builder.measure(hardware.get_qubit(1))

        runtime = hardware.create_runtime()
        results = runtime.execute(builder)
        print(results)

    def test_no_noise_model(self):
        qubit_count = 2
        hardware = get_default_qiskit_hardware(qubit_count=qubit_count)
        builder = hardware.create_builder()
        builder.had(hardware.get_qubit(0))
        builder.cnot(hardware.get_qubit(0), hardware.get_qubit(1))
        builder.measure(hardware.get_qubit(0))
        builder.measure(hardware.get_qubit(1))

        runtime = hardware.create_runtime()
        results = runtime.execute(builder)
        print(results)

    def parse_and_apply_optimiziations(
        self, hardware, qasm_file_name, parser=None, opt_config=None
    ) -> InstructionBuilder:
        """
        Helper that builds a basic hardware, applies general optimizations, parses the QASM
        then returns the resultant builder.
        """
        qasm = get_qasm2(qasm_file_name)
        if opt_config is None:
            opt_config = Qasm2Optimizations()
        qasm = DefaultOptimizers().optimize_qasm(qasm, hardware, opt_config)
        if parser is None:
            parser = Qasm2Parser()
        return parser.parse(hardware.create_builder(), qasm)

    def test_qasm_frontend_no_noise(self):
        builder = self.parse_and_apply_optimiziations(
            get_default_qiskit_hardware(4), "ghz.qasm"
        )
        runtime = builder.model.create_runtime()
        results = runtime.execute(builder)
        print(results)

    def test_qasm_frontend_noise(self):
        # Example error probabilities
        p_reset = 0.03
        p_meas = 0.1
        p_gate1 = 0.05

        # QuantumError objects
        error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
        error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
        error_gate1 = pauli_error([("X", p_gate1), ("I", 1 - p_gate1)])
        error_gate2 = error_gate1.tensor(error_gate1)

        # Add errors to noise model
        noise_bit_flip = NoiseModel()
        noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
        noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
        noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
        noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

        builder = self.parse_and_apply_optimiziations(
            get_default_qiskit_hardware(4, noise_bit_flip), "ghz.qasm"
        )
        runtime = builder.model.create_runtime()
        results = runtime.execute(builder)
        print(results)

    def test_merge_builders(self):
        qubit_count = 2
        hardware = get_default_qiskit_hardware(qubit_count)
        a = hardware.create_builder()
        a.X(hardware.get_qubit(0))

        b = hardware.create_builder()
        assert len(b.circuit.data) == 0
        b.merge_builder(a)
        assert len(b.circuit.data) == 1

    @pytest.mark.parametrize("index", [0, 1])
    def test_bitstring_ordering(self, index):
        qubit_count = 2
        hardware = get_default_qiskit_hardware(qubit_count)
        builder = hardware.create_builder()
        builder.X(hardware.get_qubit(index))
        builder.measure(hardware.get_qubit(0))
        builder.measure(hardware.get_qubit(1))
        runtime = builder.model.create_runtime()
        results = runtime.execute(builder)

        bitstring = "".join(["1" if i == index else "0" for i in range(2)])
        assert results.get(bitstring) == 1000

    @pytest.mark.parametrize(
        "qubits", [(q1, q2) for q1 in range(5) for q2 in range(q1 + 1, 6)]
    )
    def test_bitstring_ordering_qasm(self, qubits):
        """
        Test the execution of a QASM script using the qiskit backend to ensure
        that the readouts are in the correct order.
        """

        num_qubits = 6

        # Bell test for the two qubits
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{num_qubits}];
        creg c[{num_qubits}];
        h q[{qubits[0]}];
        cx q[{qubits[0]}], q[{qubits[1]}];
        measure q -> c;
        """

        # Execute the script
        hw = get_default_qiskit_hardware(num_qubits)
        result, _ = execute_qasm_with_metrics(qasm, hw)

        # Check the results
        bitstring0 = "0" * num_qubits
        bitstring1 = (
            ("0" * (qubits[0]))
            + "1"
            + ("0" * (qubits[1] - qubits[0] - 1))
            + "1"
            + ("0" * (num_qubits - qubits[1] - 1))
        )
        assert result["c"][bitstring0] + result["c"][bitstring1] == 1000

    @pytest.mark.parametrize(
        "qubits", [(q1, q2) for q1 in range(5) for q2 in range(q1 + 1, 6)]
    )
    def test_bitstring_out_of_order_qasm(self, qubits):
        """
        Test the execution of a QASM script using the qiskit backend to ensure
        that the readouts are in the correct order. The QASM script will ask
        the results of a bell state on various qubits to be stored in the first two
        bits of the classical register.
        """

        num_qubits = 6

        # Bell test for the two qubits
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{num_qubits}];
        creg c[{num_qubits}];
        h q[{qubits[0]}];
        cx q[{qubits[0]}], q[{qubits[1]}];
        measure q[{qubits[0]}] -> c[0];
        measure q[{qubits[1]}] -> c[1];
        """
        ctr = 2
        for i in range(num_qubits):
            if i not in qubits:
                qasm += f"""
                measure q[{i}] -> c[{ctr}];
                """
                ctr += 1

        # Execute the script
        hw = get_default_qiskit_hardware(num_qubits)
        result, _ = execute_qasm_with_metrics(qasm, hw)

        # Check the results
        bitstring0 = "0" * num_qubits
        bitstring1 = "11" + ("0" * (num_qubits - 2))
        assert result["c"][bitstring0] + result["c"][bitstring1] == 1000

    @pytest.mark.parametrize(
        "qubits", [(q1, q2) for q1 in range(5) for q2 in range(q1 + 1, 6)]
    )
    def test_bitstring_limited_qasm(self, qubits):
        """
        Test the execution of a QASM script using the qiskit backend to ensure
        that the readouts are in the correct order. The QASM script will ask
        the results of a bell state on various qubits to be stored in just two
        bits in a classical register.
        """

        num_qubits = 6

        # Bell test for the two qubits
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[{num_qubits}];
        creg c[2];
        h q[{qubits[0]}];
        cx q[{qubits[0]}], q[{qubits[1]}];
        measure q[{qubits[0]}] -> c[0];
        measure q[{qubits[1]}] -> c[1];
        """

        # Execute the script
        hw = get_default_qiskit_hardware(num_qubits)
        result, _ = execute_qasm_with_metrics(qasm, hw)

        # Check the results
        bitstring0 = "00"
        bitstring1 = "11"
        assert result["c"][bitstring0] + result["c"][bitstring1] == 1000

    @pytest.mark.parametrize(
        "qasm, result_expectations",
        [
            ("basic_results_formats.qasm", {"ab": {"00": 1000}, "c": 1000}),
            ("ordered_cregs.qasm", {"a": {"00": 1000}, "b": {"00": 1000}, "c": 1000}),
            ("split_measure_assign.qasm", {"a": {"01": 1000}, "b": {"10": 1000}}),
        ],
    )
    def test_multiple_cregs(self, qasm, result_expectations):
        hw = get_default_qiskit_hardware()
        result, _ = execute_qasm_with_metrics(get_qasm2(qasm), hw)
        assert all([key in result.keys() for key in result_expectations.keys()])
        for key, vals in result_expectations.items():
            if isinstance(vals, dict):
                assert result[key] == vals
            else:
                assert len(result[key].keys()) > 1
                assert sum(result[key].values()) == 1000

    @pytest.mark.parametrize("qubit_count", [2, 5, 10, 20, 37, 52])
    def test_mps_backend(self, qubit_count):
        # Tests the MPS backend
        hw = get_default_qiskit_hardware(qubit_count)
        circ = hw.create_builder()
        circ.X(hw.get_qubit(0), 0.5)
        for i in range(qubit_count - 1):
            circ.cnot(hw.get_qubit(i), hw.get_qubit(i + 1))
        for i in range(qubit_count):
            circ.measure(hw.get_qubit(i))
        engine = hw.create_engine()
        qiskitconfig.METHOD = "matrix_product_state"
        qiskitconfig.ENABLE_METADATA = True
        counts, metadata = engine.execute(circ)
        assert metadata["method"] == "matrix_product_state"
        assert (
            metadata["matrix_product_state_max_bond_dimension"]
            == qiskitconfig.OPTIONS["matrix_product_state_max_bond_dimension"]
        )
        assert (
            metadata["matrix_product_state_truncation_threshold"]
            == qiskitconfig.OPTIONS["matrix_product_state_truncation_threshold"]
        )
        assert counts["0" * qubit_count] + counts["1" * qubit_count] == 1000
        qiskitconfig.METHOD = "automatic"
        qiskitconfig.ENABLE_METADATA = False

    @pytest.mark.parametrize("qubit_count", [2, 5, 10, 20, 37, 52])
    def test_automatic_stabilizer_backend(self, qubit_count):
        # Tests that automatic settings choose a stabiliser backend when all gates
        # are cliffords
        hw = get_default_qiskit_hardware(qubit_count)
        circ = hw.create_builder()
        circ.had(hw.get_qubit(0))
        for i in range(qubit_count - 1):
            circ.cnot(hw.get_qubit(i), hw.get_qubit(i + 1))
        for i in range(qubit_count):
            circ.measure(hw.get_qubit(i))
        engine = hw.create_engine()
        qiskitconfig.ENABLE_METADATA = True
        counts, metadata = engine.execute(circ)
        assert metadata["method"] == "stabilizer"
        assert counts["0" * qubit_count] + counts["1" * qubit_count] == 1000
        qiskitconfig.ENABLE_METADATA = False

    def test_automatic_statevector_backend(self):
        # Tests that for a circuit with non-clifford gates and a small qubit count,
        # the method will default to state vector.
        hw = get_default_qiskit_hardware(2)
        circ = (
            hw.create_builder()
            .X(hw.get_qubit(0), 0.5)
            .cnot(hw.get_qubit(0), hw.get_qubit(1))
            .measure(hw.get_qubit(0))
            .measure(hw.get_qubit(1))
        )
        engine = hw.create_engine()
        qiskitconfig.ENABLE_METADATA = True
        counts, metadata = engine.execute(circ)
        assert metadata["method"] == "statevector"
        assert counts["00"] + counts["11"] == 1000
        qiskitconfig.ENABLE_METADATA = False

    def test_automatic_mps_backend(self):
        # Tests that for a circuit with non-clifford gates and a large qubit count,
        # the method will default to MPS after failing with statevector.
        qubit_count = 52
        hw = get_default_qiskit_hardware(qubit_count)
        circ = hw.create_builder()
        circ.X(hw.get_qubit(0), 0.5)
        for i in range(qubit_count - 1):
            circ.cnot(hw.get_qubit(i), hw.get_qubit(i + 1))
        for i in range(qubit_count):
            circ.measure(hw.get_qubit(i))
        engine = hw.create_engine()
        qiskitconfig.ENABLE_METADATA = True
        counts, metadata = engine.execute(circ)
        assert metadata["method"] == "matrix_product_state"
        assert (
            metadata["matrix_product_state_max_bond_dimension"]
            == qiskitconfig.OPTIONS["matrix_product_state_max_bond_dimension"]
        )
        assert (
            metadata["matrix_product_state_truncation_threshold"]
            == qiskitconfig.OPTIONS["matrix_product_state_truncation_threshold"]
        )
        assert counts["0" * qubit_count] + counts["1" * qubit_count] == 1000
        qiskitconfig.ENABLE_METADATA = False

    @pytest.mark.parametrize("qubit_pair", [[0, 2]])
    def test_cx_on_invalid_coupling(self, qubit_pair):
        """Tests that a CX gates on an invalid coupling fails due to placement
        verification."""

        model = get_default_qiskit_hardware(10)
        builder = model.create_builder()
        builder.cnot(model.qubits[qubit_pair[0]], model.qubits[qubit_pair[1]])
        builder.measure(model.qubits[qubit_pair[0]])
        builder.measure(model.qubits[qubit_pair[1]])
        engine = model.create_engine()
        with pytest.raises(RuntimeError):
            engine.execute(builder)

    @pytest.mark.parametrize("qubit_pair", [[1, 0]])
    def test_cx_on_wrong_direction(self, qubit_pair):
        """Tests that a CX gates on a coupling with a wrong direction warns about the
        wrong direction."""

        model = get_default_qiskit_hardware(10)
        builder = model.create_builder()
        builder.cnot(model.qubits[qubit_pair[0]], model.qubits[qubit_pair[1]])
        builder.measure(model.qubits[qubit_pair[0]])
        builder.measure(model.qubits[qubit_pair[1]])
        engine = model.create_engine()
        with pytest.warns(DeprecationWarning):
            results = engine.execute(builder)
        assert len(results) == 1
        assert "00" in results

    @pytest.mark.parametrize("with_seed", [False, True])
    def test_qft_circuit(self, with_seed):
        """Regression test for a QFT circuit, including a seed that has caused problems in
        the past."""
        if with_seed:
            seed(254)
        model = get_default_qiskit_hardware(10)
        qasm_str = get_qasm2("qft_5q.qasm")
        config = CompilerConfig(optimizations=Tket().default())
        results = execute_qasm(qasm_str, model, config)
        assert len(results) == 1
        assert "c" in results
        assert "00000" in results["c"]
        assert len(results["c"]) == 1


class TestQiskitOptimization:
    @pytest.mark.skip("RTCS needs to handle 4 qubits.")
    def test_minimum_eigen_solver(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/03_minimum_eigen_optimizer.html
        qubo = QuadraticProgram()
        qubo.binary_var("x")
        qubo.binary_var("y")
        qubo.binary_var("z")
        qubo.minimize(
            linear=[1, -2, 3], quadratic={("x", "y"): 1, ("x", "z"): -1, ("y", "z"): 2}
        )

        op, offset = qubo.to_ising()
        qp = QuadraticProgram()
        qp.from_ising(op, offset, linear=True)

        algorithm_globals.random_seed = 10598
        quantum_instance = QuantumInstance(
            QatBackend(),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=[0.0, 0.0])
        exact_mes = NumPyMinimumEigensolver()

        qaoa = MinimumEigenOptimizer(qaoa_mes)  # using QAOA
        exact = MinimumEigenOptimizer(
            exact_mes
        )  # using the exact classical numpy minimum eigen solver

        exact_result = exact.solve(qubo)
        qaoa_result = qaoa.solve(qubo)

        assert exact_result == qaoa_result

    @pytest.mark.skip("RTCS needs to handle 10 qubits.")
    def test_grover_adaptive_search(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/04_grover_optimizer.html
        backend = QatBackend()
        model = Model()
        x0 = model.binary_var(name="x0")
        x1 = model.binary_var(name="x1")
        x2 = model.binary_var(name="x2")
        model.minimize(-x0 + 2 * x1 - 3 * x2 - 2 * x0 * x2 - 1 * x1 * x2)
        qp = from_docplex_mp(model)

        grover_optimizer = GroverOptimizer(6, num_iterations=10, quantum_instance=backend)
        results = grover_optimizer.solve(qp)

        exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        exact_result = exact_solver.solve(qp)

        # TODO: Assert these results are what we should check against.
        self.assertEqual(results.x, exact_result.x)
        self.assertEqual(results.fval, exact_result.fval)

    @pytest.mark.skip("RTCS needs to handle 3 qubits.")
    def test_ADMM(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/05_admm_optimizer.html
        mdl = Model("ex6")

        v = mdl.binary_var(name="v")
        w = mdl.binary_var(name="w")
        t = mdl.binary_var(name="t")
        u = mdl.continuous_var(name="u")

        mdl.minimize(v + w + t + 5 * (u - 2) ** 2)
        mdl.add_constraint(v + 2 * w + t + u <= 3, "cons1")
        mdl.add_constraint(v + w + t >= 1, "cons2")
        mdl.add_constraint(v + w == 1, "cons3")

        # load quadratic program from docplex model
        qp = from_docplex_mp(mdl)

        admm_params = ADMMParameters(
            rho_initial=1001,
            beta=1000,
            factor_c=900,
            maxiter=100,
            three_block=True,
            tol=1.0e-6,
        )

        # define COBYLA optimizer to handle convex continuous problems.
        cobyla = CobylaOptimizer()

        # initialize ADMM with classical QUBO and convex optimizer
        exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())  # to solve QUBOs
        admm = ADMMOptimizer(
            params=admm_params, qubo_optimizer=exact, continuous_optimizer=cobyla
        )

        # run ADMM to solve problem
        result = admm.solve(qp)

        # initialize ADMM with quantum QUBO optimizer and classical convex optimizer
        qaoa = MinimumEigenOptimizer(QAOA(quantum_instance=QatBackend()))
        admm_q = ADMMOptimizer(
            params=admm_params, qubo_optimizer=qaoa, continuous_optimizer=cobyla
        )

        # run ADMM to solve problem
        result_q = admm_q.solve(qp)

        self.assertEqual(result_q.x, result.x)
        self.assertEqual(result_q.fval, result.fval)

    @pytest.mark.skip("RTCS needs to handle 4 qubits.")
    def test_max_cut(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html
        n = 4  # Number of nodes in graph
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, n, 1))
        elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]

        # tuple is (i,j,weight) where (i,j) is the edge
        G.add_weighted_edges_from(elist)

        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i, j] = temp["weight"]

        best_cost_brute = 0
        for b in range(2**n):
            x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
            cost = 0
            for i in range(n):
                for j in range(n):
                    cost = cost + w[i, j] * x[i] * (1 - x[j])
            if best_cost_brute < cost:
                best_cost_brute = cost
                xbest_brute = x

        colors = ["r" if xbest_brute[i] == 0 else "c" for i in range(n)]

        max_cut = Maxcut(w)
        qp = max_cut.to_quadratic_program()

        qubitOp, offset = qp.to_ising()
        exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
        result = exact.solve(qp)

        # Making the Hamiltonian in its full form and getting the lowest eigenvalue and
        # eigenvector
        ee = NumPyMinimumEigensolver()
        result = ee.compute_minimum_eigenvalue(qubitOp)

        x = max_cut.sample_most_likely(result.eigenstate)

        seed = 10598
        quantum_instance = QuantumInstance(
            QatBackend(), seed_simulator=seed, seed_transpiler=seed
        )

        # construct VQE
        spsa = SPSA(maxiter=300)
        ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
        vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)

        # run VQE
        result = vqe.compute_minimum_eigenvalue(qubitOp)

        # print results
        x = max_cut.sample_most_likely(result.eigenstate)

        vqe_optimizer = MinimumEigenOptimizer(vqe)

        # solve quadratic program
        result = vqe_optimizer.solve(qp)

        # TODO: Work out what to assert.

    @pytest.mark.skip("RTCS needs to handle 9 qubits.")
    def test_travelling_salesman(self):
        # TODO: Re-check comparisons and validity:
        #   https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html
        # Generating a graph of 3 nodes
        n = 3
        # num_qubits = n**2
        tsp = Tsp.create_random_instance(n, seed=123)
        adj_matrix = nx.to_numpy_matrix(tsp.graph)

        # Plotting args
        # colors = ["r" for node in tsp.graph.nodes]
        # pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]

        def brute_force_tsp(w, N):
            a = list(permutations(range(1, N)))
            last_best_distance = 1e10
            for i in a:
                distance = 0
                pre_j = 0
                for j in i:
                    distance = distance + w[j, pre_j]
                    pre_j = j
                distance = distance + w[pre_j, 0]
                order = (0,) + i
                if distance < last_best_distance:
                    best_order = order
                    last_best_distance = distance
            return last_best_distance, best_order

        best_distance, best_order = brute_force_tsp(adj_matrix, n)

        qp = tsp.to_quadratic_program()
        ee = NumPyMinimumEigensolver()
        # exact = MinimumEigenOptimizer(ee)
        qp2qubo = QuadraticProgramToQubo()
        qubo = qp2qubo.convert(qp)
        qubitOp, offset = qubo.to_ising()
        # exact_result = exact.solve(qubo)

        # Making the Hamiltonian in its full form and getting the lowest eigenvalue and
        # eigenvector
        result = ee.compute_minimum_eigenvalue(qubitOp)

        x = tsp.sample_most_likely(result.eigenstate)
        # z = tsp.interpret(x)

        algorithm_globals.random_seed = 123
        seed = 10598
        backend = QatBackend()
        quantum_instance = QuantumInstance(
            backend, seed_simulator=seed, seed_transpiler=seed
        )
        spsa = SPSA(maxiter=300)
        ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
        vqe = VQE(ry, optimizer=spsa, quantum_instance=quantum_instance)

        result = vqe.compute_minimum_eigenvalue(qubitOp)

        x = tsp.sample_most_likely(result.eigenstate)
        tsp.interpret(x)

        vqe_optimizer = MinimumEigenOptimizer(vqe)

        # solve quadratic program
        result = vqe_optimizer.solve(qp)


class TestQatBackend:
    @pytest.mark.skipif(
        not qutip_available, reason="Qutip is not available on this platform"
    )
    def test_simple_circuit(self):
        backend = QatBackend()
        circuit = QuantumCircuit(1)
        circuit.x(0)
        circuit.measure_all()
        result = backend.run(circuit, shots=1000).result()
        counts = result.get_counts()
        assert counts["1"] > 900
