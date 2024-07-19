import pytest
from numpy import array, random
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
)

from qat.purr.backends.qiskit_simulator import get_default_qiskit_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import Qasm2Optimizations
from qat.purr.compiler.optimisers import DefaultOptimizers
from qat.purr.integrations.qasm import Qasm2Parser
from tests.qat.qasm_utils import get_qasm2


class TestQiskitBackend:
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

    @pytest.mark.parametrize(
        "hardware",
        [
            get_default_qiskit_hardware(35),
        ],
    )
    def test_coupled_qasm_hardware(self, hardware):
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
    
    @pytest.mark.parametrize(
            "index", [0,1]
    )
    def test_bitstring_ordering(self, index):
        qubit_count = 2
        hardware = get_default_qiskit_hardware(qubit_count)
        builder = hardware.create_builder()
        builder.X(hardware.get_qubit(index))
        builder.measure(hardware.get_qubit(0))
        builder.measure(hardware.get_qubit(1))
        runtime = builder.model.create_runtime()
        results = runtime.execute(builder)

        bitstring = "".join(["1" if i==index else "0" for i in range(2) ])
        assert results.get(bitstring) == 1000

