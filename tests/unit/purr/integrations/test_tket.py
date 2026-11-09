# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2026 Oxford Quantum Circuits Ltd
import random
import re
from math import pi
from pathlib import Path

import numpy as np
import pytest
from compiler_config.config import (
    CompilerConfig,
    Qasm2Optimizations,
    Tket,
    TketOptimizations,
)
from pytket import Circuit, OpType
from pytket.architecture import Architecture, RingArch
from pytket.qasm import circuit_from_qasm_str
from pytket.utils import compare_unitaries

from qat.model.loaders.purr import EchoModelLoader
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.devices import Qubit, QubitCoupling
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.purr.compiler.instructions import PhaseShift
from qat.purr.integrations.tket import (
    TketBuilder,
    TketQasmParser,
    TketToQatIRConverter,
    get_coupling_subgraphs,
    glob_phase_x_available,
    optimize_circuit,
    run_1Q_tket_optimizations,
    run_multiQ_tket_optimizations,
    run_tket_optimizations_qasm,
)

from tests.unit.utils.models import get_jagged_echo_hardware
from tests.unit.utils.qasm_qir import get_qasm2


@pytest.mark.parametrize(
    "couplings,subgraphs",
    [
        pytest.param(
            [(0, 1), (1, 2), (2, 3), (4, 3)],
            [[(0, 1), (1, 2), (2, 3), (4, 3)]],
            id="Continuous",
        ),
        pytest.param(
            [(23, 20), (25, 16), (25, 26)],
            [[(23, 20)], [(25, 16), (25, 26)]],
            id="Split",
        ),
    ],
)
def test_get_coupling_subgraphs(couplings, subgraphs):
    assert get_coupling_subgraphs(couplings) == subgraphs


@pytest.mark.parametrize("different_logical_physical_indices", [True, False])
def test_1Q_circuit_optimisation(different_logical_physical_indices):
    hw = EchoModelLoader(4).load()

    phys_to_log_qubit_map = {qubit.index: i for i, qubit in enumerate(hw.qubits)}
    if different_logical_physical_indices:
        for i, qubit in enumerate(hw.qubits):
            phys_to_log_qubit_map.pop(qubit.index)
            phys_to_log_qubit_map[qubit.index + 5] = i
            qubit.index = qubit.index + 5

    linear = {}
    qubit_qualities = {}
    for qubit in hw.qubits:
        linear[str(qubit.index)] = {
            "0|0": random.uniform(0.8, 1.0),
            "1|1": random.uniform(0.8, 1.0),
        }
        qubit_qualities[phys_to_log_qubit_map[qubit.index]] = (
            linear[str(qubit.index)]["0|0"] + linear[str(qubit.index)]["1|1"]
        ) / 2
    best_qubit = max(qubit_qualities, key=qubit_qualities.get)

    error_mit = ErrorMitigation(readout_mitigation=ReadoutMitigation(linear=linear))
    hw.error_mitigation = error_mit

    qasm_1q_x_input = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    x q;
    measure q -> c;
    """

    circ = TketQasmParser().parse(TketBuilder(), qasm_1q_x_input).circuit
    circ = run_1Q_tket_optimizations(circ, hw)
    assert best_qubit == circ.qubits[0].index[0]

    if different_logical_physical_indices:
        inv_map = {v: k for k, v in phys_to_log_qubit_map.items()}
        assert best_qubit != inv_map[best_qubit]


def test_2Q_circuit_optimisation():
    hw = EchoModelLoader(4).load()
    linear = {}
    qubit_qualities = {}
    for qubit in hw.qubits:
        linear[str(qubit.index)] = {
            "0|0": random.uniform(0.8, 1.0),
            "1|1": random.uniform(0.8, 1.0),
        }
        qubit_qualities[qubit.index] = (
            linear[str(qubit.index)]["0|0"] + linear[str(qubit.index)]["1|1"]
        ) / 2
    error_mit = ErrorMitigation(readout_mitigation=ReadoutMitigation(linear=linear))
    hw.error_mitigation = error_mit

    qubit_pair_qualities = {}
    for coupling in hw.qubit_direction_couplings:
        qubit_pair_qualities[coupling.direction] = (
            coupling.quality
            * hw.qubit_quality(coupling.direction[0])
            * hw.qubit_quality(coupling.direction[1])
        )

    best_qubit_pair = max(qubit_pair_qualities, key=qubit_pair_qualities.get)

    qasm_2q_bell_input = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];

    h q[0];
    cx q[0], q[1];
    measure q -> c;
    """
    opts = Tket().default()
    circ = TketQasmParser().parse(TketBuilder(), qasm_2q_bell_input).circuit
    circ = run_multiQ_tket_optimizations(circ, opts, hw, use_1Q_quality=True)
    circ_indices = [idx for qubit in circ.qubits for idx in qubit.index]

    for qubit in best_qubit_pair:
        assert qubit in circ_indices


def _one_directional_jagged_hardware():
    hw = get_jagged_echo_hardware(4, [2, 4, 5, 10])
    hw.qubit_direction_couplings.extend(
        QubitCoupling(direction) for direction in [(2, 4), (4, 5), (5, 10), (10, 2)]
    )
    return hw


def _unitary(circuit: Circuit) -> np.ndarray:
    """Returns the unitary of a single-register circuit, ignoring measurements."""
    gates = Circuit(circuit.n_qubits)
    for command in circuit.get_commands():
        if command.op.type != OpType.Measure:
            gates.add_gate(command.op, [qubit.index[0] for qubit in command.qubits])
    return gates.get_unitary()


@pytest.mark.parametrize(
    "hw",
    [
        pytest.param(EchoModelLoader(4).load(), id="bidirectional_ring"),
        pytest.param(_one_directional_jagged_hardware(), id="one_directional_jagged"),
    ],
)
class TestDirectionalCXGatesWithoutMappingPass:
    """Without the mapping pass, ``q[k]`` stays on logical qubit ``k`` and only CX
    directions may change."""

    opts = TketOptimizations.DirectionalCXGates
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0], q[1];
    cx q[2], q[1];
    h q[3];
    cx q[3], q[2];
    cx q[1], q[2];
    measure q[0] -> c[3];
    measure q[1] -> c[2];
    measure q[2] -> c[1];
    measure q[3] -> c[0];
    """

    def test_qubits_are_not_remapped_or_routed(self, hw):
        circuit = circuit_from_qasm_str(
            run_tket_optimizations_qasm(self.qasm, self.opts, hw)
        )

        assert {qubit.reg_name for qubit in circuit.qubits} == {"node"}
        # No SWAPs are added and each CX keeps its qubit pair.
        two_qubit_gates = [cmd for cmd in circuit.get_commands() if len(cmd.qubits) == 2]
        assert {cmd.op.type for cmd in two_qubit_gates} == {OpType.CX}
        qubit_pairs = sorted(
            sorted(qubit.index[0] for qubit in cmd.qubits) for cmd in two_qubit_gates
        )
        assert qubit_pairs == [[0, 1], [1, 2], [1, 2], [2, 3]]
        measurements = {
            qubit.index[0]: bit.index[0] for qubit, bit in circuit.qubit_to_bit_map.items()
        }
        assert measurements == {0: 3, 1: 2, 2: 1, 3: 0}

    def test_cx_gates_follow_coupling_directions(self, hw):
        logical = {qubit.index: i for i, qubit in enumerate(hw.qubits)}
        allowed_directions = {
            tuple(logical[index] for index in coupling.direction)
            for coupling in hw.qubit_direction_couplings
        }

        circuit = circuit_from_qasm_str(
            run_tket_optimizations_qasm(self.qasm, self.opts, hw)
        )

        cx_directions = {
            tuple(qubit.index[0] for qubit in cmd.qubits)
            for cmd in circuit.commands_of_type(OpType.CX)
        }
        assert cx_directions <= allowed_directions
        assert compare_unitaries(
            _unitary(circuit), _unitary(circuit_from_qasm_str(self.qasm))
        )

    def test_gate_on_uncoupled_qubits_raises(self, hw):
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        cx q[0], q[2];
        measure q -> c;
        """
        with pytest.raises(
            ValueError, match="Failed to apply circuit to hardware topology"
        ):
            run_tket_optimizations_qasm(qasm, self.opts, hw)


def test_directional_cx_gates_on_qubit_without_couplings_raises():
    hw = EchoModelLoader(4).load()
    hw.qubit_direction_couplings = [
        coupling for coupling in hw.qubit_direction_couplings if 3 not in coupling.direction
    ]
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[4];
    creg c[4];
    cx q[0], q[1];
    cx q[2], q[3];
    measure q -> c;
    """
    with pytest.raises(
        ValueError, match=re.escape("Qubits not in the hardware's coupling graph: q[3].")
    ):
        run_tket_optimizations_qasm(qasm, TketOptimizations.DirectionalCXGates, hw)


class TestTketOptimization:
    def _build_tket_objects(self, qasm_string, directional_arch=True):
        tket_builder: TketBuilder = TketQasmParser().parse(TketBuilder(), qasm_string)
        circ = tket_builder.circuit

        if directional_arch:
            architecture = Architecture(
                [(0, 1), (1, 2), (2, 3), (4, 3), (4, 5), (6, 5), (7, 6), (0, 7)]
            )
        else:
            architecture = RingArch(8)

        return circ, architecture

    def _run_qasm2(self, tket_opt, qasm_filename=None):
        """Helper class for testing various optimization configs against a varied QASM
        file."""
        if not qasm_filename:
            qasm_string = get_qasm2("random_n5_d5.qasm")
        else:
            qasm_string = get_qasm2(qasm_filename)

        opt_config = Qasm2Optimizations()
        opt_config.tket_optimizations |= tket_opt

        circ, architecture = self._build_tket_objects(qasm_string)
        return optimize_circuit(circ, architecture, opt_config.tket_optimizations)

    def test_globalise_phased_x(self, testpath):
        # TODO: Drop compatibility for next major version release, COMPILER-909
        if hasattr(TketOptimizations, "GlobalisePhasedX"):
            if glob_phase_x_available:
                assert self._run_qasm2(TketOptimizations.GlobalisePhasedX)
            else:
                with pytest.warns(
                    RuntimeWarning,
                    match="Requested Tket optimization `GlobalisePhasedX` is unavailable",
                ):
                    assert self._run_qasm2(TketOptimizations.GlobalisePhasedX)
        else:
            with pytest.raises(
                ValueError,
                match=r"(invalid value 128|128 is not a valid TketOptimizations)",
            ):
                TketOptimizations(128)
            with Path(
                testpath,
                "files",
                "compiler_config",
                "serialised_full_compiler_config_v02_with_globalise_phased_x.json",
            ).open("r") as f:
                conf_json = f.read()
            with pytest.warns(UserWarning, match="Ignoring unknown legacy flag bits 128"):
                CompilerConfig.create_from_json(conf_json)

    def test_clifford_simp(self):
        assert self._run_qasm2(TketOptimizations.CliffordSimp)

    def test_decompose_arbitrarily_controlled_gates(self):
        assert self._run_qasm2(TketOptimizations.DecomposeArbitrarilyControlledGates)

    def test_kak_decomposition(self):
        assert self._run_qasm2(TketOptimizations.KAKDecomposition)

    def test_peephole_optimize_2Q(self):
        assert self._run_qasm2(TketOptimizations.PeepholeOptimise2Q)

    def test_remove_discarded(self):
        assert self._run_qasm2(TketOptimizations.RemoveDiscarded)

    def test_remove_barriers(self):
        assert self._run_qasm2(TketOptimizations.RemoveBarriers)

    def test_remove_redundancies(self):
        assert self._run_qasm2(TketOptimizations.RemoveRedundancies)

    def test_three_qubit_squash(self):
        # `ThreeQubitSquash` cannot have barriers, so run other QASM input.
        assert self._run_qasm2(TketOptimizations.ThreeQubitSquash, qasm_filename="ghz.qasm")

    def test_simplify_measured(self):
        assert self._run_qasm2(TketOptimizations.SimplifyMeasured)

    def test_context_simp(self):
        assert self._run_qasm2(TketOptimizations.ContextSimp)

    def test_full_peephole(self):
        assert self._run_qasm2(TketOptimizations.FullPeepholeOptimise)


class TestTketToQatIRConverter:
    def test_get_qubit(self):
        model = get_default_echo_hardware(10)
        converter = TketToQatIRConverter(model)
        for i in range(10):
            qubit = converter.get_qubit(i)
            assert isinstance(qubit, Qubit)
            assert qubit.index == i

    @pytest.mark.parametrize(
        "params",
        [
            ("0.5", 0.5),
            ("1/2", 0.5),
            ("0", 0.0),
            ("0.254", 0.254),
            ("1", 1.0),
            ("-4/3", -4 / 3),
            ("pi", pi**2),
            ("pi / 2.54", (pi**2 / 2.54)),
            ("2.54 * pi", 2.54 * pi**2),
            ("2.54/pi", 2.54),
        ],
    )
    def test_convert_parameter(self, params):
        np.isclose(TketToQatIRConverter.convert_parameter(params[0]), params[1] * np.pi)

    def test_basic_commands(self):
        model = get_default_echo_hardware(10)
        converter = TketToQatIRConverter(model)
        circ = Circuit(2).Ry(4 / 3, 0).Rx(0.254 / np.pi, 1).Rz(1 / 2, 1).CX(1, 0).ECR(0, 1)
        builder = converter.convert(circ)
        direct_builder = model.create_builder()
        q0 = model.get_qubit(0)
        q1 = model.get_qubit(1)
        direct_builder.Y(q0, 4 * np.pi / 3).X(q1, 0.254).Z(q1, np.pi / 2).cnot(q1, q0).ECR(
            q0, q1
        )
        for i, inst in enumerate(builder.instructions):
            assert type(inst) is type(direct_builder.instructions[i])
            if isinstance(inst, PhaseShift):
                assert np.isclose(
                    inst.phase % (2 * np.pi),
                    direct_builder.instructions[i].phase % (2 * np.pi),
                )
