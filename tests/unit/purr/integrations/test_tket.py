# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
import random

import numpy as np
import pytest
from compiler_config.config import Qasm2Optimizations, Tket, TketOptimizations
from pytket import Circuit
from pytket.architecture import Architecture, RingArch

from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.devices import Qubit
from qat.purr.compiler.hardware_models import ErrorMitigation, ReadoutMitigation
from qat.purr.compiler.instructions import PhaseShift
from qat.purr.integrations.tket import (
    TketBuilder,
    TketQasmParser,
    TketToQatIRConverter,
    get_coupling_subgraphs,
    optimize_circuit,
    run_1Q_tket_optimizations,
    run_multiQ_tket_optimizations,
)

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


def test_1Q_circuit_optimisation():
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
        """
        Helper class for testing various optimization configs against a varied QASM
        file.
        """
        if not qasm_filename:
            qasm_string = get_qasm2("random_n5_d5.qasm")
        else:
            qasm_string = get_qasm2(qasm_filename)

        opt_config = Qasm2Optimizations()
        opt_config.tket_optimizations |= tket_opt

        circ, architecture = self._build_tket_objects(qasm_string)
        return optimize_circuit(circ, architecture, opt_config.tket_optimizations)

    def test_globalise_phased_x(self):
        assert self._run_qasm2(TketOptimizations.GlobalisePhasedX)

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
            assert type(inst) == type(direct_builder.instructions[i])
            if isinstance(inst, PhaseShift):
                assert np.isclose(
                    inst.phase % (2 * np.pi),
                    direct_builder.instructions[i].phase % (2 * np.pi),
                )
