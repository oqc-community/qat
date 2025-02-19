# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from compiler_config.config import Qasm2Optimizations, TketOptimizations
from pytket import Circuit
from pytket.architecture import Architecture, RingArch

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.devices import Qubit
from qat.purr.compiler.instructions import PhaseShift
from qat.purr.integrations.tket import (
    TketBuilder,
    TketQasmParser,
    TketToQatIRConverter,
    get_coupling_subgraphs,
    optimize_circuit,
)

from tests.qat.qasm_utils import get_qasm2


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

    def _run_random(self, tket_opt):
        """
        Helper class for testing various optimization configs against a varied QASM
        file.
        """
        qasm_string = get_qasm2("random_n5_d5.qasm")
        opt_config = Qasm2Optimizations()
        opt_config.tket_optimizations |= tket_opt

        circ, architecture = self._build_tket_objects(qasm_string)
        return optimize_circuit(circ, architecture, opt_config.tket_optimizations)

    def test_globalise_phased_x(self):
        assert self._run_random(TketOptimizations.GlobalisePhasedX)

    def test_clifford_simp(self):
        assert self._run_random(TketOptimizations.CliffordSimp)

    def test_decompose_arbitrarily_controlled_gates(self):
        assert self._run_random(TketOptimizations.DecomposeArbitrarilyControlledGates)

    def test_kak_decomposition(self):
        assert self._run_random(TketOptimizations.KAKDecomposition)

    def test_peephole_optimize_2Q(self):
        assert self._run_random(TketOptimizations.PeepholeOptimise2Q)

    def test_remove_discarded(self):
        assert self._run_random(TketOptimizations.RemoveDiscarded)

    def test_remove_barriers(self):
        assert self._run_random(TketOptimizations.RemoveBarriers)

    def test_remove_redundancies(self):
        assert self._run_random(TketOptimizations.RemoveRedundancies)

    def test_tree_qubit_squash(self):
        assert self._run_random(TketOptimizations.ThreeQubitSquash)

    def test_simplify_measured(self):
        assert self._run_random(TketOptimizations.SimplifyMeasured)

    def test_context_simp(self):
        assert self._run_random(TketOptimizations.ContextSimp)

    def test_full_peephole(self):
        assert self._run_random(TketOptimizations.FullPeepholeOptimise)


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
