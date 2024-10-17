# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import Qasm2Optimizations, TketOptimizations
from pytket.architecture import Architecture, RingArch

from qat.purr.integrations.tket import (
    TketBuilder,
    TketQasmParser,
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
