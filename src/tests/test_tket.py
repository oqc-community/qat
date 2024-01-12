# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.config import Qasm2Optimizations, TketOptimizations
from qat.purr.integrations.tket import run_tket_optimizations

from .qasm_utils import get_qasm2


class TestTketOptimization:
    def _run_random(self, tket_opt):
        """Helper class for testing various optimization configs against a varied QASM file."""
        qasm_string = get_qasm2("random_n5_d5.qasm")
        opt_config = Qasm2Optimizations()
        opt_config.tket_optimizations |= tket_opt
        hardware = get_default_echo_hardware(8)
        return run_tket_optimizations(
            qasm_string, opt_config.tket_optimizations, hardware
        )

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
