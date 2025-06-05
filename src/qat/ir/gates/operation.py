# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from pydantic import NonNegativeInt

from qat.ir.gates.base import QubitInstruction
from qat.ir.measure import AcquireMode


class Measure(QubitInstruction):
    """Instructs a measurement to be taken on the given qubit."""

    # TODO: decide if measure should have a "return type" (COMPILER-287)
    qubit: NonNegativeInt
    clbit: NonNegativeInt
    mode: AcquireMode = AcquireMode.INTEGRATOR

    @property
    def qubits(self):
        return (self.qubit,)


class Reset(QubitInstruction):
    r"""Instructs a qubit to be reset to it's lowest energy (:math:`|0>`) state."""

    # TODO: decide if reset should have a "reset type" (COMPILER-288)
    qubit: NonNegativeInt

    @property
    def qubits(self):
        return (self.qubit,)


class Barrier(QubitInstruction):
    """A barrier is a software construct that serves two purposes:
    #. Instructs the compiler to not optimize across the barrier (e.g. squashing of gates)
    #. Instructs the scheduler to synchronize targetted qubits up until this point.
    """

    qubits: list[NonNegativeInt]
