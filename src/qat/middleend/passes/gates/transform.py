# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.core.pass_base import TransformPass
from qat.ir.gates.base import GateBase, QubitInstruction
from qat.ir.gates.gates_1q import Gate1Q, Phase, Rx, Rz, U, X, Z
from qat.ir.gates.gates_2q import CNOT, ECR, Gate2Q
from qat.ir.gates.native import X_pi_2
from qat.ir.qat_ir import QatIR
from qat.middleend.decompositions.gates import DefaultGateDecompositions
from qat.middleend.decompositions.pulses import (
    DefaultPulseDecompositions,
    PulseDecompositionBase,
)
from qat.model.hardware_model import PhysicalHardwareModel


class Decompose2QToCNOTs(TransformPass):
    r"""Iterates through the list of gates and decomposes any two-qubit gate down to CNOTs.

    This applies to any 2Q gate with the exception of ECR, :math:`ZX(\pi/4)` and
    :math:`ZX(-\pi/4)`, which sit lower in the DAG than CNOTs. The benefit of this pass is
    that CNOTs have relatively simple ir algebra and allow for easy optimisations. This
    might later be extended to include ZX-calculus techniques. An example of this is
    exploiting the fact that a CNOT is involutory, meaning we can cancel two CNOTs that
    appear sequentially.
    """

    def __init__(self):
        self.decompositions = DefaultGateDecompositions(extend_end_nodes=[CNOT, ECR])

    def run(self, ir: QatIR, *args, **kwargs):
        """:param ir: QatIR to transform."""
        new_instructions = []
        for gate in ir.instructions:
            if isinstance(gate, Gate2Q):
                new_instructions.extend(self.decompositions.decompose(gate))
            else:
                new_instructions.append(gate)
        ir.instructions = new_instructions
        return ir


class DecomposeToNativeGates(TransformPass):
    r"""Iterates through the list of gates and decomposes gate into a native gate set.

    Default native gates are :class:`Z`, :class:`X_pi_2` and :class:`ZX_pi_4`.
    """

    def __init__(self, native_gates: list[GateBase] | None = None):
        """
        :param native_gates: A list of native gates to decompose to.
        """
        self.decompositions = DefaultGateDecompositions(end_nodes=native_gates)

    def run(self, ir: QatIR, *args, **kwargs):
        """:param ir: QatIR to transform."""
        new_instructions = []
        for gate in ir.instructions:
            new_instructions.extend(self.decompositions.decompose(gate))
        ir.instructions = new_instructions
        return ir


class DecomposeToPulses(TransformPass):
    """A simple pass proof-of-concept pass that decomposes gates to pulse instructions.

    Each native gate can be written a sequence of pulse instructions. This pass decomposes
    each gate into the correct instructions. Note that this method will not be supported in
    the long-term, and will be replaced by something more performant once Qat IR is
    better established.

    A simple first step might be to write an analysis pass that extracts a set of unique
    gates + qubit(s) targets. For each gate+target pair, it could then determine the set of
    instructions for pair just once. There is the complication of how to deal with the
    Rz-gate, which is parameterised. This pass could then be adapted to use the results of
    this pass to more effeciently decompose instructions.
    """

    def __init__(
        self, model: PhysicalHardwareModel, decompositions: PulseDecompositionBase = None
    ):
        """Initiate the pass with the hardware model.

        :param model: The hardware model is needed to construct the pulses.
        """
        self.model = model
        if not decompositions:
            decompositions = DefaultPulseDecompositions()
        self.decompositions = decompositions

    def run(self, ir: QatIR, *args, **kwargs):
        """:param ir: The IR containing gates."""

        new_instructions = []
        for inst in ir.instructions:
            if not isinstance(inst, QubitInstruction):
                new_instructions.append(inst)
                continue
            new_instructions.extend(self.decompositions.decompose(inst, self.model))

        ir.instructions = new_instructions
        return ir


class Squash1QGates(TransformPass):
    r"""Squashes consecutive 1Q qubits into a single :math:`U(\theta, \phi, \lambda)` gate.

    Iterates through the list of instructions, looking for series of 1Q gates that are not
    interupted by any non-1Q gates. It then squashes these gates into a single U gate, which
    will later be composed into native gates.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        """:param ir: QatIR to transform."""
        gate_sequences: dict[int, list[int]] = {}
        new_instructions: list[GateBase] = []
        for idx, gate in enumerate(ir.instructions):
            if isinstance(gate, Gate1Q):
                gate_sequences.setdefault(gate.qubit, []).append(idx)
            else:
                for qubit in gate.qubits:
                    seq = gate_sequences.get(qubit, [])
                    if len(seq) == 0:
                        pass
                    else:
                        new_instructions.append(
                            self.squash_gate([ir.instructions[i] for i in seq])
                        )
                        del gate_sequences[qubit]
                new_instructions.append(gate)
        for seq in gate_sequences.values():
            new_instructions.append(self.squash_gate([ir.instructions[i] for i in seq]))
        ir.instructions = new_instructions
        return ir

    def squash_gate(self, gates: list[Gate1Q]):
        """Squashes a list of gates into a U gate."""
        # can probably be moved elsewhere for reuse.
        if len(gates) == 1:
            return gates[0]
        mat = np.eye(2)
        for gate in gates:
            mat = gate.matrix @ mat
        return U.from_matrix(gates[0].qubit, mat)


class Squash2QGates(TransformPass):
    r"""Squashes consecutive 1Q and 2Q qubits into a single gate.

    Iterates through the list of instructions, looking for consecutive 1Q and 2Q gates that
    act on the same pair of qubits. It then squashes these gates into a single U gate, which
    will later be composed into native gates.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        raise NotImplementedError(
            "Optimisation pass not yet implemented. Requires implementation of general 2Q "
            "decompositions."
        )


class SquashCNOTs(TransformPass):
    r"""Iterates through a list of gates and removes redundant CNOTs.

    If two CNOTs that share the same target and control qubits appear sequentially [note
    that this means with respect to these two qubits, and not other qubits], then they can
    be squashed to an identity matrix. This also exploits the fact :math:`R_{z}(\theta)`
    gates that act on the control qubit and :math:`R_{x}(\theta)` gates that act on target
    qubits can be commuted through CNOTs, allowing CNOTs to be cancelled.

    This can be considered a simple optimisation using ZX-calculus: in the future, we can
    build on this more.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        """:param ir: QatIR to transform."""
        # find all cnots and group them by control-target pairs
        cnots = [
            (pos, gate.qubits)
            for pos, gate in enumerate(ir.instructions)
            if isinstance(gate, CNOT)
        ]
        unique_pairs = set([cnot[1] for cnot in cnots])
        qubits_cnot_map = {
            pair: [cnot[0] for cnot in cnots if cnot[1] == pair] for pair in unique_pairs
        }

        # Find which CNOTs can be squashed by checking if CNOTs are sequential for those
        # qubits
        del_positions = []
        for pair, positions in qubits_cnot_map.items():
            if len(positions) == 1:
                continue
            pair_set = set(pair)
            intersections = [
                len(set(gate.qubits) & pair_set) == 0
                or (isinstance(gate, (Z, Rz, Phase)) and gate.qubit == pair[0])
                or (isinstance(gate, (X, Rx, X_pi_2)) and gate.qubit == pair[1])
                for gate in ir.instructions[positions[0] : positions[-1]]
            ]
            i = 0
            while i < len(positions) - 1:
                start_pos = positions[i] - positions[0] + 1
                end_pos = positions[i + 1] - positions[0]
                if all(intersections[start_pos:end_pos]):
                    del_positions.append(positions[i])
                    del_positions.append(positions[i + 1])
                    i += 2
                else:
                    i += 1

        # remove squashed CNOTs
        del_positions = sorted(del_positions, reverse=True)
        for pos in del_positions:
            del ir.instructions[pos]
        return ir


class SyncEndMeasurements(TransformPass):
    """Finds and synchronizes all end-circuit measurements to happen at the same time.

    Bubbles end-measurements to the end of the instruction list, and adds a barrier before
    them.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        raise NotImplementedError(
            "Transformation passes for the end measurements are not implemented yet."
        )


class SequentialEndMeasurements(TransformPass):
    """Finds all end measurements and enforces that they happen at different times.

    Bubbles end-measurements to the end of the instruction list, and adds a barrier between
    them.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        raise NotImplementedError(
            "Transformation passes for the end measurements are not implemented yet."
        )


class StaggeredEndMeasurements(TransformPass):
    """Finds all end measurements, and staggers their timings so that no Qubits with
    couplings have measurements occur at the same time.

    Bubbles end-measurements to the end of the instruction list. Some analysis is required
    to effeciently bundle together non-coupled Qubits.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        raise NotImplementedError(
            "Transformation passes for the end measurements are not implemented yet."
        )


class RemoveRedundantSWAPs(TransformPass):
    """Indentifies redundant SWAPs at the beginning and the end of a circuit, and removes
    them. This pass is designed to be used before routing.

    SWAP gates can frequently be found in circuits to allow for multi-qubit gates to be
    applied on qubits that are not connected in the hardware's topology. However, SWAP gates
    are expensive to implement (requiring three CNOT gates), and sometimes the same circuit
    can be implemented through a smart choice of logical-to-physical qubit placement.

    .. warning::
        This pass can modify a circuit that is capable of running on the given topology
        to one that might fail due to gates applied on non-connected qubits. The intention
        is that this is done before the routing and placement methods are applied to
        potentially reduce the overall 2Q-gate count.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        raise NotImplementedError("This transformation pass is not yet implemented.")
