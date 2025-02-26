# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from functools import singledispatchmethod

from qat.ir.gates.base import QubitInstruction


class DecompositionBase:
    """Decomposition objects implement the rules for decomposing gates into native gates.

    The object should be equipped with a :meth:`decompose_op` method that is implemented
    using single dispatch. This allows us to register a decomposition for each relevant gate
    / operation type. Note that decompositions should be implemented carefully, and form a
    directed acyclic graph (DAG) that terminates at :class:`NativeGate` nodes.

    Assumming all relevant nodes are implemented, an operation can be decomposed to a series
    of native gates / operations using the :meth:`decompose` method, which will recursively
    call :meth:`decompose_op`.

    As a rule of thumb, try not to use logic on gate angles to define the decomposition.
    They should be kept general. If there are instances where a choice of angle(s) can be
    decomposed more effeciently, a transformation pass should be used to substitute the
    gate. This is done to allow predictable behaviour, and has parameterised circuits in
    mind.
    """

    end_nodes = tuple()

    def __init__(
        self,
        end_nodes: list[QubitInstruction] | None = None,
        extend_end_nodes: list[QubitInstruction] | None = None,
    ):
        """
        :param end_nodes: Optionally specify the nodes to end the decomposition recursion
            at. Defaults to :code:`decompositions.end_nodes`.
        :param extend_end_nodes: By default the decomposition will recurse until all gates
            are a :class:`NativeGate`. You can optionally add additional (non
            :class:`NativeGate`) gates to terminate at. This argument should only be
            specified if :code:`end_nodes=None`.
        """

        if end_nodes:
            self.end_nodes = tuple(set(end_nodes))
        elif extend_end_nodes:
            end_nodes = set(self.end_nodes)
            end_nodes.update(extend_end_nodes)
            self.end_nodes = tuple(end_nodes)

    def decompose(self, gate: QubitInstruction, *args):
        """Recursively decomposes an instruction into a product of native gates.

        :param gate: Gate to decompose.
        """
        if isinstance(gate, self.end_nodes):
            return [gate]

        gate_list = []
        for g in self.decompose_op(gate, *args):
            gate_list.extend(self.decompose(g, *args))

        return gate_list

    @singledispatchmethod
    def decompose_op(self, gate: QubitInstruction, *args):
        """Implements the definition of a decomposition of a qubit operation.

        The definition does not have to be in terms of end nodes, but decompositions must
        form a DAG. For example,

        .. code-block:: python

            CNOT -> {ECR, X, Z} -> {ZX_pi_4, Z_phase, X_pi_2}
            U -> {x_pi_2, Z_phase}
        """
        raise NotImplementedError(
            f"Decomposition for gate {gate.__class__.__name__} not implemented."
        )
