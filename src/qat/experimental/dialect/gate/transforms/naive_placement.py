# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""A pass that assigns each algorithmic qubit to a physical one by rewriting its type.

The pass uses a naive placement approach to assign qubits.
"""

from dataclasses import dataclass
from enum import auto

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.ir import SSAValue, StrEnum
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.gate.ir import (
    AlgorithmicQubitType,
    AllocatePhysicalQubitOp,
    DeallocateQubitOp,
    MeasureOp,
    QubitType,
)
from qat.experimental.dialect.gate.ir.ops import AllocateQubitOp, TwoQubitGate
from qat.experimental.passes.pass_ordering import OrderedPass
from qat.experimental.system_data.circuit.qubits import QubitView
from qat.experimental.system_data.circuit.topology import TopologyView


class PlacementPolicy(StrEnum):
    """The policy used to assign algorithmic qubits to physical ones."""

    best_coupling = auto()
    explicit = auto()
    as_written = auto()


@dataclass(frozen=True)
class AllocDetail:
    """What the pass learned about one qubit allocation before any rewriting.

    :ivar alloc_op: The ``AllocateQubitOp`` being analysed.
    :ivar is_algorithmic: Whether the allocation is for an algorithmic qubit
    :ivar place: Whether the qubit is to be kept and placed on a physical qubit.
    :ivar drop: Whether the qubit is to be dropped (unused)
    :ivar qubit_values: Every SSA value in the qubit's chain, after the allocation result
        to the last gate result, in order.
    :ivar dealloc_op: The ``DeallocateQubitOp`` ending the chain, or ``None``.
    """

    alloc_op: AllocateQubitOp
    is_algorithmic: bool
    place: bool
    drop: bool
    qubit_values: list[SSAValue]
    dealloc_op: DeallocateQubitOp | None


@dataclass(frozen=True)
class NaiveQubitPlacementPass(OrderedPass, ModulePass):
    """A pass that assigns each algorithmic qubit to a physical one by rewriting its type.

    The pass uses a naive placement policy to assign qubits, which can be one of:

    - best_coupling: assign qubits to the highest fidelity or best coupled physical qubits
      (1 or 2 qubits only). Inputs that mix physical and algorithmic qubits are rejected.
    - explicit: assign qubits to the physical qubits specified in the explicit_qubit_ids
    - as_written: assign qubits to the physical qubits in the order they are allocated

    Unused allocated qubits are dropped (including those that are deallocated before any
    use). The pass will raise an error if a two-qubit gate is applied to uncoupled qubits.
    Couplings are directed, so the qubit view must list a link in both directions for a gate
    to be accepted either way round.

    Functions that take or return qubits are rejected and should be inlined first. The
    input module must be valid, as for any xDSL pass.

    The pass validates up front so the module is left unchanged if validation fails.
    """

    name = "gate.naive-qubit-placement"

    qubit_view: QubitView
    policy: PlacementPolicy
    explicit_qubit_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.policy is PlacementPolicy.explicit and self.explicit_qubit_ids is None:
            raise ValueError("the explicit policy needs explicit_qubit_ids")
        if (
            self.policy is not PlacementPolicy.explicit
            and self.explicit_qubit_ids is not None
        ):
            raise ValueError(
                f"explicit_qubit_ids is only used by the explicit policy, not {self.policy}"
            )
        if self.policy is PlacementPolicy.explicit:
            if len(set(self.explicit_qubit_ids)) != len(self.explicit_qubit_ids):
                raise ValueError(
                    f"explicit_qubit_ids must be unique, but got {self.explicit_qubit_ids}"
                )
            for str_id in self.explicit_qubit_ids:
                if str_id not in self.qubit_view.qubits:
                    raise ValueError(
                        f"explicit qubit id {str_id} is not present in the qubit view"
                    )

    @staticmethod
    def _qubit_value_chain(value: SSAValue) -> list[SSAValue]:
        """Follow the chain of qubit values from an allocation to its uses terminating when
        there are no more uses or an operation doesn't return an algorithmic qubit in the
        same place as an algorithmic qubit operand.

        When the terminating operation isn't a measurement or a deallocation, raise an
        exception. Return the list of qubit values in the chain.
        """
        qubit_values = []
        while value.uses:
            uses = list(value.uses)
            if len(uses) != 1:
                raise PassFailedException(
                    "Qubit placement pass only supports single-use qubits, "
                    f"but got {len(uses)} uses for {value}"
                )
            use = uses[0]
            if use.operation.regions:
                raise PassFailedException(
                    "Qubit placement does not support qubits passed through"
                    f" {use.operation.name}"
                )
            # assume the qubit value is at the same index in the results and operands lists.
            # Ops outside the gate dialect, such as calls, have signatures that retyping
            # would not update, so they are rejected below
            if (
                use.operation.name.startswith("gate.op.")
                and use.index < len(use.operation.results)
                and isinstance(use.operation.results[use.index].type, QubitType)
            ):
                value = use.operation.results[use.index]
                qubit_values.append(value)
            elif isinstance(use.operation, DeallocateQubitOp | MeasureOp):
                break
            else:
                raise PassFailedException(
                    "Qubit placement does not support qubits consumed by "
                    f"{use.operation.name}"
                )
        return qubit_values

    @staticmethod
    def _reject_qubit_function_signatures(op: ModuleOp) -> None:
        """Raise if any function, including a declaration, takes or returns a qubit.

        Qubits must come from allocations in the module so that the pass can follow them and
        place them. A qubit argument would be placed by the caller, and a returned qubit has
        no allocation to place, so the pass cannot handle either without updating signatures
        and call sites. Functions should be inlined before placement. Other region-holding
        ops, such as loops, are rejected when the qubit chain is followed.
        """
        for func_op in op.walk():
            if not isinstance(func_op, FuncOp):
                continue
            signature = (*func_op.function_type.inputs, *func_op.function_type.outputs)
            for qubit_type in signature:
                if isinstance(qubit_type, QubitType):
                    raise PassFailedException(
                        "Qubit placement does not support qubits in function signatures, "
                        f"but got {qubit_type} on {func_op.sym_name.data}. Inline "
                        "functions before placement"
                    )

    def _analyse_allocs(self, alloc_ops: list[AllocateQubitOp]) -> list[AllocDetail]:
        """Analyse each allocation operation, recording what is needed to validate and
        rewrite it.

        For every allocation this follows the qubit's value chain (raising if the qubit is
        used in an unsupported way), notes whether the qubit is algorithmic or physical, and
        decides whether it is kept and placed or dropped as unused.

        Do this up front so the results can be used in both validation and rewriting.

        :param alloc_ops: The allocation operations in the module.
        :returns: One :class:`AllocDetail` per allocation, in the same order.
        """
        alloc_details = []
        for alloc_op in alloc_ops:
            qubit_values = self._qubit_value_chain(alloc_op.result)
            alloc_uses = list(alloc_op.result.uses)
            # check operation result type to capture both AllocateAlgorithmicQubitOp and
            # AllocateQubitOp with AlgorithmicQubitType
            is_algorithmic = isinstance(alloc_op.result.type, AlgorithmicQubitType)
            used = bool(alloc_uses) and not isinstance(
                alloc_uses[0].operation, DeallocateQubitOp
            )
            place = is_algorithmic and used
            # we could drop unused physical allocations but assume any inputs with physical
            # qubits have been deliberately designed that way
            drop = is_algorithmic and not used

            dealloc_op = (
                alloc_uses[0].operation
                if alloc_uses and isinstance(alloc_uses[0].operation, DeallocateQubitOp)
                else None
            )
            alloc_detail = AllocDetail(
                qubit_values=qubit_values,
                alloc_op=alloc_op,
                is_algorithmic=is_algorithmic,
                place=place,
                drop=drop,
                dealloc_op=dealloc_op,
            )
            alloc_details.append(alloc_detail)
        return alloc_details

    def _validate_and_generate_qubit_ids(
        self, alloc_details: list[AllocDetail], topology_view: TopologyView | None
    ) -> tuple[int, ...]:
        """Generate integer physical qubit indices for the kept allocated qubits according
        to the placement policy."""
        match self.policy:
            case PlacementPolicy.as_written:
                indices = {props.index for props in self.qubit_view.qubits.values()}
                qubit_ids = []
                for k, d in enumerate(d for d in alloc_details if d.is_algorithmic):
                    if not d.place:
                        continue
                    if k not in indices:
                        raise PassFailedException(
                            f"Qubit index {k} is not present in the qubit view"
                        )
                    qubit_ids.append(k)
                qubit_ids = tuple(qubit_ids)
            case PlacementPolicy.explicit:
                qubit_ids = []
                declared = [d for d in alloc_details if d.is_algorithmic]
                if not declared:
                    # already placed, so there is nothing to map the ids onto
                    return ()
                if len(self.explicit_qubit_ids) != len(declared):
                    raise PassFailedException(
                        f"Number of explicit qubit ids ({len(self.explicit_qubit_ids)}) "
                        f"does not match number of allocated qubits ({len(declared)})"
                    )
                for str_id, d in zip(self.explicit_qubit_ids, declared, strict=True):
                    if not d.place:
                        continue
                    qubit_ids.append(self.qubit_view.qubits[str_id].index)
                qubit_ids = tuple(qubit_ids)
            case PlacementPolicy.best_coupling:
                qubit_ids = self._best_coupling_ids(alloc_details, topology_view)
            case _:
                raise ValueError(f"Unknown placement policy: {self.policy}")

        return qubit_ids

    def _best_coupling_ids(
        self, alloc_details: list[AllocDetail], topology_view: TopologyView | None
    ) -> tuple[int, ...]:
        """Generate physical qubit indices for the kept allocated qubits according to the
        best coupling placement policy.

        This policy is only implemented for 1 or 2 qubits.
        """
        place = [d for d in alloc_details if d.place]
        if place and any(not d.is_algorithmic for d in alloc_details):
            raise PassFailedException(
                "best_coupling placement does not support inputs that mix physical and "
                "algorithmic qubits"
            )
        if len(place) > 2:
            raise NotImplementedError(
                "Best coupling placement policy is not implemented"
                f" for more than 2 qubits - {len(place)} requested"
            )
        if len(place) == 1:
            fidelities = {
                qubit_id: props.measurement_fidelity
                for qubit_id, props in self.qubit_view.qubits.items()
                if props.measurement_fidelity is not None
            }
            if not fidelities:
                raise PassFailedException(
                    "best_coupling placement needs a measurement fidelity, but no "
                    "qubit in the qubit view has one"
                )
            best_qubit = max(fidelities, key=fidelities.get)
            qubit_ids = (self.qubit_view.qubits[best_qubit].index,)
        elif len(place) == 2:
            if len(self.qubit_view.qubits) == 2:
                if topology_view is not None and topology_view.nnz == 1:
                    qubit_ids = topology_view.qubit_indices_for_edge(0)
                else:
                    # This path will only get past the later coupling validation if
                    # the circuit only has single qubit gates
                    qubit_ids = tuple(
                        self.qubit_view.qubits[qubit_id].index
                        for qubit_id in self.qubit_view.qubits
                    )
            else:
                if topology_view is None:
                    raise PassFailedException(
                        "No coupled qubits are available in the qubit view for"
                        " best coupling placement policy"
                    )
                if not topology_view.sorted_gate_fidelities:
                    raise PassFailedException(
                        "No gate fidelities are available in the qubit view for"
                        " best coupling placement policy"
                    )
                fidelity, (source_id, target_id), gate_name = (
                    topology_view.sorted_gate_fidelities[0]
                )
                # Note - gate type ignored at present
                qubit_ids = (
                    self.qubit_view.qubits[source_id].index,
                    self.qubit_view.qubits[target_id].index,
                )
        else:
            qubit_ids = ()
        return qubit_ids

    def _assign_physical_indices(
        self,
        op: ModuleOp,
        qubit_ids: tuple[int, ...],
        alloc_details: list[AllocDetail],
        topology_view: TopologyView | None,
    ) -> dict[SSAValue, int]:
        """Assign physical qubit indices to the kept allocated qubits and their uses.

        Also check existing physical qubit indices exist
        """
        physical_index_by_value: dict[SSAValue, int] = {}

        def assign(detail: AllocDetail, physical_index: int) -> None:
            if physical_index in physical_index_by_value.values():
                raise PassFailedException(
                    f"Physical qubit index {physical_index} is already assigned"
                )
            physical_index_by_value[detail.alloc_op.result] = physical_index
            for qubit_value in detail.qubit_values:
                physical_index_by_value[qubit_value] = physical_index

        physical = [d for d in alloc_details if not d.is_algorithmic]
        for physical_alloc_detail in physical:
            physical_index = physical_alloc_detail.alloc_op.result.type.index.value.data
            if physical_index not in {
                props.index for props in self.qubit_view.qubits.values()
            }:
                raise PassFailedException(
                    f"Physical qubit index {physical_index} is not present "
                    "in the qubit view"
                )
            assign(physical_alloc_detail, physical_index)

        place = [d for d in alloc_details if d.place]
        for physical_index, detail in zip(qubit_ids, place, strict=True):
            assign(detail, physical_index)

        # validate all two-qubit indices including any pre-existing physical ones
        self._validate_two_qubit_gates_on_coupling(
            op, physical_index_by_value, topology_view
        )
        return physical_index_by_value

    def _validate_two_qubit_gates_on_coupling(
        self,
        op: ModuleOp,
        physical_index_by_value: dict[SSAValue, int],
        topology_view: TopologyView | None,
    ) -> None:
        """Validate that all two-qubit gates are applied to coupled qubits according to the
        topology view.

        Couplings are directed: a gate from ``qubit_0`` to ``qubit_1`` is only accepted if
        the topology view has an edge in that direction, even for symmetric gates such as
        CZ. Data where a physical link is stored in one direction only will reject the gate
        in the other. This is not a problem when both directions are listed, as in DDQ
        data.
        """
        id_by_index = {
            props.index: qubit_id for qubit_id, props in self.qubit_view.qubits.items()
        }
        for gate_op in op.walk():
            if not isinstance(gate_op, TwoQubitGate):
                continue

            source_id = id_by_index[physical_index_by_value[gate_op.qubit_0]]
            target_id = id_by_index[physical_index_by_value[gate_op.qubit_1]]
            if topology_view is None or not topology_view.are_coupled(source_id, target_id):
                raise PassFailedException(
                    f"Two-qubit gate applied to uncoupled qubits: {source_id}->{target_id}"
                )

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Replace each algorithmic qubit allocation with a physical one under the chosen
        policy and remove any unused allocations. Inputs with physical allocations are
        processed without change apart from validating the qubit indices.

        :param ctx: xDSL context (unused).
        :param op: Module to optimize.
        """
        self._reject_qubit_function_signatures(op)
        alloc_ops = [
            gate_op for gate_op in op.walk() if isinstance(gate_op, AllocateQubitOp)
        ]

        alloc_details = self._analyse_allocs(alloc_ops)
        topology_view = (
            TopologyView.derive(self.qubit_view) if self.qubit_view.interactions else None
        )
        qubit_ids = self._validate_and_generate_qubit_ids(alloc_details, topology_view)
        physical_index_by_value = self._assign_physical_indices(
            op, qubit_ids, alloc_details, topology_view
        )

        place = [d for d in alloc_details if d.place]
        dropped = [d for d in alloc_details if d.drop]

        rewriter = Rewriter()
        for detail in place:
            physical_index = physical_index_by_value[detail.alloc_op.result]
            new_alloc_op = AllocatePhysicalQubitOp(physical_index)
            rewriter.replace_op(detail.alloc_op, new_alloc_op)
            new_type = new_alloc_op.result.type
            for qubit_value in detail.qubit_values:
                rewriter.replace_value_with_new_type(qubit_value, new_type)

        for detail in dropped:
            if detail.dealloc_op:
                rewriter.erase_op(detail.dealloc_op)
            rewriter.erase_op(detail.alloc_op)
