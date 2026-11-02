# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects import func, scf
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.gate.ir import (
    AlgorithmicQubitType,
    AllocateAlgorithmicQubitOp,
    AllocatePhysicalQubitOp,
    AllocateQubitOp,
    BarrierOp,
    CXOp,
    DeallocateQubitOp,
    Gate,
    HOp,
    MeasureOp,
    PhysicalQubitType,
)
from qat.experimental.dialect.gate.transforms.naive_placement import (
    NaiveQubitPlacementPass,
    PlacementPolicy,
)
from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    ProbabilityEntry,
    QubitCouplingData,
    QubitData,
    ReadoutProbabilityData,
    TwoQubitGateFidelityData,
)
from qat.experimental.system_data.circuit.qubits import QubitView

from tests.unit.utils.ir import create_context

_CONTEXT = create_context(Gate)


class TestNaiveQubitPlacementPass:
    @pytest.mark.parametrize(
        ("policy", "explicit_qubit_ids", "expected_indices"),
        [
            (PlacementPolicy.as_written, None, (0, 1)),
            (PlacementPolicy.explicit, ("q0", "q1"), (0, 1)),
            (PlacementPolicy.best_coupling, None, (0, 1)),
        ],
    )
    def test_pass_places_qubits_when_they_are_all_used(
        self,
        policy: PlacementPolicy,
        explicit_qubit_ids: tuple[str, ...] | None,
        expected_indices: tuple[int, ...],
    ):
        # Test the qubit placement pass
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        cx = CXOp(h.result, q1.result)
        m0 = MeasureOp(cx.result_qubit_control)
        free = DeallocateQubitOp(cx.result_qubit_target)

        module = ModuleOp([q0, q1, h, cx, m0, free])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view, policy=policy, explicit_qubit_ids=explicit_qubit_ids
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"
        allocs = [
            gate_op for gate_op in gate_ops if isinstance(gate_op, AllocatePhysicalQubitOp)
        ]
        assert [alloc.result.type.index.value.data for alloc in allocs] == list(
            expected_indices
        )

        control_index, target_index = expected_indices
        assert h.result.type == PhysicalQubitType(control_index)
        assert cx.result_qubit_control.type == PhysicalQubitType(control_index)
        assert cx.result_qubit_target.type == PhysicalQubitType(target_index)

    @pytest.mark.parametrize(
        ("policy", "explicit_qubit_ids"),
        [
            (PlacementPolicy.as_written, None),
            (PlacementPolicy.explicit, ("q0",)),
            (PlacementPolicy.best_coupling, None),
        ],
    )
    def test_qubit_allocation_without_uses_is_removed(
        self, policy: PlacementPolicy, explicit_qubit_ids: tuple[str, ...] | None
    ):
        # Test the qubit placement pass with a qubit that has no uses
        q0 = AllocateAlgorithmicQubitOp()
        module = ModuleOp([q0])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view, policy=policy, explicit_qubit_ids=explicit_qubit_ids
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert len(gate_ops) == 0

    @pytest.mark.parametrize(
        ("policy", "explicit_qubit_ids"),
        [
            (PlacementPolicy.as_written, None),
            (PlacementPolicy.explicit, ("q0",)),
            (PlacementPolicy.best_coupling, None),
        ],
    )
    def test_qubit_allocation_then_deallocation_are_both_removed(
        self, policy: PlacementPolicy, explicit_qubit_ids: tuple[str, ...] | None
    ):
        # Test the qubit placement pass with a qubit that is allocated and then deallocated
        q0 = AllocateAlgorithmicQubitOp()
        free = DeallocateQubitOp(q0.result)
        module = ModuleOp([q0, free])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view, policy=policy, explicit_qubit_ids=explicit_qubit_ids
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert len(gate_ops) == 0

    @pytest.mark.parametrize(
        ("policy", "explicit_qubit_ids"),
        [
            (PlacementPolicy.as_written, None),
            (PlacementPolicy.explicit, ("q0",)),
            (PlacementPolicy.best_coupling, None),
        ],
    )
    def test_qubit_allocated_then_measured_is_placed(
        self, policy: PlacementPolicy, explicit_qubit_ids: tuple[str, ...] | None
    ):
        # Test the qubit placement pass with a qubit that is allocated and then measured
        q0 = AllocateAlgorithmicQubitOp()
        m0 = MeasureOp(q0.result)
        module = ModuleOp([q0, m0])
        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0, readout_probability=self._confusion(0.95)),)
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view, policy=policy, explicit_qubit_ids=explicit_qubit_ids
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert len(gate_ops) == 2
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been removed"
        allocs = [
            gate_op for gate_op in gate_ops if isinstance(gate_op, AllocatePhysicalQubitOp)
        ]
        assert [alloc.result.type.index.value.data for alloc in allocs] == [0]

    @pytest.mark.parametrize(
        ("policy", "explicit_qubit_ids"),
        [
            (PlacementPolicy.as_written, None),
            (PlacementPolicy.explicit, ("q0", "q1")),
        ],
    )
    def test_two_qubit_gate_on_uncoupled_pair_fails(
        self, policy: PlacementPolicy, explicit_qubit_ids: tuple[str, ...] | None
    ):
        # Test that the qubit placement pass raises an error if a two-qubit gate is applied
        # to a pair of qubits that are not coupled in the canonical system data
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        cx = CXOp(q0.result, q1.result)
        module = ModuleOp([q0, q1, cx])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            ),
            couplings=(),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view, policy=policy, explicit_qubit_ids=explicit_qubit_ids
        )
        with pytest.raises(
            PassFailedException,
            match="Two-qubit gate applied to uncoupled qubits",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_as_written_counts_unused_allocation_before_used_one(self):
        # Unused allocations still count as declared, so the second allocation is placed
        # on index 1 even though the first is removed
        unused = AllocateAlgorithmicQubitOp()
        used = AllocateAlgorithmicQubitOp()
        m = MeasureOp(used.result)
        module = ModuleOp([unused, used, m])
        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1))
        )
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert len(gate_ops) == 2
        assert isinstance(gate_ops[0], AllocatePhysicalQubitOp)
        assert gate_ops[0].result.type.index.value.data == 1

    def test_as_written_fails_when_requested_index_not_in_view(self):
        # Two used allocations need indices 0 and 1, but the view only has index 0
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        m0 = MeasureOp(q0.result)
        m1 = MeasureOp(q1.result)
        module = ModuleOp([q0, q1, m0, m1])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )

        with pytest.raises(
            PassFailedException, match="Qubit index 1 is not present in the qubit view"
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_explicit_policy_without_explicit_qubit_ids_fails(self):
        # Test that the explicit policy raises an error if explicit_qubit_ids is missing
        canonical = CanonicalSystemData()
        view = QubitView.derive(canonical)
        with pytest.raises(
            ValueError, match="the explicit policy needs explicit_qubit_ids"
        ):
            NaiveQubitPlacementPass(qubit_view=view, policy=PlacementPolicy.explicit)

    @pytest.mark.parametrize(
        ("explicit_qubit_ids", "message"),
        [
            (("q0", "q9"), "explicit qubit id q9 is not present in the qubit view"),
            (("q0", "q1", "q0"), r"must be unique, but got \('q0', 'q1', 'q0'\)"),
        ],
    )
    def test_explicit_policy_rejects_unknown_or_repeated_qubit_ids(
        self, explicit_qubit_ids: tuple[str, ...], message: str
    ):
        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1))
        )
        with pytest.raises(ValueError, match=message):
            NaiveQubitPlacementPass(
                qubit_view=QubitView.derive(canonical),
                policy=PlacementPolicy.explicit,
                explicit_qubit_ids=explicit_qubit_ids,
            )

    def test_explicit_policy_can_be_applied_twice(self):
        # Once placed, no algorithmic declarations remain, so reapplying must not fail
        # on the explicit id count
        q0 = AllocateAlgorithmicQubitOp()
        m0 = MeasureOp(q0.result)
        module = ModuleOp([q0, m0])
        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=3),),
        )
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical),
            policy=PlacementPolicy.explicit,
            explicit_qubit_ids=("q0",),
        )

        pass_instance.apply(_CONTEXT, module)
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        allocs = [
            op for op in module.body.block.ops if isinstance(op, AllocatePhysicalQubitOp)
        ]
        assert [alloc.result.type.index.value.data for alloc in allocs] == [3]

    def test_explicit_policy_with_explicit_qubit_ids_places_qubits(self):
        # Test that the explicit policy works with explicit_qubit_ids
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        cx = CXOp(h.result, q1.result)
        m0 = MeasureOp(cx.result_qubit_control)
        free = DeallocateQubitOp(cx.result_qubit_target)

        module = ModuleOp([q0, q1, h, cx, m0, free])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=1),
                QubitData(id="q1", index=0),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.explicit,
            explicit_qubit_ids=("q0", "q1"),
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"
        allocs = [
            gate_op for gate_op in gate_ops if isinstance(gate_op, AllocatePhysicalQubitOp)
        ]
        assert [alloc.result.type.index.value.data for alloc in allocs] == [1, 0]
        assert h.result.type == PhysicalQubitType(1)
        assert cx.result_qubit_control.type == PhysicalQubitType(1)
        assert cx.result_qubit_target.type == PhysicalQubitType(0)

    def test_explicit_policy_maps_ids_by_declaration_position_when_earlier_one_unused(self):
        # Test that the explicit policy validates against the number of kept allocations
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q1.result)
        m0 = MeasureOp(h.result)
        free = DeallocateQubitOp(q0.result)

        module = ModuleOp([q0, q1, h, m0, free])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            )
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.explicit,
            explicit_qubit_ids=("q1", "q0"),
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"
        allocs = [
            gate_op for gate_op in gate_ops if isinstance(gate_op, AllocatePhysicalQubitOp)
        ]
        assert [alloc.result.type.index.value.data for alloc in allocs] == [0]

    def test_explicit_policy_rejects_list_that_omits_unused_declaration(self):
        # Test that the explicit policy validates against the number of kept allocations
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        m0 = MeasureOp(h.result)
        free = DeallocateQubitOp(q1.result)

        module = ModuleOp([q0, q1, h, m0, free])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            )
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.explicit,
            explicit_qubit_ids=("q0",),
        )
        with pytest.raises(
            PassFailedException,
            match=r"Number of explicit qubit ids \(1\) does not match .*\(2\)",
        ):
            pass_instance.apply(_CONTEXT, module)

    def _confusion(self, fidelity: float) -> ReadoutProbabilityData:
        return ReadoutProbabilityData(
            probability_entries=tuple(
                ProbabilityEntry(
                    prepared_state=p,
                    measured_state=m,
                    probability=fidelity if p == m else 1 - fidelity,
                )
                for p in (0, 1)
                for m in (0, 1)
            )
        )

    def test_best_coupling_policy_with_single_allocated_qubit(self):
        # Test that the best coupling policy works with a single allocated qubit
        q0 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        m0 = MeasureOp(h.result)

        module = ModuleOp([q0, h, m0])
        confusion_q0 = self._confusion(0.95)
        confusion_q1 = self._confusion(0.90)

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0, readout_probability=confusion_q0),
                QubitData(id="q1", index=1, readout_probability=confusion_q1),
            )
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"
        alloc_op = gate_ops[0]
        assert isinstance(alloc_op, AllocatePhysicalQubitOp)
        assert alloc_op.result.type.index.value.data == 0

    def test_best_coupling_policy_with_two_allocated_qubits(self):
        # Test that the best coupling policy works with two allocated qubits
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        cx = CXOp(h.result, q1.result)
        m0 = MeasureOp(cx.result_qubit_control)
        free = DeallocateQubitOp(cx.result_qubit_target)

        module = ModuleOp([q0, q1, h, cx, m0, free])

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
                QubitData(id="q2", index=2),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
                QubitCouplingData(
                    source_qubit_id="q1",
                    target_qubit_id="q2",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.95),),
                ),
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q2",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"
        assert isinstance(gate_ops[0], AllocatePhysicalQubitOp)
        assert gate_ops[0].result.type.index.value.data == 0
        assert isinstance(gate_ops[1], AllocatePhysicalQubitOp)
        assert gate_ops[1].result.type.index.value.data == 2

    def test_best_coupling_policy_fails_for_unsupported_gate_direction(self):
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        cx = CXOp(q1.result, q0.result)
        free0 = DeallocateQubitOp(cx.result_qubit_control)
        free1 = DeallocateQubitOp(cx.result_qubit_target)
        module = ModuleOp([q0, q1, cx, free0, free1])

        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1)),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
            ),
        )
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical),
            policy=PlacementPolicy.best_coupling,
        )
        with pytest.raises(PassFailedException, match="uncoupled qubits"):
            pass_instance.apply(_CONTEXT, module)

    def test_best_coupling_policy_with_more_than_two_allocated_qubits_fails(self):
        # Test that the best coupling policy raises an error with >2 allocated qubits
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        q2 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        cx1 = CXOp(h.result, q1.result)
        cx2 = CXOp(cx1.result_qubit_control, q2.result)
        m0 = MeasureOp(cx2.result_qubit_control)
        free1 = DeallocateQubitOp(cx1.result_qubit_target)  # q1
        free2 = DeallocateQubitOp(cx2.result_qubit_target)  # q2

        module = ModuleOp([q0, q1, q2, h, cx1, cx2, m0, free1, free2])

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
                QubitData(id="q2", index=2),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
                QubitCouplingData(
                    source_qubit_id="q1",
                    target_qubit_id="q2",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.95),),
                ),
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q2",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.99),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )
        with pytest.raises(
            NotImplementedError,
            match="not implemented for more than 2 qubits - 3 requested",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_best_coupling_with_zero_kept_qubits(self):
        # Test that the best coupling policy works with 0 kept qubits
        module = ModuleOp([])
        canonical = CanonicalSystemData(
            qubits=(),
            couplings=(),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )
        pass_instance.apply(_CONTEXT, module)

        gate_ops = list(module.body.block.ops)
        module.verify()
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been removed"

    def test_best_coupling_places_on_both_qubits_when_view_has_two_and_no_couplings(self):
        # Test that the best coupling policy works with only two qubits in the view and
        # no couplings
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h0 = HOp(q0.result)
        h1 = HOp(q1.result)
        m0 = MeasureOp(h0.result)
        m1 = MeasureOp(h1.result)

        module = ModuleOp([q0, q1, h0, h1, m0, m1])

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            ),
            couplings=(),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )

        pass_instance.apply(_CONTEXT, module)

        module.verify()
        assert h0.result.type == PhysicalQubitType(0)
        assert h1.result.type == PhysicalQubitType(1)

    def test_best_coupling_places_cx_on_two_qubit_view_without_fidelity_data(self):
        # With only two qubits in the view, the CX is placed without a fidelity search,
        # even though the coupling carries no fidelity data
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        cx = CXOp(q0.result, q1.result)
        m0 = MeasureOp(cx.result_qubit_control)
        m1 = MeasureOp(cx.result_qubit_target)

        module = ModuleOp([q0, q1, cx, m0, m1])

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )

        pass_instance.apply(_CONTEXT, module)

        module.verify()
        assert cx.result_qubit_control.type == PhysicalQubitType(0)
        assert cx.result_qubit_target.type == PhysicalQubitType(1)

    def test_best_coupling_places_cx_using_direction_of_sole_coupling(self):
        # The view is ordered q0, q1 but the only coupling is q1 -> q0 with no fidelity
        # data, so the coupling's source must take the first allocated qubit
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        cx = CXOp(q0.result, q1.result)
        m0 = MeasureOp(cx.result_qubit_control)
        m1 = MeasureOp(cx.result_qubit_target)

        module = ModuleOp([q0, q1, cx, m0, m1])

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q1",
                    target_qubit_id="q0",
                    gate_fidelities=(),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )

        pass_instance.apply(_CONTEXT, module)

        module.verify()
        assert cx.result_qubit_control.type == PhysicalQubitType(1)
        assert cx.result_qubit_target.type == PhysicalQubitType(0)

    def test_best_coupling_places_on_both_qubits_when_view_has_three_and_no_couplings_fails(
        self,
    ):
        # Test that the best coupling policy fails with no couplings and more than 2 qubits
        # in the view
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        h0 = HOp(q0.result)
        h1 = HOp(q1.result)
        m0 = MeasureOp(h0.result)
        m1 = MeasureOp(h1.result)

        module = ModuleOp([q0, q1, h0, h1, m0, m1])

        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
                QubitData(id="q2", index=2),
            ),
            couplings=(),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )

        with pytest.raises(
            PassFailedException,
            match="No coupled qubits are available in the qubit view",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_best_coupling_single_qubit_and_no_fidelities_fails(self):
        # Test that the best coupling policy works with a single allocated qubit and
        # no fidelities
        q0 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        m0 = MeasureOp(h.result)

        module = ModuleOp([q0, h, m0])

        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.best_coupling,
        )
        with pytest.raises(
            PassFailedException,
            match="best_coupling placement needs a measurement fidelity",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_best_coupling_two_qubits_and_no_gate_fidelities_fails(self):
        # With more than two qubits in the view there is no shortcut, so a coupling
        # without fidelity data cannot be ranked
        q0 = AllocateAlgorithmicQubitOp()
        q1 = AllocateAlgorithmicQubitOp()
        cx = CXOp(q0.result, q1.result)
        module = ModuleOp([q0, q1, cx])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
                QubitData(id="q2", index=2),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0", target_qubit_id="q1", gate_fidelities=()
                ),
            ),
        )
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical),
            policy=PlacementPolicy.best_coupling,
        )

        with pytest.raises(
            PassFailedException, match="No gate fidelities are available in the qubit view"
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_qubit_passed_to_function_call_fails(self):
        # Retyping the call's values would leave the callee's signature out of date
        qubit_type = AlgorithmicQubitType()
        body = Block(arg_types=[qubit_type])
        body.add_op(func.ReturnOp(body.args[0]))
        callee = func.FuncOp("f", ((qubit_type,), (qubit_type,)), Region(body))
        q0 = AllocateAlgorithmicQubitOp()
        call = func.CallOp("f", [q0.result], [qubit_type])
        m = MeasureOp(call.results[0])
        module = ModuleOp([callee, q0, call, m])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )

        with pytest.raises(PassFailedException, match="function signatures"):
            pass_instance.apply(_CONTEXT, module)

    @pytest.mark.parametrize("qubit_type", [AlgorithmicQubitType(), PhysicalQubitType(0)])
    def test_external_function_returning_qubit_fails(self, qubit_type):
        # There is no allocation for the returned qubit, so it cannot be placed
        declaration = func.FuncOp.external("f", (), (qubit_type,))
        call = func.CallOp("f", [], [qubit_type])
        module = ModuleOp([declaration, call, MeasureOp(call.results[0])])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )

        with pytest.raises(PassFailedException, match="function signatures"):
            pass_instance.apply(_CONTEXT, module)

    def test_function_declaration_without_qubits_is_accepted(self):
        declaration = func.FuncOp.external("f", (), ())
        q0 = AllocateAlgorithmicQubitOp()
        module = ModuleOp([declaration, q0, MeasureOp(q0.result)])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )

        pass_instance.apply(_CONTEXT, module)

    @pytest.mark.parametrize(
        "qubit_type", [AlgorithmicQubitType(), PhysicalQubitType(0)], ids=["alg", "phys"]
    )
    def test_qubit_block_argument_with_local_allocation_fails(self, qubit_type):
        # The caller places a qubit argument, so it cannot be tracked or checked for
        # clashes with local allocations
        body = Block(arg_types=[qubit_type])
        local = AllocateAlgorithmicQubitOp()
        body.add_ops(
            [
                local,
                BarrierOp([body.args[0], local.result]),
                func.ReturnOp(),
            ]
        )
        module = ModuleOp([func.FuncOp("f", ((qubit_type,), ()), Region(body))])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )

        with pytest.raises(PassFailedException, match="function signatures"):
            pass_instance.apply(_CONTEXT, module)

    def test_qubit_used_more_than_once_fails(self):
        # Both ops consume the same qubit value
        q0 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        m = MeasureOp(q0.result)
        module = ModuleOp([q0, h, m])
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical), policy=PlacementPolicy.as_written
        )

        with pytest.raises(PassFailedException, match="only supports single-use qubits"):
            pass_instance.apply(_CONTEXT, module)

    @pytest.mark.parametrize(
        "policy", [PlacementPolicy.as_written, PlacementPolicy.best_coupling]
    )
    def test_explicit_qubit_ids_rejected_for_non_explicit_policy(
        self, policy: PlacementPolicy
    ):
        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        with pytest.raises(ValueError, match="only used by the explicit policy"):
            NaiveQubitPlacementPass(
                qubit_view=QubitView.derive(canonical),
                policy=policy,
                explicit_qubit_ids=("q0",),
            )

    def test_pass_works_on_circuit_with_scf_for_loop(self):
        # Test that the pass works on a circuit with an scf.for loop
        q0 = AllocateAlgorithmicQubitOp()
        h = HOp(q0.result)
        m0 = MeasureOp(h.result)

        repeat = 100
        start_op = ConstantOp.from_int_and_width(0, IndexType())
        stop_op = ConstantOp.from_int_and_width(repeat, IndexType())
        step_op = ConstantOp.from_int_and_width(1, IndexType())
        loop = scf.ForOp(
            start_op,
            stop_op,
            step_op,
            [],
            Block(ops=[q0, h, m0, scf.YieldOp()], arg_types=[IndexType()]),
        )

        module = ModuleOp([start_op, stop_op, step_op, loop])

        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.walk())
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"

    def test_pass_raises_on_circuit_with_scf_for_loop_nested_beneath_qubit_allocation(self):
        # Test that the pass raises an error on a circuit with an scf.for loop nested
        # beneath a qubit allocation
        q0 = AllocateAlgorithmicQubitOp()
        body = Block(arg_types=[IndexType(), AlgorithmicQubitType()])
        h = HOp(body.args[1])
        body.add_ops([h, scf.YieldOp(h.result)])

        repeat = 100
        start_op = ConstantOp.from_int_and_width(0, IndexType())
        stop_op = ConstantOp.from_int_and_width(repeat, IndexType())
        step_op = ConstantOp.from_int_and_width(1, IndexType())
        loop = scf.ForOp(start_op, stop_op, step_op, [q0.result], body)

        module = ModuleOp([q0, start_op, stop_op, step_op, loop])
        module.verify()

        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        with pytest.raises(
            PassFailedException,
            match="Qubit placement does not support qubits passed through scf.for",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_input_containing_physical_qubits_is_accepted(self):
        # Test that the pass accepts input containing physical qubits
        q0 = AllocatePhysicalQubitOp(0)
        h = HOp(q0.result)
        m0 = MeasureOp(h.result)

        module = ModuleOp([q0, h, m0])

        canonical = CanonicalSystemData(qubits=(QubitData(id="q0", index=0),))
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()

    def test_input_containing_physical_qubits_without_coupling_fails(self):
        # Test that the pass raises an error if the input contains physical qubits without
        # coupling
        q0 = AllocatePhysicalQubitOp(0)
        q1 = AllocatePhysicalQubitOp(1)
        cx = CXOp(q0.result, q1.result)

        module = ModuleOp([q0, q1, cx])

        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1)),
            couplings=(),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        with pytest.raises(
            PassFailedException,
            match="Two-qubit gate applied to uncoupled qubits",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_input_containing_physical_qubit_index_not_in_view_fails(self):
        # Test that the pass raises an error if the input contains physical qubits with
        # indices not in the view
        q0 = AllocatePhysicalQubitOp(0)
        q1 = AllocatePhysicalQubitOp(1)
        cx = CXOp(q0.result, q1.result)

        module = ModuleOp([q0, q1, cx])

        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=2), QubitData(id="q1", index=3)),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        with pytest.raises(
            PassFailedException,
            match="Physical qubit index 0 is not present in the qubit view",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_qubit_placement_clash_with_pre_existing_physical_qubit_fails(self):
        # Test that the pass raises an error if the placement of algorithmic qubits
        # clashes with pre-existing physical qubits
        q0 = AllocatePhysicalQubitOp(0)
        q1 = AllocateAlgorithmicQubitOp()
        h = HOp(q1.result)
        m0 = MeasureOp(h.result)

        module = ModuleOp([q0, q1, h, m0])

        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1)),
            couplings=(),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        with pytest.raises(
            PassFailedException,
            match="Physical qubit index 0 is already assigned",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_best_coupling_rejects_mixed_physical_and_algorithmic_qubits(self):
        q0 = AllocatePhysicalQubitOp(0)
        q1 = AllocateAlgorithmicQubitOp()
        m0 = MeasureOp(q0.result)
        m1 = MeasureOp(q1.result)
        module = ModuleOp([q0, q1, m0, m1])
        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1)),
        )
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical),
            policy=PlacementPolicy.best_coupling,
        )

        with pytest.raises(PassFailedException, match="mix physical and algorithmic"):
            pass_instance.apply(_CONTEXT, module)

    def test_two_physical_qubits_with_same_index_fails(self):
        # Test that the pass raises an error if the input contains two physical qubits with
        # the same index
        q0 = AllocatePhysicalQubitOp(0)
        q1 = AllocatePhysicalQubitOp(0)
        cx = CXOp(q0.result, q1.result)

        module = ModuleOp([q0, q1, cx])

        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1)),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view,
            policy=PlacementPolicy.as_written,
        )
        with pytest.raises(
            PassFailedException,
            match="Physical qubit index 0 is already assigned",
        ):
            pass_instance.apply(_CONTEXT, module)

    def test_pass_places_qubits_when_they_use_generic_allocate_op(self):
        # Test the qubit placement pass
        q0 = AllocateQubitOp(AlgorithmicQubitType())
        h = HOp(q0.result)
        m0 = MeasureOp(h.result)

        module = ModuleOp([q0, h, m0])
        canonical = CanonicalSystemData(
            qubits=(
                QubitData(id="q0", index=0),
                QubitData(id="q1", index=1),
            ),
            couplings=(
                QubitCouplingData(
                    source_qubit_id="q0",
                    target_qubit_id="q1",
                    gate_fidelities=(TwoQubitGateFidelityData(gate="cx", fidelity=0.97),),
                ),
            ),
        )
        view = QubitView.derive(canonical)
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=view, policy=PlacementPolicy.as_written
        )
        pass_instance.apply(_CONTEXT, module)

        module.verify()
        gate_ops = list(module.body.block.ops)
        assert not any(
            isinstance(result.type, AlgorithmicQubitType)
            for gate_op in gate_ops
            for result in gate_op.results
        ), "AlgorithmicQubitType should have been replaced"

    def test_pass_raises_on_qubit_yielded_from_scf_if(self):
        # Test that the pass rejects an algorithmic qubit allocated in an scf.if branch and
        # yielded out as the if's result, as the result type could not be updated
        cond = ConstantOp.from_int_and_width(1, 1)

        then_q = AllocateAlgorithmicQubitOp()
        then_h = HOp(then_q.result)
        then_block = Block(ops=[then_q, then_h, scf.YieldOp(then_h.result)])

        else_q = AllocateAlgorithmicQubitOp()
        else_block = Block(ops=[else_q, scf.YieldOp(else_q.result)])

        if_op = scf.IfOp(
            cond,
            [AlgorithmicQubitType()],
            Region(then_block),
            Region(else_block),
        )
        module = ModuleOp([cond, if_op])
        module.verify()

        canonical = CanonicalSystemData(
            qubits=(QubitData(id="q0", index=0), QubitData(id="q1", index=1))
        )
        pass_instance = NaiveQubitPlacementPass(
            qubit_view=QubitView.derive(canonical),
            policy=PlacementPolicy.as_written,
        )
        with pytest.raises(PassFailedException, match="consumed"):
            pass_instance.apply(_CONTEXT, module)
