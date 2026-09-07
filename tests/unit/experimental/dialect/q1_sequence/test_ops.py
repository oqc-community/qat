# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1 import NopOp, Registers, StopOp
from qat.experimental.dialect.q1_cf import (
    JmpBranchOp,
    UnaryPredicate,
    UnaryPredicateBranchOp,
)
from qat.experimental.dialect.q1_scf import ForOp, YieldOp
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionPathConnectionAttr,
    ConnectionAttr,
    InputConfigAttr,
    LocalOscillatorConfigAttr,
    MixerCorrectionConfigAttr,
    ModuleConfigAttr,
    OutputConfigAttr,
    OutputPathConnectionAttr,
    SequencerConfigAttr,
    make_acquisition,
    make_sequencer_config,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp, find_enclosing_sequence
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
)


def _module_config(
    slot_idx: int = 1,
    kind: QbloxModuleKind = QbloxModuleKind.qrm_rf,
    outputs: list[OutputConfigAttr] | None = None,
    inputs: list[InputConfigAttr] | None = None,
    local_oscillators: list[LocalOscillatorConfigAttr] | None = None,
) -> ModuleConfigAttr:
    return ModuleConfigAttr(
        slot_idx,
        "cluster0",
        kind,
        outputs or [],
        inputs or [],
        local_oscillators or [],
    )


class TestSequenceOpConstruction:
    def test_minimal(self):
        seq = SequenceOp("ch0", [StopOp()])
        assert seq.channel_id.data == "ch0"
        assert "channel_id" in seq.attributes
        assert "sym_name" not in seq.attributes
        assert seq.port_id.data == "ch0"
        assert len(seq.waveforms) == 0
        assert len(seq.weights) == 0
        assert len(seq.acquisitions) == 0
        assert seq.slot_idx is None
        assert seq.seq_idx is None
        assert seq.sequencer_config is None
        assert seq.module_config is None

    def test_with_distinct_port_id(self):
        seq = SequenceOp("ch0", [StopOp()], port_id="Q0/drive")

        assert seq.channel_id.data == "ch0"
        assert seq.port_id.data == "Q0/drive"

    def test_port_id_is_not_inferred_from_sequencer_config(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            sequencer_config=SequencerConfigAttr(port_id="Q0/drive"),
        )

        assert seq.port_id.data == "ch0"

    def test_with_physical_allocation_and_config(self):
        module_config = _module_config()
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=2,
            sequencer_config=make_sequencer_config(integration_length=1024),
            module_config=module_config,
        )
        assert seq.slot_idx.data == 1
        assert seq.seq_idx.data == 2
        assert seq.sequencer_config.integration_length.data == 1024
        assert seq.module_config == module_config

    def test_with_body_ops(self):
        seq = SequenceOp("ch0", [NopOp(), StopOp()])
        ops = list(seq.body.block.ops)
        assert len(ops) == 2

    def test_string_attr_channel_id(self):
        seq = SequenceOp(StringAttr("ch0"), [StopOp()])
        assert seq.channel_id.data == "ch0"

    def test_with_waveforms(self):
        wf = make_waveform("wf0", 0, [0.1, 0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf]))
        assert len(seq.waveforms) == 1
        assert seq.waveforms.data[0].waveform_name.data == "wf0"

    def test_with_weights(self):
        w = make_weight("w0", 0, [1.0, 0.0])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w]))
        assert len(seq.weights) == 1
        assert seq.weights.data[0].weight_name.data == "w0"

    def test_with_acquisitions(self):
        acq = make_acquisition("acq0", 0, 1)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([acq]))
        assert len(seq.acquisitions) == 1
        assert seq.acquisitions.data[0].num_bins.data == 1

    def test_with_all_tables(self):
        wf = make_waveform("wf0", 0, [0.5])
        w = make_weight("w0", 0, [1.0])
        acq = make_acquisition("acq0", 0, 2)
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            waveforms=ArrayAttr([wf]),
            weights=ArrayAttr([w]),
            acquisitions=ArrayAttr([acq]),
        )
        assert len(seq.waveforms) == 1
        assert len(seq.weights) == 1
        assert len(seq.acquisitions) == 1


class TestSequenceOpVerify:
    def test_valid(self):
        seq = SequenceOp("ch0", [StopOp()])
        seq.verify_()

    def test_empty_channel_id_fails(self):
        seq = SequenceOp("", [StopOp()])
        with pytest.raises(VerifyException, match="channel_id must be non-empty"):
            seq.verify_()

    def test_empty_port_id(self):
        seq = SequenceOp("ch0", [StopOp()], port_id="")
        with pytest.raises(VerifyException, match="port_id must be non-empty"):
            seq.verify_()

    def test_partial_physical_allocation_fails(self):
        seq = SequenceOp("ch0", [StopOp()], slot_idx=1)
        with pytest.raises(VerifyException, match="must be set together"):
            seq.verify_()

    def test_duplicate_physical_allocation_fails(self):
        seq0 = SequenceOp(
            "ch0", [StopOp()], instrument_id="cluster0", slot_idx=1, seq_idx=0
        )
        seq1 = SequenceOp(
            "ch1", [StopOp()], instrument_id="cluster0", slot_idx=1, seq_idx=0
        )
        ModuleOp([seq0, seq1])
        with pytest.raises(VerifyException, match="Duplicate physical allocation"):
            seq0.verify_()

    def test_shared_module_requires_consistent_configuration(self):
        seq0 = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            module_config=_module_config(kind=QbloxModuleKind.qrm),
        )
        seq1 = SequenceOp(
            "ch1",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=1,
            module_config=_module_config(kind=QbloxModuleKind.qrm_rf),
        )
        ModuleOp([seq0, seq1])

        with pytest.raises(VerifyException, match="conflicting module configurations"):
            seq0.verify_()

    def test_module_kind_limits_sequencer_index(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=6,
            module_config=_module_config(kind=QbloxModuleKind.qcm),
        )
        with pytest.raises(VerifyException, match="invalid for qcm"):
            seq.verify_()

    def test_acquisition_config_requires_readout_sequencer(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            sequencer_config=make_sequencer_config(integration_length=1024),
            module_config=_module_config(kind=QbloxModuleKind.qcm),
        )
        with pytest.raises(VerifyException, match="acquisition-capable sequencer"):
            seq.verify_()

    def test_acquisition_table_requires_readout_sequencer(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            acquisitions=ArrayAttr([make_acquisition("acq0", 0, 1)]),
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            module_config=_module_config(kind=QbloxModuleKind.qcm),
        )

        with pytest.raises(VerifyException, match="acquisition-capable sequencer"):
            seq.verify_()

    def test_acquisition_enabled_requires_readout_sequencer(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            sequencer_config=SequencerConfigAttr(acquisition_enabled=True),
            module_config=_module_config(kind=QbloxModuleKind.qcm),
        )

        with pytest.raises(VerifyException, match="acquisition-capable sequencer"):
            seq.verify_()

    def test_valid_readout_allocation(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=5,
            sequencer_config=make_sequencer_config(integration_length=1024),
            module_config=_module_config(),
        )
        seq.verify_()

    def test_module_acquisition_tables_respect_memory_limit(self):
        module_config = _module_config(kind=QbloxModuleKind.qrm)
        seq0 = SequenceOp(
            "ch0",
            [StopOp()],
            acquisitions=ArrayAttr([make_acquisition("acq0", 0, 1_500_001)]),
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            module_config=module_config,
        )
        seq1 = SequenceOp(
            "ch1",
            [StopOp()],
            acquisitions=ArrayAttr([make_acquisition("acq1", 0, 1_500_000)]),
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=1,
            module_config=module_config,
        )
        ModuleOp([seq0, seq1])

        with pytest.raises(VerifyException, match="3000000-bin module limit"):
            seq0.verify_()

    def test_module_rejects_unsupported_mixer_correction(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            sequencer_config=SequencerConfigAttr(
                mixer=MixerCorrectionConfigAttr(phase_offset=0.0, gain_ratio=1.0)
            ),
            module_config=_module_config(kind=QbloxModuleKind.qrc),
        )

        with pytest.raises(VerifyException, match="does not support mixer correction"):
            seq.verify_()

    def test_valid_connections_against_module_lanes(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            sequencer_config=SequencerConfigAttr(
                output_path_connections=[OutputPathConnectionAttr(0, SignalPath.iq)],
                acquisition_path_connections=[
                    AcquisitionPathConnectionAttr(0, SignalPath.iq)
                ],
                local_oscillator_id="lo0",
            ),
            module_config=_module_config(
                outputs=[OutputConfigAttr(0)],
                inputs=[InputConfigAttr(0)],
                local_oscillators=[LocalOscillatorConfigAttr("lo0", 6_000_000_000)],
            ),
        )
        seq.verify_()

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param(
                SequencerConfigAttr(
                    output_path_connections=[OutputPathConnectionAttr(0, SignalPath.iq)]
                ),
                "output 0 is absent",
                id="unconfigured-output",
            ),
            pytest.param(
                SequencerConfigAttr(
                    acquisition_path_connections=[
                        AcquisitionPathConnectionAttr(0, SignalPath.iq)
                    ]
                ),
                "input 0 is absent",
                id="unconfigured-input",
            ),
            pytest.param(
                SequencerConfigAttr(local_oscillator_id="lo0"),
                "unknown local oscillator",
                id="unknown-oscillator",
            ),
        ],
    )
    def test_connections_must_exist_in_module_configuration(self, config, expected):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            sequencer_config=config,
            module_config=_module_config(),
        )
        with pytest.raises(VerifyException, match=expected):
            seq.verify_()

    def test_qrc_sequencer_cannot_drive_unreachable_output(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=1,
            sequencer_config=SequencerConfigAttr(
                output_path_connections=[OutputPathConnectionAttr(2, SignalPath.iq)]
            ),
            module_config=_module_config(
                kind=QbloxModuleKind.qrc, outputs=[OutputConfigAttr(2)]
            ),
        )
        with pytest.raises(VerifyException, match="cannot drive output 2"):
            seq.verify_()

    def test_qrc_waveform_only_sequencer_cannot_configure_acquisition_connection(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=8,
            sequencer_config=SequencerConfigAttr(
                output_path_connections=[OutputPathConnectionAttr(2, SignalPath.iq)],
                acquisition_path_connections=[
                    AcquisitionPathConnectionAttr(0, SignalPath.iq)
                ],
            ),
            module_config=_module_config(
                kind=QbloxModuleKind.qrc,
                outputs=[OutputConfigAttr(2)],
                inputs=[InputConfigAttr(0)],
            ),
        )
        with pytest.raises(VerifyException, match="acquisition-capable sequencer"):
            seq.verify_()

    def test_qrc_waveform_only_sequencer_cannot_use_bulk_input_connection(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=8,
            sequencer_config=SequencerConfigAttr(
                connections=[ConnectionAttr(DirectionKind.input, [0])]
            ),
            module_config=_module_config(
                kind=QbloxModuleKind.qrc, inputs=[InputConfigAttr(0)]
            ),
        )
        with pytest.raises(VerifyException, match="acquisition-capable sequencer"):
            seq.verify_()

    def test_qrc_connection_obeys_output_reachability(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=8,
            sequencer_config=SequencerConfigAttr(
                connections=[ConnectionAttr(DirectionKind.output, [0])]
            ),
            module_config=_module_config(
                kind=QbloxModuleKind.qrc, outputs=[OutputConfigAttr(0)]
            ),
        )
        with pytest.raises(VerifyException, match="cannot drive connection output"):
            seq.verify_()

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (DirectionKind.output, "unconfigured outputs"),
            (DirectionKind.input, "unconfigured inputs"),
            (DirectionKind.io, "unconfigured outputs"),
        ],
    )
    def test_connection_requires_configured_module_lanes(self, kind, expected):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            sequencer_config=SequencerConfigAttr(connections=[ConnectionAttr(kind, [0])]),
            module_config=_module_config(kind=QbloxModuleKind.qrm),
        )

        with pytest.raises(VerifyException, match=expected):
            seq.verify_()

    def test_optional_properties_are_omitted_when_absent(self):
        seq = SequenceOp("ch0", [StopOp()])
        assert "instrument_id" not in seq.properties
        assert "slot_idx" not in seq.properties
        assert "seq_idx" not in seq.properties
        assert "sequencer_config" not in seq.properties
        assert "module_config" not in seq.properties

    def test_module_config_must_match_physical_allocation(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
            module_config=_module_config(slot_idx=2),
        )

        with pytest.raises(VerifyException, match="does not match"):
            seq.verify_()

    def test_module_config_requires_physical_allocation(self):
        seq = SequenceOp("ch0", [StopOp()], module_config=_module_config())

        with pytest.raises(VerifyException, match="requires instrument_id"):
            seq.verify_()

    def test_port_id_must_match_sequencer_config_without_module_config(self):
        seq = SequenceOp(
            "ch0",
            [StopOp()],
            port_id="physical-a",
            sequencer_config=SequencerConfigAttr(port_id="physical-b"),
        )

        with pytest.raises(VerifyException, match="port_id conflicts"):
            seq.verify_()

    def test_duplicate_allocation_fails_without_module_config(self):
        first = SequenceOp(
            "ch0",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
        )
        second = SequenceOp(
            "ch1",
            [StopOp()],
            instrument_id="cluster0",
            slot_idx=1,
            seq_idx=0,
        )
        ModuleOp([first, second])

        with pytest.raises(VerifyException, match="Duplicate physical allocation"):
            first.verify_()

    def test_duplicate_waveform_indices_fail(self):
        wf0 = make_waveform("a", 0, [0.1])
        wf1 = make_waveform("b", 0, [0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf0, wf1]))
        with pytest.raises(VerifyException, match="Duplicate index 0 in waveforms"):
            seq.verify_()

    def test_duplicate_weight_indices_fail(self):
        w0 = make_weight("a", 1, [1.0])
        w1 = make_weight("b", 1, [0.5])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w0, w1]))
        with pytest.raises(VerifyException, match="Duplicate index 1 in weights"):
            seq.verify_()

    def test_duplicate_acquisition_indices_fail(self):
        a0 = make_acquisition("a", 2, 1)
        a1 = make_acquisition("b", 2, 1)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([a0, a1]))
        with pytest.raises(
            VerifyException,
            match="Duplicate index 2 in acquisitions",
        ):
            seq.verify_()

    def test_distinct_indices_pass(self):
        wf0 = make_waveform("a", 0, [0.1])
        wf1 = make_waveform("b", 1, [0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf0, wf1]))
        seq.verify_()

    def test_missing_terminator_fails(self):
        seq = SequenceOp("ch0", [NopOp()])
        with pytest.raises(VerifyException, match="must end with a terminator"):
            seq.verify_()

    def test_empty_body_fails(self):
        seq = SequenceOp("ch0", Region([]))
        with pytest.raises(VerifyException, match="must contain at least one block"):
            seq.verify_()

    def test_duplicate_waveform_names_fail(self):
        wf0 = make_waveform("same", 0, [0.1])
        wf1 = make_waveform("same", 1, [0.2])
        seq = SequenceOp("ch0", [StopOp()], waveforms=ArrayAttr([wf0, wf1]))
        with pytest.raises(VerifyException, match="Duplicate name 'same' in waveforms"):
            seq.verify_()

    def test_duplicate_weight_names_fail(self):
        w0 = make_weight("dup", 0, [1.0])
        w1 = make_weight("dup", 1, [0.5])
        seq = SequenceOp("ch0", [StopOp()], weights=ArrayAttr([w0, w1]))
        with pytest.raises(VerifyException, match="Duplicate name 'dup' in weights"):
            seq.verify_()

    def test_duplicate_acquisition_names_fail(self):
        a0 = make_acquisition("acq", 0, 1)
        a1 = make_acquisition("acq", 1, 2)
        seq = SequenceOp("ch0", [StopOp()], acquisitions=ArrayAttr([a0, a1]))
        with pytest.raises(
            VerifyException,
            match="Duplicate name 'acq' in acquisitions",
        ):
            seq.verify_()


class TestSequenceOpMultiBlock:
    """Verify that SequenceOp accepts multi-block regions (q1_cf CFG form)."""

    def test_two_block_with_jmp_terminator(self):
        """A two-block SequenceOp where the entry block ends with q1_cf.jmp_branch."""

        exit_block = Block([StopOp()])
        entry_block = Block([NopOp(), JmpBranchOp([], exit_block)])
        region = Region([entry_block, exit_block])
        seq = SequenceOp("ch0", region)
        seq.verify_()  # must not raise

    def test_two_block_cond_branch(self):
        """A two-block SequenceOp with q1_cf.unary_predicate_branch (else is fall-
        through)."""

        else_block = Block([StopOp()])
        then_block = Block([StopOp()])
        # rs must be a block argument to satisfy IsolatedFromAbove
        entry_block = Block(arg_types=[Registers.R0])
        rs = entry_block.args[0]
        entry_block.add_op(
            UnaryPredicateBranchOp(UnaryPredicate.eqz, rs, [], [], then_block, else_block)
        )
        # entry -> else -> then (else must immediately follow the branch's block
        # to satisfy the q1_cf fall-through invariant)
        region = Region([entry_block, else_block, then_block])
        seq = SequenceOp("ch0", region)
        seq.verify()  # must not raise (verifies nested q1_cf ops too)

    def test_block_without_terminator_fails(self):
        """A block ending in a non-terminator op must be rejected."""

        exit_block = Block([StopOp()])
        entry_block = Block([NopOp()])  # missing terminator
        region = Region([entry_block, exit_block])
        seq = SequenceOp("ch0", region)

        with pytest.raises(VerifyException, match="must end with a terminator"):
            seq.verify_()


class TestFindEnclosingSequence:
    def test_direct_parent(self):
        """Op directly inside a SequenceOp body is found immediately."""
        nop = NopOp()
        seq = SequenceOp("ch0", [nop, StopOp()])
        assert find_enclosing_sequence(nop) is seq

    def test_nested_inside_for_op(self):
        """Op nested inside a ForOp body that is itself inside a SequenceOp is found by
        walking past the ForOp."""
        _reg = Registers.UNALLOCATED_INT

        inner_nop = NopOp()
        for_body = Block(arg_types=[_reg])
        for_body.add_op(inner_nop)
        for_body.add_op(YieldOp())

        entry = Block(arg_types=[_reg])
        (count,) = entry.args
        for_op = ForOp(count, [], Region([for_body]))
        entry.add_ops([for_op, StopOp()])
        seq = SequenceOp("ch0", Region([entry]))

        assert find_enclosing_sequence(inner_nop) is seq

    def test_no_ancestor_raises(self):
        """A standalone op with no SequenceOp ancestor raises ValueError."""
        nop = NopOp()
        with pytest.raises(ValueError, match="No SequenceOp found in the parent chain"):
            find_enclosing_sequence(nop)
