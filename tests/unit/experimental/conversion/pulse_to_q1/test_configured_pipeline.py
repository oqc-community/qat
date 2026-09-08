# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.context import Context
from xdsl.dialects import func, scf
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    ModuleOp,
    StringAttr,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.experimental.backend.qblox.pre_emission_verification import (
    QbloxPreEmissionVerificationPass,
)
from qat.experimental.conversion.pulse_to_q1.passes import (
    Q1PulseLegalisationPass,
    Q1PulseValidationPass,
    create_qblox_configured_q1_pipeline,
)
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    DurationImm,
    IntRegisterType,
    MoveImmRdOp,
    PlayImmImmImmOp,
    Q1RegisterType,
    Registers,
    StopOp,
    SU32Imm,
    UI5Imm,
    UI6Imm,
    UI10Imm,
    UI24Imm,
    WaitImmOp,
)
from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1_cf import JmpBranchOp
from qat.experimental.dialect.q1_cf.transforms.linearise_q1_cf import LineariseQ1CfToQ1Pass
from qat.experimental.dialect.q1_scf import YieldOp as Q1ScfYieldOp
from qat.experimental.dialect.q1_scf.transforms.lower_scf import LowerScfToQ1ScfPass
from qat.experimental.dialect.q1_scf.transforms.lower_to_cf import LowerQ1ScfToQ1CfPass
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionPathConnectionAttr,
    InputConfigAttr,
    ModuleConfigAttr,
    SequencerConfigAttr,
    UnweightedAcquireConfigAttr,
    make_acquisition,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPassPipeline
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind, SignalPath
from qat.experimental.system_data.hardware.qblox.target import DEFAULT_QBLOX_TARGET

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)


def _canonical_data() -> CanonicalSystemData:
    return canonical_data(
        configurations=[supplied([sequencer(0)])],
        carrier_frequency=4_800_000_000,
        oscillator_frequency=4_600_000_000,
        instrument_id="cluster0",
    )


def _readout_canonical_data() -> CanonicalSystemData:
    return canonical_data(
        kind=QbloxModuleKind.qrm_rf,
        configurations=[supplied([sequencer(0, inputs=[0])])],
        slot=3,
        instrument_id="cluster0",
        carrier_frequency=6_200_000_000,
        oscillator_frequency=6_000_000_000,
    )


def _pulse_module(*body_ops) -> ModuleOp:
    return ModuleOp(
        [func.FuncOp("main", ((), ()), Region(Block([*body_ops, func.ReturnOp()])))]
    )


def _frame() -> tuple[ConstantOp, CreateFrameOp]:
    frequency = ConstantOp(FrequencyAttr(4_800_000_000))
    return frequency, CreateFrameOp(frequency, StringAttr("port-0"))


def _configured_pipeline(canonical: CanonicalSystemData):
    return create_qblox_configured_q1_pipeline(canonical)


def _assert_emission_ready(module: ModuleOp) -> SequenceOp:
    [sequence] = list(module.body.block.ops)
    assert isinstance(sequence, SequenceOp)
    assert sequence.instrument_id is not None
    assert len(sequence.body.blocks) == 1
    for operation in sequence.body.block.ops:
        assert isinstance(operation, Q1AsmOperation)
        for value in (*operation.operands, *operation.results):
            if isinstance(value.type, Q1RegisterType):
                assert value.type.is_allocated
    return sequence


def test_downstream_ordering_constraints_reject_misordered_lowering():
    with pytest.raises(VerifyException, match="q1-pulse-validation"):
        OrderedPassPipeline((Q1PulseLegalisationPass(), Q1PulseValidationPass()))

    with pytest.raises(VerifyException, match="lower-scf-to-q1-scf"):
        OrderedPassPipeline((LowerQ1ScfToQ1CfPass(), LowerScfToQ1ScfPass()))

    with pytest.raises(VerifyException, match="lower-q1-scf-to-q1-cf"):
        OrderedPassPipeline((LineariseQ1CfToQ1Pass(), LowerQ1ScfToQ1CfPass()))


def test_pipeline_lowers_normalized_pulse_to_configured_flat_q1():
    frequency, frame = _frame()
    duration = ConstantOp(TimeAttr(8e-9))
    wait = WaitOp(frame, duration)
    module = _pulse_module(frequency, frame, duration, wait)

    _configured_pipeline(_canonical_data()).apply(Context(), module)

    sequence = _assert_emission_ready(module)
    assert sequence.channel_id.data == "port_0"
    assert not any(isinstance(op, CreateFrameOp) for op in sequence.walk())


def test_pipeline_preserves_timing_instruction_through_control_flow_lowering():
    frequency, frame = _frame()
    amplitude = ConstantOp(AmplitudeAttr(0.25))
    duration = ConstantOp(TimeAttr(8e-9))
    start = StartContinuousWaveformOp(frame, amplitude)
    wait = WaitOp(start, duration)
    stop = StopContinuousWaveformOp(wait)
    module = _pulse_module(frequency, frame, amplitude, duration, start, wait, stop)

    _configured_pipeline(_canonical_data()).apply(Context(), module)

    sequence = _assert_emission_ready(module)
    waits = [
        operation
        for operation in sequence.body.block.ops
        if isinstance(operation, WaitImmOp)
    ]
    assert [operation.duration.data for operation in waits] == [8]


def test_program_acquisition_length_overrides_hardware_default():
    frequency = ConstantOp(FrequencyAttr(6_200_000_000))
    frame = CreateFrameOp(frequency, StringAttr("port-0"))
    duration = ConstantOp(TimeAttr(1000e-9))
    acquire = AcquireOp(frame, duration)
    module = _pulse_module(frequency, frame, duration, acquire)

    _configured_pipeline(_readout_canonical_data()).apply(Context(), module)

    sequence = _assert_emission_ready(module)
    assert sequence.sequencer_config.integration_length.data == 1000


def test_pipeline_reports_unsupported_nested_control_flow_lowering():
    frequency, frame = _frame()
    amplitude = ConstantOp(AmplitudeAttr(0.25))
    duration = ConstantOp(TimeAttr(8e-9))
    start = StartContinuousWaveformOp(frame, amplitude)
    wait = WaitOp(start, duration)
    stop = StopContinuousWaveformOp(wait)
    lower = ArithConstantOp.from_int_and_width(0, IndexType())
    upper = ArithConstantOp.from_int_and_width(2, IndexType())
    step = ArithConstantOp.from_int_and_width(1, IndexType())
    loop = scf.ForOp(
        lower,
        upper,
        step,
        [],
        Block([start, wait, stop, scf.YieldOp()], arg_types=[IndexType()]),
    )
    module = _pulse_module(
        frequency,
        frame,
        amplitude,
        duration,
        lower,
        upper,
        step,
        loop,
    )

    with pytest.raises(PassFailedException, match="lower bound is not a static integer"):
        _configured_pipeline(_canonical_data()).apply(Context(), module)


def _module_config() -> ModuleConfigAttr:
    return ModuleConfigAttr(2, "cluster0", QbloxModuleKind.qcm_rf)


def _configured_module(sequence: SequenceOp) -> ModuleOp:
    sequence.properties["module_config"] = _module_config()
    return ModuleOp([sequence])


def _sequence_with(*operations) -> SequenceOp:
    return SequenceOp(
        "q0.drive",
        [*operations, StopOp()],
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
    )


@pytest.mark.parametrize(
    ("operation", "name"),
    [
        pytest.param(
            ConstantOp(FrequencyAttr(4_800_000_000)),
            "pulse.constant",
            id="pulse",
        ),
        pytest.param(scf.YieldOp(), "scf.yield", id="scf"),
        pytest.param(Q1ScfYieldOp(), "q1_scf.yield", id="q1-scf"),
        pytest.param(
            ArithConstantOp.from_int_and_width(1, IndexType()),
            "arith.constant",
            id="non-q1-assembly",
        ),
    ],
)
def test_pre_emission_verification_rejects_residual_operations(operation, name):
    module = _configured_module(_sequence_with(operation))

    with pytest.raises(PassFailedException, match=name):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_rejects_unresolved_synchronization():
    first_frequency, first_frame = _frame()
    second_frequency = ConstantOp(FrequencyAttr(4_800_000_000))
    second_frame = CreateFrameOp(second_frequency, StringAttr("q1.drive"))
    synchronize = SynchronizeOp(first_frame, second_frame)
    sequence = _sequence_with(
        synchronize,
        first_frequency,
        first_frame,
        second_frequency,
        second_frame,
    )

    with pytest.raises(PassFailedException, match="pulse.sync"):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_rejects_q1_cf():
    sequence = _sequence_with()
    block = sequence.body.block
    block.insert_op_before(JmpBranchOp([], block), block.first_op)

    with pytest.raises(PassFailedException, match="q1_cf.jmp_branch"):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_rejects_unrealized_cast():
    move = MoveImmRdOp(SU32Imm(1), Registers.R1)
    cast = UnrealizedConversionCastOp.get([move.rd], [Registers.R1])
    module = _configured_module(_sequence_with(move, cast))

    with pytest.raises(PassFailedException, match="unrealized_conversion_cast"):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_rejects_unallocated_register():
    move = MoveImmRdOp(SU32Imm(1), IntRegisterType.unallocated())

    with pytest.raises(PassFailedException, match="unallocated"):
        QbloxPreEmissionVerificationPass().apply(
            Context(), _configured_module(_sequence_with(move))
        )


def test_pre_emission_verification_wraps_sequence_invariant_failure():
    sequence = _sequence_with()
    sequence.properties["sequencer_config"] = SequencerConfigAttr(port_id="other.port")

    with pytest.raises(
        PassFailedException,
        match="Sequence 'q0.drive' failed Qblox verification:.*port_id conflicts",
    ):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_rejects_non_sequence_top_level_operation():
    module = ModuleOp([StopOp()])

    with pytest.raises(PassFailedException, match="top-level"):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_requires_sequence():
    with pytest.raises(PassFailedException, match="at least one top-level"):
        QbloxPreEmissionVerificationPass().apply(Context(), ModuleOp([]))


def test_pre_emission_verification_requires_complete_sequence_configuration():
    sequence = SequenceOp("q0.drive", [StopOp()])

    with pytest.raises(PassFailedException, match="missing its Qblox allocation"):
        QbloxPreEmissionVerificationPass().apply(Context(), ModuleOp([sequence]))


def test_pre_emission_verification_rejects_unallocated_block_argument():
    sequence = SequenceOp(
        "q0.drive",
        Region(Block([StopOp()], arg_types=[IntRegisterType.unallocated()])),
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
    )

    with pytest.raises(PassFailedException, match="unallocated"):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_rejects_duplicate_allocation():
    first = _sequence_with()
    second = SequenceOp(
        "q1.drive",
        [StopOp()],
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
    )
    first.properties["module_config"] = _module_config()
    second.properties["module_config"] = _module_config()
    module = ModuleOp([first, second])

    with pytest.raises(PassFailedException, match="Duplicate physical allocation"):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_rejects_acquisition_on_control_module():
    module_config = _module_config()
    sequence = SequenceOp(
        "q0.drive",
        [StopOp()],
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(acquisition_enabled=True),
        module_config=module_config,
    )

    with pytest.raises(PassFailedException, match="acquisition-capable sequencer"):
        QbloxPreEmissionVerificationPass().apply(
            Context(),
            ModuleOp([sequence]),
        )


def test_pre_emission_verification_accepts_referenced_tables():
    module_config = ModuleConfigAttr(
        3,
        "cluster0",
        QbloxModuleKind.qrm,
        inputs=[InputConfigAttr(0)],
    )
    sequence = SequenceOp(
        "q0.readout",
        [
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            AcquireWeightedImmImmImmImmImmOp(
                UI5Imm(0), UI24Imm(0), UI6Imm(0), UI6Imm(1), DurationImm(4)
            ),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("waveform_0", 0, [0.0]),
                make_waveform("waveform_1", 1, [0.0]),
            ]
        ),
        weights=ArrayAttr(
            [
                make_weight("weight_0", 0, [0.0]),
                make_weight("weight_1", 1, [0.0]),
            ]
        ),
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(
            acquisition_path_connections=[AcquisitionPathConnectionAttr(0, SignalPath.iq)]
        ),
        module_config=module_config,
    )
    module = ModuleOp([sequence])

    QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_requires_acquisition_route():
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [StopOp()],
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(acquisition_enabled=True),
        module_config=module_config,
    )
    module = ModuleOp([sequence])

    with pytest.raises(PassFailedException, match="without a configured acquisition route"):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_rejects_multi_block_sequence():
    sequence = _sequence_with()
    sequence.body.add_block(Block([StopOp()]))

    with pytest.raises(PassFailedException, match="exactly one block"):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def _qrm_module_config() -> ModuleConfigAttr:
    return ModuleConfigAttr(
        3,
        "cluster0",
        QbloxModuleKind.qrm,
        inputs=[InputConfigAttr(0)],
    )


def test_pre_emission_verification_uses_named_acquisition_bin_capacity_spec():
    """The aggregate acquisition-bin check uses the authoritative Qblox target limit."""
    limit = DEFAULT_QBLOX_TARGET.module_spec(QbloxModuleKind.qrm).acquisition_memory_bins
    assert limit is not None
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [StopOp()],
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, limit + 1)]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(
            acquisition_path_connections=[AcquisitionPathConnectionAttr(0, SignalPath.iq)]
        ),
        module_config=module_config,
    )
    module = ModuleOp([sequence])

    with pytest.raises(
        PassFailedException, match=rf"exceeding the {limit}-bin module limit"
    ):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_rejects_missing_acquisition_table_entry():
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [AcquireImmImmImmOp(UI5Imm(3), UI24Imm(0), DurationImm(4)), StopOp()],
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
        module_config=module_config,
    )
    module = ModuleOp([sequence])

    with pytest.raises(
        PassFailedException,
        match="uses acquisition indices \\[3\\].*absent from its acquisition table",
    ):
        QbloxPreEmissionVerificationPass().apply(Context(), module)


def test_pre_emission_verification_requires_square_weight_integration_length():
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [AcquireImmImmImmOp(UI5Imm(0), UI24Imm(0), DurationImm(4)), StopOp()],
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
        module_config=module_config,
    )

    with pytest.raises(
        PassFailedException,
        match="square-weight acquisition without a configured integration length",
    ):
        QbloxPreEmissionVerificationPass().apply(
            Context(),
            ModuleOp([sequence]),
        )


@pytest.mark.parametrize(
    "configuration",
    [
        pytest.param(
            {"acquisition_disabled": True},
            id="combined-path-disabled",
        ),
        pytest.param(
            {"acquisition_enabled": False},
            id="explicitly-not-enabled",
        ),
    ],
)
def test_pre_emission_verification_rejects_disabled_acquisition_path(configuration):
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [AcquireImmImmImmOp(UI5Imm(0), UI24Imm(0), DurationImm(4)), StopOp()],
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(
            unweighted_acquire=UnweightedAcquireConfigAttr(4),
            **configuration,
        ),
        module_config=module_config,
    )

    with pytest.raises(
        PassFailedException, match="acquisition path is explicitly disabled"
    ):
        QbloxPreEmissionVerificationPass().apply(
            Context(),
            ModuleOp([sequence]),
        )


def test_pre_emission_verification_rejects_missing_waveform_table_entry():
    sequence = _sequence_with(PlayImmImmImmOp(UI10Imm(1), UI10Imm(2), DurationImm(4)))

    with pytest.raises(
        PassFailedException,
        match="waveform indices \\[1, 2\\].*absent from its waveform table",
    ):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_rejects_missing_weight_table_entry():
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [
            AcquireWeightedImmImmImmImmImmOp(
                UI5Imm(0),
                UI24Imm(0),
                UI6Imm(1),
                UI6Imm(2),
                DurationImm(4),
            ),
            StopOp(),
        ],
        acquisitions=ArrayAttr([make_acquisition("acq_0", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
        module_config=module_config,
    )

    with pytest.raises(
        PassFailedException,
        match="weight indices \\[1, 2\\].*absent from its weight table",
    ):
        QbloxPreEmissionVerificationPass().apply(
            Context(),
            ModuleOp([sequence]),
        )


def test_pre_emission_verification_uses_named_waveform_sample_capacity_spec():
    """The per-sequencer waveform-sample check must reject at the authoritative Qblox
    sequencer target limit, not an ad hoc constant."""
    limit = DEFAULT_QBLOX_TARGET.sequencer(
        QbloxModuleKind.qcm_rf, 0
    ).sequencer_spec.waveform_sample_capacity
    sequence = SequenceOp(
        "q0.drive",
        [StopOp()],
        waveforms=ArrayAttr([make_waveform("wf_0", 0, [0.0] * (limit + 1))]),
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
    )

    with pytest.raises(
        PassFailedException, match=f"exceeding the {limit} sample sequencer capacity"
    ):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_uses_named_weight_sample_capacity_spec():
    """The per-sequencer weight-sample check must reject at the authoritative Qblox readout
    target limit, not an ad hoc constant."""
    readout_spec = DEFAULT_QBLOX_TARGET.sequencer(
        QbloxModuleKind.qrm, 0
    ).sequencer_spec.readout
    assert readout_spec is not None
    limit = readout_spec.weight_sample_capacity
    module_config = _qrm_module_config()
    sequence = SequenceOp(
        "q0.readout",
        [StopOp()],
        weights=ArrayAttr([make_weight("weight_0", 0, [0.0] * (limit + 1))]),
        instrument_id="cluster0",
        slot_idx=3,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
        module_config=module_config,
    )

    with pytest.raises(
        PassFailedException, match=f"exceeding the {limit} sample sequencer capacity"
    ):
        QbloxPreEmissionVerificationPass().apply(Context(), ModuleOp([sequence]))
