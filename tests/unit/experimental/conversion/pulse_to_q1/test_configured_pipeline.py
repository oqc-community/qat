# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from collections import defaultdict
from dataclasses import replace

import pytest
from frozendict import frozendict
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
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.experimental.backend.qblox.pre_emission_verification import (
    QbloxPreEmissionVerificationPass,
)
from qat.experimental.conversion.pulse_to_q1.hardware_binding import (
    QbloxHardwareBindingPass,
)
from qat.experimental.conversion.pulse_to_q1.passes import (
    PulseToQ1LoweringPass,
    Q1PreAcquireTransformationPass,
    Q1PulseLegalisationPass,
    Q1PulseValidationPass,
    create_qblox_configured_q1_pipeline,
)
from qat.experimental.conversion.pulse_to_q1.sequence_convention import (
    QbloxSequenceInitialisationPass,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    IntegrateOp,
    PhaseAttr,
    PhaseShiftOp,
    PulseOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SynchronizeOp,
    TimeAttr,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.schedule import build_pulse_schedule
from qat.experimental.dialect.pulse.transforms.pipeline import PulsePipelineManager
from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    DurationImm,
    IntRegisterType,
    LabelOp,
    LoopRdImmOp,
    MoveImmRdOp,
    PlayImmImmImmOp,
    Q1RegisterType,
    Registers,
    SetAwgOffsImmImmOp,
    StopOp,
    SU32Imm,
    UI5Imm,
    UI6Imm,
    UI10Imm,
    UI24Imm,
    UpdParamImmOp,
)
from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1.ir.schedule import build_q1_schedule
from qat.experimental.dialect.q1.transforms.reg_alloc import (
    LinearScanRegisterAllocationPass,
)
from qat.experimental.dialect.q1.transforms.verify_allocation import VerifyQ1AllocationPass
from qat.experimental.dialect.q1_cf import JmpBranchOp
from qat.experimental.dialect.q1_cf.transforms.linearise_q1_cf import LineariseQ1CfToQ1Pass
from qat.experimental.dialect.q1_scf import YieldOp as Q1ScfYieldOp
from qat.experimental.dialect.q1_scf.transforms.lower_scf import LowerScfToQ1ScfPass
from qat.experimental.dialect.q1_scf.transforms.lower_to_cf import LowerQ1ScfToQ1CfPass
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionPathConnectionAttr,
    InputConfigAttr,
    ModuleConfigAttr,
    NcoConfigAttr,
    SequencerConfigAttr,
    UnweightedAcquireConfigAttr,
    make_acquisition,
    make_dense_floats,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPassPipeline
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.hardware.qblox import DEFAULT_QBLOX_TARGET
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind, SignalPath
from qat.experimental.system_data.hardware.qblox.target import Q1SequencerType
from qat.experimental.system_data.pulse.constraints import PulseLevelConstraints

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


@irdl_op_definition
class _DynamicIndexSourceOp(IRDLOperation):
    name = "test.dynamic_index_source"
    result = result_def(IndexType)

    def __init__(self):
        super().__init__(result_types=[IndexType()])


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
    canonical = _canonical_data()

    with pytest.raises(VerifyException, match="qblox-hardware-binding"):
        OrderedPassPipeline(
            (
                Q1OutliningPass(),
                QbloxSequenceInitialisationPass(),
                QbloxHardwareBindingPass(canonical),
                Q1PulseValidationPass(),
            )
        )

    with pytest.raises(VerifyException, match="qblox-sequence-initialisation"):
        OrderedPassPipeline(
            (
                Q1OutliningPass(),
                QbloxHardwareBindingPass(canonical),
                Q1PulseValidationPass(),
            )
        )

    with pytest.raises(VerifyException, match="acquire-pre-q1-transformation"):
        OrderedPassPipeline(
            (
                Q1OutliningPass(),
                QbloxHardwareBindingPass(canonical),
                QbloxSequenceInitialisationPass(),
                Q1PulseValidationPass(),
                Q1PulseLegalisationPass(),
                Q1PreAcquireTransformationPass(),
            )
        )

    with pytest.raises(VerifyException, match="q1-pulse-legalisation"):
        OrderedPassPipeline(
            (
                Q1OutliningPass(),
                QbloxHardwareBindingPass(canonical),
                QbloxSequenceInitialisationPass(),
                Q1PulseValidationPass(),
                Q1PreAcquireTransformationPass(),
                PulseToQ1LoweringPass(),
            )
        )

    with pytest.raises(VerifyException, match="lower-scf-to-q1-scf"):
        OrderedPassPipeline((LowerQ1ScfToQ1CfPass(), LowerScfToQ1ScfPass()))

    with pytest.raises(VerifyException, match="lower-q1-scf-to-q1-cf"):
        OrderedPassPipeline((LineariseQ1CfToQ1Pass(), LowerQ1ScfToQ1CfPass()))

    assert (
        LineariseQ1CfToQ1Pass in QbloxPreEmissionVerificationPass().required_predecessors()
    )


def test_square_validation_is_independent_of_bound_q1_core_clock():
    frequency = ConstantOp(FrequencyAttr(6_200_000_000))
    frame = CreateFrameOp(frequency, StringAttr("port-0"))
    width = ConstantOp(TimeAttr(12e3))
    amplitude = ConstantOp(AmplitudeAttr(0.5))
    waveform = SquareWaveformOp(width, amplitude)
    pulse = PulseOp(frame, waveform)
    module = _pulse_module(frequency, frame, width, amplitude, waveform, pulse)
    readout_data = replace(
        DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.readout),
        q1_core_clock_period_ns=8,
    )
    target_data = replace(
        DEFAULT_QBLOX_TARGET,
        sequencer_specs=frozendict(
            {
                **DEFAULT_QBLOX_TARGET.sequencer_specs,
                Q1SequencerType.readout: readout_data,
            }
        ),
    )

    create_qblox_configured_q1_pipeline(_readout_canonical_data(), target_data).apply(
        Context(), module
    )


def test_pipeline_lowers_normalized_pulse_to_configured_flat_q1():
    frequency, frame = _frame()
    duration = ConstantOp(TimeAttr(8e3))
    wait = WaitOp(frame, duration)
    module = _pulse_module(frequency, frame, duration, wait)

    _configured_pipeline(_canonical_data()).apply(Context(), module)

    sequence = _assert_emission_ready(module)
    assert sequence.channel_id.data == "port_0"
    assert not any(isinstance(op, CreateFrameOp) for op in sequence.walk())


@pytest.mark.parametrize(
    ("checked_pass", "previous_checkpoints"),
    [
        (LinearScanRegisterAllocationPass, 0),
        (LowerQ1ScfToQ1CfPass, 1),
        (LineariseQ1CfToQ1Pass, 2),
    ],
)
@pytest.mark.parametrize(
    ("register_type", "issue"),
    [
        (IntRegisterType.unallocated(), "an unallocated Q1 register"),
        (IntRegisterType.infinite_register(42), "a virtual Q1 register (inf_gpr_42)"),
    ],
)
def test_allocation_failure_names_checked_stage(
    checked_pass, previous_checkpoints, register_type, issue
):
    frequency, frame = _frame()
    duration = ConstantOp(TimeAttr(8e3))
    module = _pulse_module(frequency, frame, duration, WaitOp(frame, duration))
    completed_checkpoints = []
    injected_sequence = None

    def inject_after_checked_pass(previous, module, following):
        nonlocal injected_sequence
        if isinstance(previous, VerifyQ1AllocationPass):
            completed_checkpoints.append(previous)
        if isinstance(previous, checked_pass):
            assert len(completed_checkpoints) == previous_checkpoints
            assert isinstance(following, VerifyQ1AllocationPass)
            [injected_sequence] = [
                operation
                for operation in module.walk()
                if isinstance(operation, SequenceOp)
            ]
            injected_sequence.body.blocks[0].add_op(MoveImmRdOp(SU32Imm(0), register_type))

    pipeline = replace(
        _configured_pipeline(_canonical_data()), callback=inject_after_checked_pass
    )

    with pytest.raises(PassFailedException) as exc_info:
        pipeline.apply(Context(), module)

    assert injected_sequence is not None
    assert len(completed_checkpoints) == previous_checkpoints
    assert str(exc_info.value) == (
        f"Register allocation verification failed after pass {checked_pass.name!r}. "
        "Found 1 issue(s):\n"
        f"Sequence {injected_sequence.channel_id.data!r}: result 0 of operation "
        f"{MoveImmRdOp.name!r} is {issue}."
    )


def test_pipeline_preserves_timing_instruction_through_control_flow_lowering():
    frequency, frame = _frame()
    amplitude = ConstantOp(AmplitudeAttr(0.25))
    duration = ConstantOp(TimeAttr(8e3))
    start = StartContinuousWaveformOp(frame, amplitude)
    wait = WaitOp(start, duration)
    stop = StopContinuousWaveformOp(wait)
    module = _pulse_module(frequency, frame, amplitude, duration, start, wait, stop)

    _configured_pipeline(_canonical_data()).apply(Context(), module)

    sequence = _assert_emission_ready(module)
    timing = [
        operation.duration.data if isinstance(operation, UpdParamImmOp) else "offset"
        for operation in sequence.body.block.ops
        if isinstance(operation, SetAwgOffsImmImmOp | UpdParamImmOp)
    ]
    # Prologue latch, the start offset latched by the wait, then the stop offset latched
    # before the sequence ends.
    assert timing == [4, "offset", 8, "offset", 4]


def _square_pulse_body(frame: CreateFrameOp) -> list:
    amplitude = ConstantOp(AmplitudeAttr(0.25))
    duration = ConstantOp(TimeAttr(8e3))
    start = StartContinuousWaveformOp(frame, amplitude)
    wait = WaitOp(start, duration)
    return [amplitude, duration, start, wait, StopContinuousWaveformOp(wait)]


def _phase_shift_body(frame: CreateFrameOp) -> list:
    phase = ConstantOp(PhaseAttr(0.5))
    return [phase, PhaseShiftOp(frame, phase)]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            _square_pulse_body,
            ["set_awg_offs", "upd_param", "set_awg_offs", "upd_param"],
            id="square_pulse",
        ),
        pytest.param(_phase_shift_body, ["set_ph_delta", "upd_param"], id="phase_shift"),
    ],
)
def test_pipeline_latches_pending_parameters_at_end_of_shot_loop(body, expected):
    """A shot ending in a parameter update latches it before looping back to ``wait_sync``,
    and the loop body still opens with the shot-synchronisation prologue."""
    frequency, frame = _frame()
    lower = ArithConstantOp.from_int_and_width(0, IndexType())
    upper = ArithConstantOp.from_int_and_width(3, IndexType())
    step = ArithConstantOp.from_int_and_width(1, IndexType())
    loop = scf.ForOp(
        lower.result,
        upper.result,
        step.result,
        [],
        Block([*body(frame), scf.YieldOp()], arg_types=[IndexType()]),
    )
    module = _pulse_module(frequency, frame, lower, upper, step, loop)

    _configured_pipeline(_canonical_data()).apply(Context(), module)

    ops = list(_assert_emission_ready(module).body.block.ops)
    [label] = [index for index, op in enumerate(ops) if isinstance(op, LabelOp)]
    [loop_back] = [index for index, op in enumerate(ops) if isinstance(op, LoopRdImmOp)]
    shot = [op.name.rsplit(".", 1)[-1] for op in ops[label + 1 : loop_back]]
    assert shot == ["wait_sync", "reset_ph", "upd_param", *expected]


def test_program_acquisition_length_overrides_hardware_default():
    frequency = ConstantOp(FrequencyAttr(6_200_000_000))
    frame = CreateFrameOp(frequency, StringAttr("port-0"))
    duration = ConstantOp(TimeAttr(1000e3))
    acquire = AcquireOp(frame, duration)
    integrate = IntegrateOp(acquire.acquisition_result)
    module = _pulse_module(frequency, frame, duration, acquire, integrate)

    _configured_pipeline(_readout_canonical_data()).apply(Context(), module)

    sequence = _assert_emission_ready(module)
    assert sequence.sequencer_config.integration_length.data == 1000


def test_pipeline_reports_unsupported_nested_control_flow_lowering():
    frequency, frame = _frame()
    amplitude = ConstantOp(AmplitudeAttr(0.25))
    duration = ConstantOp(TimeAttr(8e3))
    start = StartContinuousWaveformOp(frame, amplitude)
    wait = WaitOp(start, duration)
    stop = StopContinuousWaveformOp(wait)
    lower = ArithConstantOp.from_int_and_width(0, IndexType())
    upper = _DynamicIndexSourceOp()
    step = ArithConstantOp.from_int_and_width(1, IndexType())
    loop = scf.ForOp(
        lower.result,
        upper.result,
        step.result,
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

    with pytest.raises(
        PassFailedException, match="Dynamic For loop bounds not currently supported"
    ):
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


@pytest.mark.parametrize(
    ("table_name", "entry", "invalid_data", "message"),
    [
        pytest.param(
            "waveforms",
            make_waveform("invalid", 0, [0.0]),
            make_dense_floats([1.5]),
            "invalid waveform table entry.*out of DAC range",
            id="waveform",
        ),
        pytest.param(
            "weights",
            make_weight("invalid", 0, [0.0]),
            make_dense_floats([-1.5]),
            "invalid weight table entry.*out of ADC range",
            id="weight",
        ),
    ],
)
def test_pre_emission_verification_checks_table_entries(
    table_name, entry, invalid_data, message
):
    object.__setattr__(entry, "data", invalid_data)
    sequence = SequenceOp(
        "q0.drive",
        [StopOp()],
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=0,
        sequencer_config=SequencerConfigAttr(),
        **{table_name: ArrayAttr([entry])},
    )

    with pytest.raises(PassFailedException, match=message):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


@pytest.mark.parametrize("frequency", [600_000_000.0, float("inf")])
def test_pre_emission_verification_rejects_out_of_range_nco_frequency(frequency):
    sequence = _sequence_with()
    sequence.properties["sequencer_config"] = SequencerConfigAttr(
        nco=NcoConfigAttr(frequency=frequency)
    )

    with pytest.raises(PassFailedException, match="NCO frequency .* is outside"):
        QbloxPreEmissionVerificationPass().apply(Context(), _configured_module(sequence))


def test_pre_emission_verification_enforces_instruction_capacity():
    sequence = _sequence_with(MoveImmRdOp(SU32Imm(1), Registers.R1))
    control_spec = replace(
        DEFAULT_QBLOX_TARGET.sequencer_spec(Q1SequencerType.control),
        instruction_capacity=1,
    )
    target_data = replace(
        DEFAULT_QBLOX_TARGET,
        sequencer_specs=frozendict(
            {
                **DEFAULT_QBLOX_TARGET.sequencer_specs,
                Q1SequencerType.control: control_spec,
            }
        ),
    )

    with pytest.raises(PassFailedException, match="exceeding the 1 instruction capacity"):
        QbloxPreEmissionVerificationPass(target_data).apply(
            Context(), _configured_module(sequence)
        )


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
    limit = DEFAULT_QBLOX_TARGET.sequencer_spec(
        Q1SequencerType.control
    ).waveform_sample_capacity
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
    limit = DEFAULT_QBLOX_TARGET.sequencer_spec(
        Q1SequencerType.readout
    ).readout.weight_sample_capacity
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


_PULSE_EVENT_KINDS = {
    "pulse": "play",
    "readout": "acquire",
    "start continuous waveform": "offset",
    "stop continuous waveform": "offset",
}


def _pulse_events(module: ModuleOp) -> dict[str, list[tuple[str, float]]]:
    """Return the start time of each play, acquire and offset change per channel."""
    events = defaultdict(list)
    for resource, timeline in build_pulse_schedule(
        module, time_unit="ns"
    ).timelines.items():
        for event in timeline:
            if (kind := _PULSE_EVENT_KINDS.get(event.label)) is not None:
                events[resource.replace("-", "_")].append((kind, event.start))
    return dict(events)


def _q1_events(module: ModuleOp) -> dict[str, list[tuple[str, float]]]:
    """Return the start time of each play, acquire and offset change per sequence."""
    events = defaultdict(list)
    for resource, timeline in build_q1_schedule(module).timelines.items():
        offset = 0j
        for event in timeline:
            if event.label in ("play", "acquire"):
                events[resource].append((event.label, event.start))
            elif (
                event.signal is not None and event.signal.size and event.signal[0] != offset
            ):
                offset = event.signal[0]
                events[resource].append(("offset", event.start))
    return dict(events)


def _channel_with_phase_shifts(port_id: str, waveform: SquareWaveformOp) -> list:
    frequency = ConstantOp(FrequencyAttr(200_000_000))
    frame = CreateFrameOp(frequency, StringAttr(port_id))
    operations = [frequency, frame]
    current = frame.result
    for phase in (0.5, 0.25, 0.125):
        constant = ConstantOp(PhaseAttr(phase))
        shift = PhaseShiftOp(current, constant)
        pulse = PulseOp(shift.result, waveform)
        operations.extend([constant, shift, pulse])
        current = pulse.result
    duration = ConstantOp(TimeAttr(16_000))
    acquire = AcquireOp(current, duration, label="readout")
    return [*operations, duration, acquire, IntegrateOp(acquire.acquisition_result)]


def _channel_with_continuous_waveform(port_id: str, waveform: SquareWaveformOp) -> list:
    frequency = ConstantOp(FrequencyAttr(200_000_000))
    frame = CreateFrameOp(frequency, StringAttr(port_id))
    pulse = PulseOp(frame, waveform)
    phase = ConstantOp(PhaseAttr(0.5))
    shift = PhaseShiftOp(pulse.result, phase)
    amplitude = ConstantOp(AmplitudeAttr(0.25))
    start = StartContinuousWaveformOp(shift.result, amplitude)
    duration = ConstantOp(TimeAttr(20_000))
    wait = WaitOp(start.result, duration)
    stop = StopContinuousWaveformOp(wait.result)
    return [frequency, frame, pulse, phase, shift, amplitude, start, duration, wait, stop]


def test_pipeline_keeps_pulse_timing_with_phase_shifts_on_several_channels():
    """Phase updates must not delay later operations on any channel."""
    canonical = canonical_data(
        QbloxModuleKind.qrm,
        configurations=[
            supplied([sequencer(0, outputs=(0,), inputs=(0,))]),
            supplied([sequencer(1, outputs=(1,))]),
        ],
        oscillator_frequency=None,
        carrier_frequency=200_000_000,
    )
    width = ConstantOp(TimeAttr(16_000))
    amplitude = ConstantOp(AmplitudeAttr(0.5))
    waveform = SquareWaveformOp(width, amplitude)
    *readout_ops, integrate = _channel_with_phase_shifts("port-0", waveform)
    body = [
        width,
        amplitude,
        waveform,
        *readout_ops,
        integrate,
        *_channel_with_continuous_waveform("port-1", waveform),
        func.ReturnOp(integrate.result),
    ]
    module = ModuleOp(
        [func.FuncOp("main", ((), (integrate.result.type,)), Region(Block(body)))]
    )
    PulsePipelineManager(
        PulseLevelConstraints.derive(canonical)
    ).build_default_pipeline().apply(Context(), module)
    expected = _pulse_events(module)

    create_qblox_configured_q1_pipeline(canonical).apply(Context(), module)
    actual = _q1_events(module)

    # Every Q1 sequence starts with the same prologue: set_latch_en and upd_param.
    prologue = 2 * DEFAULT_QBLOX_TARGET.q1asm.min_rt_duration_ns
    assert actual == {
        channel: [(kind, start + prologue) for kind, start in events]
        for channel, events in expected.items()
    }
