# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from numpy import array
from xdsl.context import Context
from xdsl.dialects import func, scf
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import FloatAttr, IndexType, ModuleOp, StringAttr, f64
from xdsl.ir import Block, Region

from qat.experimental.backend.qblox.codegen import emit_qblox_program
from qat.experimental.conversion.pulse_to_q1.passes import (
    create_qblox_configured_q1_pipeline,
)
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    GaussianWaveformOp,
    IntegrateOp,
    PulseOp,
    SquareWaveformOp,
    TimeAttr,
    WaitOp,
    WeightsAttr,
)
from qat.experimental.dialect.pulse.transforms.pipeline import PulsePipelineManager
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.hardware.qblox import QbloxHardwareView
from qat.experimental.system_data.hardware.qblox.configuration import (
    BooleanOutputConnection,
    QbloxSequencerConfiguration,
    SequencerConnection,
)
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind
from qat.experimental.system_data.pulse.constraints import PulseLevelConstraints

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)


def _pulse_module(*operations, result=None) -> ModuleOp:
    result_types = () if result is None else (result.type,)
    return_op = func.ReturnOp() if result is None else func.ReturnOp(result)
    return ModuleOp(
        [func.FuncOp("main", ((), result_types), Region(Block([*operations, return_op])))]
    )


def _compile(module: ModuleOp, canonical: CanonicalSystemData):
    PulsePipelineManager(
        PulseLevelConstraints.derive(canonical)
    ).build_default_pipeline().apply(Context(), module)
    create_qblox_configured_q1_pipeline(canonical).apply(Context(), module)
    hardware_view = QbloxHardwareView.derive(canonical)
    return emit_qblox_program(module, hardware_view)


def _control_module(port_id: str, carrier: int) -> ModuleOp:
    frequency = ConstantOp(FrequencyAttr(carrier))
    frame = CreateFrameOp(frequency, StringAttr(port_id))
    width = ConstantOp(TimeAttr(16_000))
    amplitude = ConstantOp(AmplitudeAttr(0.5))
    fractional_breadth = ArithConstantOp(FloatAttr(0.2, f64), f64)
    waveform = GaussianWaveformOp(width, amplitude, fractional_breadth, regularize=False)
    return _pulse_module(
        frequency,
        frame,
        width,
        amplitude,
        fractional_breadth,
        waveform,
        PulseOp(frame, waveform),
    )


def _acquisition_module(
    port_id: str,
    carrier: int,
    weights: WeightsAttr | None = None,
) -> ModuleOp:
    frequency = ConstantOp(FrequencyAttr(carrier))
    frame = CreateFrameOp(frequency, StringAttr(port_id))
    duration = ConstantOp(TimeAttr(16_000))
    acquire = AcquireOp(frame, duration, weights=weights, label="readout")
    integrate = IntegrateOp(acquire.acquisition_result)
    return ModuleOp(
        [
            func.FuncOp(
                "main",
                ((), (integrate.result.type,)),
                Region(
                    Block(
                        [
                            frequency,
                            frame,
                            duration,
                            acquire,
                            integrate,
                            func.ReturnOp(integrate.result),
                        ]
                    )
                ),
            )
        ]
    )


def _pulse_chain(port_id: str, carrier: int):
    frequency = ConstantOp(FrequencyAttr(carrier))
    frame = CreateFrameOp(frequency, StringAttr(port_id))
    duration = ConstantOp(TimeAttr(8_000))
    amplitude = ConstantOp(AmplitudeAttr(0.5))
    waveform = SquareWaveformOp(duration, amplitude)
    return frequency, frame, duration, amplitude, waveform, PulseOp(frame, waveform)


def _repeated_wait_module(port_ids: tuple[str, ...], carrier: int) -> ModuleOp:
    body_ops = []
    for port_id in port_ids:
        frequency = ConstantOp(FrequencyAttr(carrier))
        frame = CreateFrameOp(frequency, StringAttr(port_id))
        duration = ConstantOp(TimeAttr(8_000))
        body_ops.extend([frequency, frame, duration, WaitOp(frame, duration)])

    lower = ArithConstantOp.from_int_and_width(0, IndexType())
    upper = ArithConstantOp.from_int_and_width(2, IndexType())
    step = ArithConstantOp.from_int_and_width(1, IndexType())
    loop = scf.ForOp(
        lower,
        upper,
        step,
        [],
        Block([*body_ops, scf.YieldOp()], arg_types=[IndexType()]),
    )
    return _pulse_module(lower, upper, step, loop)


def _canonical_for(
    kind: QbloxModuleKind,
    acquire: bool = False,
) -> CanonicalSystemData:
    rf = kind in (QbloxModuleKind.qcm_rf, QbloxModuleKind.qrm_rf)
    return canonical_data(
        kind,
        configurations=[
            supplied(
                [sequencer(0, outputs=(0,), inputs=(0,) if acquire else ())],
            )
        ],
        carrier_frequency=4_200_000_000 if rf else 200_000_000,
        oscillator_frequency=4_000_000_000 if rf else None,
    )


def test_compiles_qcm_rf_control_with_lo_nco_split_and_waveforms():
    canonical = _canonical_for(QbloxModuleKind.qcm_rf)

    program = _compile(_control_module("port-0", 4_200_000_000), canonical)

    package = program.packages["port_0"]
    assert package.seq_config.nco.freq == 200_000_000
    assert len(package.sequence.waveforms) == 2
    assert package.mod_config.lo.out0_freq == 4_000_000_000
    assert "play 0, 1, 16" in package.sequence.program


def test_compiles_qcm_baseband_control_without_local_oscillator():
    canonical = _canonical_for(QbloxModuleKind.qcm)

    program = _compile(_control_module("port-0", 200_000_000), canonical)

    package = program.packages["port_0"]
    assert package.seq_config.nco.freq == 200_000_000
    assert package.mod_config.lo.model_dump(exclude_none=True) == {}


def test_compiles_qrm_rf_square_acquisition_and_table():
    canonical = _canonical_for(QbloxModuleKind.qrm_rf, acquire=True)

    program = _compile(_acquisition_module("port-0", 4_200_000_000), canonical)

    package = program.packages["port_0"]
    assert package.mod_config.lo.out0_in0_freq == 4_000_000_000
    assert package.seq_config.square_weight_acq.integration_length == 16
    assert package.sequence.acquisitions == {"readout": {"index": 0, "num_bins": 1}}
    assert "acquire 0, R1, 16" in package.sequence.program


def test_compiles_qrm_weighted_acquisition_and_both_weight_tables():
    canonical = _canonical_for(QbloxModuleKind.qrm, acquire=True)
    weights = WeightsAttr(array([0.5 + 0.25j, 0.25 - 0.5j]))

    program = _compile(
        _acquisition_module("port-0", 200_000_000, weights=weights),
        canonical,
    )

    package = program.packages["port_0"]
    assert sorted(package.sequence.weights.values(), key=lambda entry: entry["index"]) == [
        {"data": [0.5, 0.25], "index": 0},
        {"data": [0.25, -0.5], "index": 1},
    ]
    assert "acquire_weighted 0, R1, R2, R3, 16" in package.sequence.program


def test_compiles_multiple_sequences_sharing_one_module():
    canonical = canonical_data(
        QbloxModuleKind.qcm,
        configurations=[
            supplied([sequencer(0, outputs=(0,))]),
            supplied([sequencer(1, outputs=(1,))]),
        ],
        oscillator_frequency=None,
        carrier_frequency=200_000_000,
    )

    program = _compile(
        _pulse_module(
            *_pulse_chain("port-0", 200_000_000),
            *_pulse_chain("port-1", 200_000_000),
        ),
        canonical,
    )

    assert set(program.packages) == {"port_0", "port_1"}
    assert {
        (package.instrument_id, package.slot_idx) for package in program.packages.values()
    } == {("cluster", 2)}
    assert {package.seq_idx for package in program.packages.values()} == {0, 1}
    assert len({str(package.mod_config) for package in program.packages.values()}) == 1


def test_compiles_repeated_sequences_with_legacy_synchronization_scaffolding():
    canonical = canonical_data(
        QbloxModuleKind.qcm,
        configurations=[
            supplied([sequencer(0, outputs=(0,))]),
            supplied([sequencer(1, outputs=(1,))]),
        ],
        oscillator_frequency=None,
        carrier_frequency=200_000_000,
    )

    module = _repeated_wait_module(("port-0", "port-1"), 200_000_000)
    create_qblox_configured_q1_pipeline(canonical).apply(Context(), module)
    program = emit_qblox_program(module, QbloxHardwareView.derive(canonical))

    for package in program.packages.values():
        lines = package.sequence.program.splitlines()
        assert lines[:3] == ["set_mrk 3", "set_latch_en 1, 4", "upd_param 4"]
        assert lines[-1] == "stop"
        label = next(index for index, line in enumerate(lines) if line.endswith(":"))
        assert lines[label + 1 : label + 4] == [
            "wait_sync 4",
            "reset_ph",
            "upd_param 4",
        ]
        assert package.seq_config.sync_en is True


def test_repeated_compilation_serializes_deterministically():
    canonical = _canonical_for(QbloxModuleKind.qcm_rf)

    first = _compile(_control_module("port-0", 4_200_000_000), canonical)
    second = _compile(_control_module("port-0", 4_200_000_000), canonical)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


def test_compiles_qrc_control_and_acquisition_port_banks():
    canonical = canonical_data(
        QbloxModuleKind.qrc,
        configurations=[
            supplied(
                [
                    sequencer(
                        8,
                        outputs=(2,),
                        values={"awg": {"mod_en": True}},
                    )
                ]
            ),
            supplied(
                [
                    sequencer(
                        0,
                        outputs=(0,),
                        inputs=(0,),
                        values={
                            "demod_en_acq": True,
                            "awg": {"gain_path0": 0.5, "mod_en": True},
                        },
                    )
                ],
                module_values={"lo": {"out0_in0_en": True}},
            ),
        ],
    )
    readout_frequency = ConstantOp(FrequencyAttr(4_200_000_000))
    readout_frame = CreateFrameOp(readout_frequency, StringAttr("port-1"))
    readout_duration = ConstantOp(TimeAttr(16_000))
    readout_acquire = AcquireOp(
        readout_frame, readout_duration, weights=None, label="readout"
    )
    readout_integrate = IntegrateOp(readout_acquire.acquisition_result)

    program = _compile(
        _pulse_module(
            *_pulse_chain("port-0", 4_200_000_000),
            readout_frequency,
            readout_frame,
            readout_duration,
            readout_acquire,
            readout_integrate,
            result=readout_integrate.result,
        ),
        canonical,
    )

    control = program.packages["port_0"]
    readout = program.packages["port_1"]
    assert {
        "physical_channel_id": control.physical_channel_id,
        "seq_idx": control.seq_idx,
        "connections": control.seq_config.connection.model_dump(exclude_none=True),
        "awg_modulation": control.seq_config.awg.mod_en,
        "program": control.sequence.program,
    } == {
        "physical_channel_id": "port-0",
        "seq_idx": 8,
        "connections": {"bulk_value": ["out2"]},
        "awg_modulation": None,
        "program": ("set_mrk 3\nset_latch_en 1, 4\nupd_param 4\nplay 0, 1, 8\nstop\n"),
    }
    assert {
        "physical_channel_id": readout.physical_channel_id,
        "seq_idx": readout.seq_idx,
        "connections": readout.seq_config.connection.model_dump(exclude_none=True),
        "awg_gain": readout.seq_config.awg.gain_path0,
        "awg_modulation": readout.seq_config.awg.mod_en,
        "demodulation": readout.seq_config.demod_en_acq,
        "integration_length": (readout.seq_config.square_weight_acq.integration_length),
        "acquisitions": readout.sequence.acquisitions,
    } == {
        "physical_channel_id": "port-1",
        "seq_idx": 0,
        "connections": {"bulk_value": ["out0", "in0"]},
        "awg_gain": 0.5,
        "awg_modulation": None,
        "demodulation": None,
        "integration_length": 16,
        "acquisitions": {"readout": {"index": 0, "num_bins": 1}},
    }
    assert readout.mod_config.lo.out0_in0_en is None


@pytest.mark.parametrize(
    ("kind", "acquire", "expected_connections"),
    [
        pytest.param(
            QbloxModuleKind.qcm,
            False,
            {"bulk_value": ["out0"]},
            id="qcm",
        ),
        pytest.param(
            QbloxModuleKind.qcm_rf,
            False,
            {"bulk_value": ["out0"]},
            id="qcm-rf",
        ),
        pytest.param(
            QbloxModuleKind.qrm,
            True,
            {"bulk_value": ["out0", "in0"]},
            id="qrm",
        ),
        pytest.param(
            QbloxModuleKind.qrm_rf,
            True,
            {"bulk_value": ["out0", "in0"]},
            id="qrm-rf",
        ),
        pytest.param(
            QbloxModuleKind.qrc,
            True,
            {"bulk_value": ["out0", "in0"]},
            id="qrc",
        ),
    ],
)
def test_all_five_module_kinds_emit_independent_expected_payload_values(
    kind: QbloxModuleKind,
    acquire: bool,
    expected_connections: dict[str, object],
):
    canonical = _canonical_for(kind, acquire=acquire)
    carrier = 4_200_000_000 if kind.value.endswith("_rf") else 200_000_000
    program = _compile(_pulse_module(*_pulse_chain("port-0", carrier)), canonical)
    package = program.packages["port_0"]

    assert (
        package.pulse_channel_id,
        package.physical_channel_id,
        package.instrument_id,
        package.slot_idx,
        package.seq_idx,
    ) == ("port_0", "port-0", "cluster", 2, 0)
    assert package.sequence.program == (
        "set_mrk 3\nset_latch_en 1, 4\nupd_param 4\nplay 0, 1, 8\nstop\n"
    )
    assert package.sequence.waveforms == {
        "waveform_0_I": {"data": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "index": 0},
        "waveform_0_Q": {"data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "index": 1},
    }
    assert package.sequence.weights == {}
    assert package.sequence.acquisitions == {}
    assert (
        package.seq_config.connection.model_dump(exclude_none=True) == expected_connections
    )


@pytest.mark.parametrize(
    "kind",
    [
        QbloxModuleKind.qcm_rf,
        QbloxModuleKind.qrm_rf,
        QbloxModuleKind.qrc,
    ],
)
@pytest.mark.parametrize("enabled", [True, False])
def test_rf_boolean_output_alias_reaches_runtime_payload(kind, enabled):
    canonical = canonical_data(
        kind,
        configurations=[
            supplied(
                [
                    QbloxSequencerConfiguration(
                        index=0,
                        connection=SequencerConnection(
                            boolean_output_connections=(
                                BooleanOutputConnection(0, enabled),
                            )
                        ),
                    )
                ]
            )
        ],
    )
    carrier = 4_200_000_000

    program = _compile(_control_module("port-0", carrier), canonical)

    assert program.packages["port_0"].seq_config.connection.out0 is enabled


@pytest.mark.parametrize("kind", [QbloxModuleKind.qrm_rf, QbloxModuleKind.qrc])
@pytest.mark.parametrize("enabled", [True, False])
def test_rf_boolean_acquisition_alias_reaches_runtime_payload(kind, enabled):
    canonical = canonical_data(
        kind,
        configurations=[
            supplied(
                [
                    QbloxSequencerConfiguration(
                        index=0,
                        connection=SequencerConnection(
                            boolean_output_connections=(BooleanOutputConnection(0, True),),
                            combined_acquisition_alias=enabled,
                        ),
                    )
                ]
            )
        ],
    )
    carrier = 4_200_000_000

    module = (
        _acquisition_module("port-0", carrier)
        if enabled
        else _control_module("port-0", carrier)
    )
    program = _compile(module, canonical)

    assert program.packages["port_0"].seq_config.connection.acq is enabled
