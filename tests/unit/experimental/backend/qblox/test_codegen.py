# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import Block, Region
from xdsl.irdl import irdl_op_definition, region_def
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.target_data import TARGET_DATA
from qat.executables import Executable
from qat.experimental.backend.qblox.codegen import emit_qblox_program
from qat.experimental.dialect.q1 import (
    AcquireImmImmImmOp,
    AcquireWeightedImmImmImmImmImmOp,
    AcquireWeightedImmRsRsRsImmOp,
    AddRsImmRdOp,
    DurationImm,
    MoveImmRdOp,
    MoveRsRdOp,
    PlayImmImmImmOp,
    PlayRsRsImmOp,
    Registers,
    StopOp,
    SU32Imm,
    UI5Imm,
    UI6Imm,
    UI10Imm,
    UI24Imm,
)
from qat.experimental.dialect.q1.ir.abstract_ops import Q1AsmOperation
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionPathConnectionAttr,
    ConnectionAttr,
    InputConfigAttr,
    ModuleConfigAttr,
    NcoConfigAttr,
    OutputConfigAttr,
    OutputPathConnectionAttr,
    SequencerConfigAttr,
    UnweightedAcquireConfigAttr,
    make_acquisition,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
)


@irdl_op_definition
class _NestedQ1AsmOp(Q1AsmOperation):
    name = "q1.test.nested"
    body = region_def("single_block")

    def __init__(self, *operations):
        super().__init__(regions=[Region(Block(operations))])

    def assembly_line_args(self):
        return ()


def _sequencer_config(
    readout: bool = False,
    port_id: str = "drive",
    output_id: int = 0,
    nco_frequency: float = 200e6,
) -> SequencerConfigAttr:
    return SequencerConfigAttr(
        port_id=port_id,
        carrier_frequency=200e6,
        connections=[
            ConnectionAttr(
                DirectionKind.io if readout else DirectionKind.output, [output_id]
            )
        ],
        output_path_connections=[OutputPathConnectionAttr(output_id)],
        acquisition_path_connections=(
            [AcquisitionPathConnectionAttr(output_id, SignalPath.i)] if readout else []
        ),
        disabled_outputs=[],
        disabled_acquisition_paths=[],
        nco=NcoConfigAttr(frequency=nco_frequency),
        unweighted_acquire=UnweightedAcquireConfigAttr(16) if readout else None,
    )


def _module_config(
    instrument_id: str = "cluster0",
    slot_idx: int = 2,
    kind: QbloxModuleKind = QbloxModuleKind.qcm,
    output_ids: tuple[int, ...] = (0,),
    input_ids: tuple[int, ...] | None = None,
) -> ModuleConfigAttr:
    readout = kind in (QbloxModuleKind.qrm, QbloxModuleKind.qrm_rf, QbloxModuleKind.qrc)
    input_ids = output_ids if input_ids is None else input_ids
    return ModuleConfigAttr(
        slot_idx,
        instrument_id,
        kind,
        outputs=[OutputConfigAttr(index) for index in output_ids],
        inputs=[InputConfigAttr(index) for index in input_ids] if readout else [],
    )


def _sequence(
    channel_id: str = "drive",
    instrument_id: str = "cluster0",
    slot_idx: int = 2,
    seq_idx: int = 0,
    readout: bool = False,
    output_id: int = 0,
    operations=None,
    waveforms=None,
    weights=None,
    acquisitions=None,
) -> SequenceOp:
    kind = QbloxModuleKind.qrm if readout else QbloxModuleKind.qcm
    return SequenceOp(
        channel_id,
        [StopOp()] if operations is None else operations,
        port_id=channel_id,
        waveforms=ArrayAttr([] if waveforms is None else waveforms),
        weights=ArrayAttr([] if weights is None else weights),
        acquisitions=ArrayAttr([] if acquisitions is None else acquisitions),
        instrument_id=instrument_id,
        slot_idx=slot_idx,
        seq_idx=seq_idx,
        sequencer_config=_sequencer_config(
            readout=readout, port_id=channel_id, output_id=output_id
        ),
        module_config=_module_config(
            instrument_id,
            slot_idx,
            kind=kind,
            output_ids=(output_id,),
        ),
    )


def test_emits_verified_sequence_as_json_payload_mappings():
    sequence = _sequence(
        "q0.drive.channel",
        waveforms=[make_waveform("drive", 2, [0.25, -0.5])],
    )

    program = emit_qblox_program(ModuleOp([sequence]), metadata={"revision": 1})

    assert tuple(program.packages) == ("q0.drive.channel",)
    package = program.packages["q0.drive.channel"]
    assert package.pulse_channel_id == "q0.drive.channel"
    assert package.physical_channel_id == "q0.drive.channel"
    assert package.seq_config.nco.freq == 200e6
    assert package.sequence.waveforms == {"drive": {"data": [0.25, -0.5], "index": 2}}
    assert package.sequence.weights == {}
    assert package.sequence.acquisitions == {}
    assert package.sequence.program == "stop\n"
    assert program.metadata == {"revision": 1}
    assert program.fw_version == TARGET_DATA.fw_version


def test_emits_readout_tables_without_loss():
    sequence = _sequence(
        "q0.readout.channel",
        readout=True,
        weights=[make_weight("integration", 3, [1.0, 0.5])],
        acquisitions=[make_acquisition("result", 4, 128)],
    )

    package = emit_qblox_program(ModuleOp([sequence])).packages[sequence.channel_id.data]

    assert package.sequence.weights["integration"] == {
        "index": 3,
        "data": [1.0, 0.5],
    }
    assert package.sequence.acquisitions["result"] == {
        "index": 4,
        "num_bins": 128,
    }
    assert package.seq_config.square_weight_acq.integration_length == 16
    assert package.seq_config.connection.acq_I == "in0"


def test_sequence_payload_shape_matches_qblox_driver_contract():
    sequence = _sequence(
        "q0.drive.channel",
        waveforms=[make_waveform("drive", 2, [0.25, -0.5])],
    )

    package = emit_qblox_program(ModuleOp([sequence])).packages["q0.drive.channel"]

    assert package.sequence == type(package.sequence)(
        program="stop\n",
        waveforms={"drive": {"data": [0.25, -0.5], "index": 2}},
        weights={},
        acquisitions={},
    )


def test_shared_payload_round_trips_through_executable_serialization():
    program = emit_qblox_program(ModuleOp([_sequence()]), metadata={"revision": 1})
    executable = Executable[QbloxProgram](programs=[program])

    restored = Executable[QbloxProgram].deserialize(executable.serialize())

    assert restored.programs == [program]


def test_rejects_duplicate_logical_package_key_before_dict_insertion():
    module = ModuleOp(
        [
            _sequence("drive", instrument_id="cluster0"),
            _sequence("drive", instrument_id="cluster1"),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate Qblox package key 'drive'"):
        emit_qblox_program(module)


def test_rejects_duplicate_physical_allocation():
    module = ModuleOp(
        [
            _sequence("drive"),
            _sequence("other"),
        ]
    )

    with pytest.raises(PassFailedException, match="Duplicate physical allocation"):
        emit_qblox_program(module)


def test_rejects_conflicting_module_configurations_on_shared_module():
    first = _sequence("first", seq_idx=0)
    second = _sequence("second", seq_idx=1)
    second.properties["module_config"] = _module_config(output_ids=(0, 1))

    with pytest.raises(PassFailedException, match="conflicting module configurations"):
        emit_qblox_program(ModuleOp([first, second]))


def test_rejects_missing_allocation_or_configuration():
    with pytest.raises(PassFailedException, match="missing its Qblox allocation"):
        emit_qblox_program(ModuleOp([SequenceOp("drive", [StopOp()])]))


def test_rejects_nested_q1asm_before_table_reference_validation():
    sequence = _sequence(
        operations=[
            _NestedQ1AsmOp(PlayImmImmImmOp(UI10Imm(1), UI10Imm(2), DurationImm(4))),
            StopOp(),
        ]
    )

    with pytest.raises(PassFailedException, match="requires flat Q1ASM"):
        emit_qblox_program(ModuleOp([sequence]))


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        pytest.param(
            PlayImmImmImmOp(UI10Imm(1), UI10Imm(2), DurationImm(4)),
            r"waveform indices \[1, 2\].*absent",
            id="waveforms",
        ),
        pytest.param(
            AcquireWeightedImmImmImmImmImmOp(
                UI5Imm(0), UI24Imm(0), UI6Imm(1), UI6Imm(2), DurationImm(4)
            ),
            r"weight indices \[1, 2\].*absent",
            id="weights",
        ),
        pytest.param(
            AcquireImmImmImmOp(UI5Imm(1), UI24Imm(0), DurationImm(4)),
            r"acquisition indices \[1\].*absent",
            id="acquisitions",
        ),
    ],
)
def test_rejects_missing_immediate_table_references(operation, expected):
    readout = not isinstance(operation, PlayImmImmImmOp)
    sequence = _sequence(
        readout=readout,
        operations=[operation, StopOp()],
        acquisitions=[make_acquisition("result", 0, 1)] if readout else [],
    )

    with pytest.raises(PassFailedException, match=expected):
        emit_qblox_program(ModuleOp([sequence]))


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        pytest.param(
            {"waveforms": [make_waveform("large", 0, [0.0] * 16_385)]},
            "16385 waveform samples",
            id="waveform",
        ),
        pytest.param(
            {"weights": [make_weight("large", 0, [0.0] * 16_385)]},
            "16385 weight samples",
            id="weight",
        ),
    ],
)
def test_rejects_tables_exceeding_sequencer_capacity(table, expected):
    sequence = _sequence(readout="weights" in table, **table)

    with pytest.raises(PassFailedException, match=expected):
        emit_qblox_program(ModuleOp([sequence]))


def test_rejects_aggregate_acquisition_bins_exceeding_module_capacity():
    shared_config = _module_config(kind=QbloxModuleKind.qrm)
    sequences = [
        _sequence(
            f"readout-{index}",
            seq_idx=index,
            readout=True,
            acquisitions=[make_acquisition("result", 0, bins)],
        )
        for index, bins in enumerate((1_500_000, 1_500_001))
    ]
    for sequence in sequences:
        sequence.properties["module_config"] = shared_config

    with pytest.raises(PassFailedException, match="3000001"):
        emit_qblox_program(ModuleOp(sequences))


def test_rejects_acquisition_instruction_on_qrc_control_sequencer():
    sequence = SequenceOp(
        "control",
        [AcquireImmImmImmOp(UI5Imm(0), UI24Imm(0), DurationImm(4)), StopOp()],
        port_id="control",
        acquisitions=ArrayAttr([make_acquisition("result", 0, 1)]),
        instrument_id="cluster0",
        slot_idx=2,
        seq_idx=8,
        sequencer_config=_sequencer_config(
            port_id="control",
            output_id=2,
        ),
        module_config=_module_config(
            kind=QbloxModuleKind.qrc,
            output_ids=(0, 1, 2, 3),
            input_ids=(0, 1),
        ),
    )

    with pytest.raises(PassFailedException, match="acquisition-capable sequencer"):
        emit_qblox_program(ModuleOp([sequence]))


def _register_play(index: int = 1):
    immediate = MoveImmRdOp(SU32Imm(index), Registers.R1)
    return immediate, PlayRsRsImmOp(immediate.rd, immediate.rd, DurationImm(4))


def test_rejects_aliased_register_table_reference():
    immediate = MoveImmRdOp(SU32Imm(1), Registers.R1)
    alias = MoveRsRdOp(immediate.rd, Registers.R2)
    second = MoveImmRdOp(SU32Imm(2), Registers.R3)
    sequence = _sequence(
        operations=[
            immediate,
            alias,
            second,
            PlayRsRsImmOp(alias.rd, second.rd, DurationImm(4)),
            StopOp(),
        ],
        waveforms=[
            make_waveform("first", 1, [0.25]),
            make_waveform("second", 2, [-0.5]),
        ],
    )

    with pytest.raises(ValueError, match="non-static register table reference"):
        emit_qblox_program(ModuleOp([sequence]))


def test_rejects_dynamically_computed_register_table_reference():
    immediate = MoveImmRdOp(SU32Imm(0), Registers.R1)
    dynamic = AddRsImmRdOp(immediate.rd, SU32Imm(1), Registers.R2)
    sequence = _sequence(
        operations=[
            immediate,
            dynamic,
            PlayRsRsImmOp(dynamic.rd, dynamic.rd, DurationImm(4)),
            StopOp(),
        ],
        waveforms=[make_waveform("first", 1, [0.25])],
    )

    with pytest.raises(ValueError, match="non-static register table reference"):
        emit_qblox_program(ModuleOp([sequence]))


@pytest.mark.parametrize(
    ("index", "waveforms", "expected"),
    [
        pytest.param(1024, [], "out-of-range waveform index 1024", id="range"),
        pytest.param(1, [], "missing waveform index 1", id="missing"),
    ],
)
def test_rejects_invalid_register_waveform_reference(index, waveforms, expected):
    immediate, play = _register_play(index)
    sequence = _sequence(
        operations=[immediate, play, StopOp()],
        waveforms=waveforms,
    )

    with pytest.raises(ValueError, match=expected):
        emit_qblox_program(ModuleOp([sequence]))


@pytest.mark.parametrize(
    ("indices", "weights", "expected"),
    [
        pytest.param((32, 0), [], r"out-of-range weight index 32", id="range"),
        pytest.param(
            (1, 2),
            [make_weight("first", 1, [0.25])],
            "missing weight index 2",
            id="missing",
        ),
    ],
)
def test_rejects_invalid_register_weight_reference(indices, weights, expected):
    bin_index = MoveImmRdOp(SU32Imm(0), Registers.R1)
    first = MoveImmRdOp(SU32Imm(indices[0]), Registers.R2)
    second = MoveImmRdOp(SU32Imm(indices[1]), Registers.R3)
    acquire = AcquireWeightedImmRsRsRsImmOp(
        UI5Imm(0), bin_index.rd, first.rd, second.rd, DurationImm(4)
    )
    sequence = _sequence(
        "readout",
        readout=True,
        operations=[bin_index, first, second, acquire, StopOp()],
        weights=weights,
        acquisitions=[make_acquisition("result", 0, 1)],
    )

    with pytest.raises(ValueError, match=expected):
        emit_qblox_program(ModuleOp([sequence]))


def test_accepts_static_register_waveform_references():
    immediate, play = _register_play()
    sequence = _sequence(
        operations=[immediate, play, StopOp()],
        waveforms=[make_waveform("pulse", 1, [0.25])],
    )

    emit_qblox_program(ModuleOp([sequence]))


def test_accepts_static_register_weight_references():
    bin_index = MoveImmRdOp(SU32Imm(0), Registers.R1)
    first = MoveImmRdOp(SU32Imm(1), Registers.R2)
    second = MoveImmRdOp(SU32Imm(2), Registers.R3)
    acquire = AcquireWeightedImmRsRsRsImmOp(
        UI5Imm(0), bin_index.rd, first.rd, second.rd, DurationImm(4)
    )
    sequence = _sequence(
        "readout",
        readout=True,
        operations=[bin_index, first, second, acquire, StopOp()],
        weights=[
            make_weight("first", 1, [0.25]),
            make_weight("second", 2, [-0.5]),
        ],
        acquisitions=[make_acquisition("result", 0, 1)],
    )

    emit_qblox_program(ModuleOp([sequence]))


def test_uses_target_data_waveform_capacity():
    control_data = TARGET_DATA.CONTROL_SEQUENCER_DATA.model_copy(
        update={"max_sample_size_waveforms": 1}
    )
    target_data = TARGET_DATA.model_copy(update={"CONTROL_SEQUENCER_DATA": control_data})
    sequence = _sequence(waveforms=[make_waveform("large", 0, [0.0, 0.0])])

    with pytest.raises(PassFailedException, match="2 waveform samples"):
        emit_qblox_program(ModuleOp([sequence]), target_data)


def test_rejects_program_exceeding_target_data_instruction_capacity():
    control_data = TARGET_DATA.CONTROL_SEQUENCER_DATA.model_copy(
        update={"max_num_instructions": 1}
    )
    target_data = TARGET_DATA.model_copy(update={"CONTROL_SEQUENCER_DATA": control_data})
    sequence = _sequence(operations=[MoveImmRdOp(SU32Imm(0), Registers.R1), StopOp()])

    with pytest.raises(PassFailedException, match="2 instructions"):
        emit_qblox_program(ModuleOp([sequence]), target_data)


def test_qrc_control_uses_control_sequencer_instruction_capacity():
    readout_data = TARGET_DATA.READOUT_SEQUENCER_DATA.model_copy(
        update={"max_num_instructions": 1}
    )
    control_data = TARGET_DATA.CONTROL_SEQUENCER_DATA.model_copy(
        update={"max_num_instructions": 2}
    )
    target_data = TARGET_DATA.model_copy(
        update={
            "CONTROL_SEQUENCER_DATA": control_data,
            "READOUT_SEQUENCER_DATA": readout_data,
        }
    )
    sequence = _sequence(
        seq_idx=8,
        operations=[MoveImmRdOp(SU32Imm(0), Registers.R1), StopOp()],
    )
    sequence.properties["sequencer_config"] = _sequencer_config(output_id=2)
    sequence.properties["module_config"] = _module_config(
        kind=QbloxModuleKind.qrc,
        output_ids=(2,),
        input_ids=(),
    )

    emit_qblox_program(ModuleOp([sequence]), target_data)

    readout_sequence = _sequence(
        readout=True,
        operations=[MoveImmRdOp(SU32Imm(0), Registers.R1), StopOp()],
    )
    readout_sequence.properties["module_config"] = _module_config(kind=QbloxModuleKind.qrc)
    with pytest.raises(PassFailedException, match="2 instructions.*readout sequencer"):
        emit_qblox_program(ModuleOp([readout_sequence]), target_data)


def test_rejects_nco_frequency_outside_target_data_range():
    control_data = TARGET_DATA.CONTROL_SEQUENCER_DATA.model_copy(
        update={"nco_min_freq": -100e6, "nco_max_freq": 100e6}
    )
    target_data = TARGET_DATA.model_copy(update={"CONTROL_SEQUENCER_DATA": control_data})
    sequence = _sequence()
    sequence.properties["sequencer_config"] = _sequencer_config(nco_frequency=200e6)

    with pytest.raises(PassFailedException, match="outside.*selected control sequencer"):
        emit_qblox_program(ModuleOp([sequence]), target_data)


def test_accepts_nco_frequency_inside_widened_target_data_range():
    control_data = TARGET_DATA.CONTROL_SEQUENCER_DATA.model_copy(
        update={"nco_min_freq": -600e6, "nco_max_freq": 600e6}
    )
    target_data = TARGET_DATA.model_copy(update={"CONTROL_SEQUENCER_DATA": control_data})
    sequence = _sequence()
    sequence.properties["sequencer_config"] = _sequencer_config(nco_frequency=550e6)

    emit_qblox_program(ModuleOp([sequence]), target_data)
