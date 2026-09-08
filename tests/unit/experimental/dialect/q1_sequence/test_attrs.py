# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import re

import pytest
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    Float64Type,
    IntAttr,
    IntegerType,
    NoneAttr,
    Signedness,
    StringAttr,
    VectorType,
)
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    AcquisitionPathConnectionAttr,
    ConnectionAttr,
    DirectionKindAttr,
    InputConfigAttr,
    LocalOscillatorConfigAttr,
    MixerCorrectionConfigAttr,
    ModuleConfigAttr,
    NcoConfigAttr,
    OutputConfigAttr,
    OutputPathConnectionAttr,
    SequencerConfigAttr,
    UnweightedAcquireConfigAttr,
    WaveformAttr,
    WeightAttr,
    f32,
    make_acquisition,
    make_module_config,
    make_sequencer_config,
    make_waveform,
    make_weight,
)
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    AcqTableIndex,
    BinCountImm,
    WaveformTableIndex,
    WeightTableIndex,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    SignalPath,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    Q1SequencerType,
)


class TestWaveformAttr:
    @pytest.mark.parametrize(
        ("element_type", "values", "should_fail"),
        [
            pytest.param(f32, [0.1, 0.2], False, id="f32-accepted"),
            pytest.param(Float64Type(), [0.1, 0.2], True, id="f64-rejected"),
            pytest.param(
                IntegerType(32, Signedness.SIGNED),
                [1, 2],
                True,
                id="i32-rejected",
            ),
        ],
    )
    def test_data_type(self, element_type, values, should_fail):
        vec = VectorType(element_type, [len(values)])
        data = DenseIntOrFPElementsAttr.from_list(vec, values)
        if should_fail:
            with pytest.raises(VerifyException):
                WaveformAttr(StringAttr("wf"), WaveformTableIndex(0), data)
        else:
            wf = WaveformAttr(StringAttr("wf"), WaveformTableIndex(0), data)
            wf.verify()

    @pytest.mark.parametrize(
        ("values", "should_fail"),
        [
            pytest.param([0.0, 0.5, -0.5], False, id="within-range"),
            pytest.param([-1.0, 1.0], False, id="boundary"),
            pytest.param([1.1], True, id="above-max"),
            pytest.param([-1.1], True, id="below-min"),
        ],
    )
    def test_data_range(self, values, should_fail):
        if should_fail:
            with pytest.raises(VerifyException, match="out of DAC range"):
                make_waveform("wf", 0, values)
        else:
            wf = make_waveform("wf", 0, values)
            wf.verify()


class TestWeightAttr:
    @pytest.mark.parametrize(
        ("element_type", "values", "should_fail"),
        [
            pytest.param(f32, [1.0, 0.0], False, id="f32-accepted"),
            pytest.param(Float64Type(), [1.0, 0.0], True, id="f64-rejected"),
            pytest.param(
                IntegerType(32, Signedness.SIGNED),
                [1, 2],
                True,
                id="i32-rejected",
            ),
        ],
    )
    def test_data_type(self, element_type, values, should_fail):
        vec = VectorType(element_type, [len(values)])
        data = DenseIntOrFPElementsAttr.from_list(vec, values)
        if should_fail:
            with pytest.raises(VerifyException):
                WeightAttr(StringAttr("w"), WeightTableIndex(0), data)
        else:
            w = WeightAttr(StringAttr("w"), WeightTableIndex(0), data)
            w.verify()

    @pytest.mark.parametrize(
        ("values", "should_fail"),
        [
            pytest.param([0.0, 0.5, -0.5], False, id="within-range"),
            pytest.param([-1.0, 1.0], False, id="boundary"),
            pytest.param([1.1], True, id="above-max"),
            pytest.param([-1.1], True, id="below-min"),
        ],
    )
    def test_data_range(self, values, should_fail):
        if should_fail:
            with pytest.raises(VerifyException, match="out of ADC range"):
                make_weight("w", 0, values)
        else:
            w = make_weight("w", 0, values)
            w.verify()


class TestAcquisitionAttr:
    def test_construction(self):
        a = AcquisitionAttr(StringAttr("a"), AcqTableIndex(5), BinCountImm(100))
        assert a.index.data == 5
        assert a.num_bins.data == 100

    def test_construction_via_helper(self):
        a = make_acquisition("acq", 3, 42)
        assert a.acquisition_name.data == "acq"
        assert a.index.data == 3
        assert a.num_bins.data == 42


class TestSequencerConfigAttr:
    def test_construction(self):
        config = SequencerConfigAttr(unweighted_acquire=UnweightedAcquireConfigAttr(1024))
        assert config.integration_length.data == 1024
        assert config.has_acquisition_config

    def test_construction_via_helper(self):
        config = make_sequencer_config(
            2048,
            port_id="q0/measure",
            carrier_frequency=6.24e9,
            connections=[ConnectionAttr(DirectionKind.output, [0])],
            output_path_connections=[OutputPathConnectionAttr(0, SignalPath.iq)],
            local_oscillator_id="lo0",
            enable_sync=True,
            nco=NcoConfigAttr(frequency=240e6),
        )

        assert config.port_id.data == "q0/measure"
        assert config.carrier_frequency.value.data == 6.24e9
        assert config.connections.data[0].connection == "out0"
        assert config.output_path_connections.data[0].path.data is SignalPath.iq
        assert config.local_oscillator_id.data == "lo0"
        assert bool(config.enable_sync.value.data)
        assert config.nco.frequency.value.data == 240e6
        assert config.integration_length.data == 2048

    def test_absent_values_are_none_attrs(self):
        config = SequencerConfigAttr()
        assert isinstance(config.nco, NoneAttr)
        assert isinstance(config.connections, NoneAttr)
        assert isinstance(config.output_path_connections, NoneAttr)
        assert isinstance(config.acquisition_path_connections, NoneAttr)
        assert isinstance(config.acquisition_enabled, NoneAttr)
        assert isinstance(config.disabled_outputs, NoneAttr)
        assert isinstance(config.disabled_acquisition_paths, NoneAttr)
        assert isinstance(config.acquisition_disabled, NoneAttr)
        assert config.integration_length is None
        assert not config.has_acquisition_config

    def test_empty_acquisition_collections_are_not_configuration(self):
        config = SequencerConfigAttr(
            acquisition_path_connections=ArrayAttr([]),
            disabled_acquisition_paths=ArrayAttr([]),
        )

        assert not config.has_acquisition_config

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "acquisition_path_connections": [
                    AcquisitionPathConnectionAttr(0, SignalPath.i)
                ]
            },
            {"disabled_acquisition_paths": [SignalPath.i]},
            {"acquisition_enabled": True},
            {"acquisition_enabled": False},
            {"acquisition_disabled": True},
            {"acquisition_disabled": False},
            {"unweighted_acquire": UnweightedAcquireConfigAttr(1024)},
        ],
    )
    def test_explicit_acquisition_state_is_configuration(self, kwargs):
        assert SequencerConfigAttr(**kwargs).has_acquisition_config

    def test_connections_and_configuration_are_typed(self):
        config = SequencerConfigAttr(
            port_id="q0/measure",
            carrier_frequency=6.24e9,
            connections=[
                ConnectionAttr(DirectionKind.output, [0, 1]),
                ConnectionAttr(DirectionKind.input, [0]),
                ConnectionAttr(DirectionKind.io, [2]),
            ],
            output_path_connections=[
                OutputPathConnectionAttr(0, SignalPath.i),
                OutputPathConnectionAttr(1, SignalPath.q),
                OutputPathConnectionAttr(2, SignalPath.i),
            ],
            acquisition_path_connections=[
                AcquisitionPathConnectionAttr(0, SignalPath.i),
                AcquisitionPathConnectionAttr(1, SignalPath.q),
            ],
            acquisition_enabled=True,
            disabled_outputs=[3],
            disabled_acquisition_paths=[SignalPath.iq],
            acquisition_disabled=False,
            local_oscillator_id="lo0",
            enable_sync=False,
            nco=NcoConfigAttr(frequency=240e6),
            mixer=MixerCorrectionConfigAttr(phase_offset=0.5, gain_ratio=1.1),
        )
        assert config.port_id.data == "q0/measure"
        assert config.carrier_frequency.value.data == 6.24e9
        assert [connection.connection for connection in config.connections] == [
            "out0_1",
            "in0",
            "io2",
        ]
        assert [
            (connection.output_id.data, connection.path.data)
            for connection in config.output_path_connections
        ] == [
            (0, SignalPath.i),
            (1, SignalPath.q),
            (2, SignalPath.i),
        ]
        assert [
            (connection.input_id.data, connection.path.data)
            for connection in config.acquisition_path_connections
        ] == [(0, SignalPath.i), (1, SignalPath.q)]
        assert bool(config.acquisition_enabled.value.data)
        assert [output.data for output in config.disabled_outputs] == [3]
        assert [path.data for path in config.disabled_acquisition_paths] == [SignalPath.iq]
        assert not bool(config.acquisition_disabled.value.data)
        assert config.local_oscillator_id.data == "lo0"
        assert not bool(config.enable_sync.value.data)
        assert config.nco.frequency.value.data == 240e6
        assert config.mixer.gain_ratio.value.data == 1.1

    def test_merge_bound_distinguishes_absent_and_empty_connections(self):
        resolved = SequencerConfigAttr(
            output_path_connections=[OutputPathConnectionAttr(0, SignalPath.iq)],
            acquisition_path_connections=[AcquisitionPathConnectionAttr(0, SignalPath.i)],
        )

        unresolved = SequencerConfigAttr().merge_bound(resolved)

        assert unresolved.output_path_connections == resolved.output_path_connections
        assert (
            unresolved.acquisition_path_connections == resolved.acquisition_path_connections
        )
        with pytest.raises(VerifyException, match="conflict at output_path_connections"):
            SequencerConfigAttr(
                output_path_connections=[], acquisition_path_connections=[]
            ).merge_bound(resolved)

    def test_merge_bound_preserves_equal_and_program_owned_nested_values(self):
        program = SequencerConfigAttr(
            port_id="q0/measure",
            nco=NcoConfigAttr(frequency=100e6),
            unweighted_acquire=UnweightedAcquireConfigAttr(1024),
        )
        bound = SequencerConfigAttr(
            port_id="q0/measure",
            nco=NcoConfigAttr(frequency=100e6, phase_offs=0.25),
            unweighted_acquire=UnweightedAcquireConfigAttr(2048),
        )

        merged = program.merge_bound(bound)

        assert merged.port_id == program.port_id
        assert merged.nco.frequency == program.nco.frequency
        assert merged.nco.phase_offs == bound.nco.phase_offs
        assert merged.unweighted_acquire == program.unweighted_acquire

    def test_with_unweighted_acquire_preserves_other_parameters(self):
        config = SequencerConfigAttr(
            port_id="q0/measure",
            nco=NcoConfigAttr(frequency=240e6),
            unweighted_acquire=UnweightedAcquireConfigAttr(1024),
        )

        updated = config.with_unweighted_acquire(UnweightedAcquireConfigAttr(2048))

        assert updated.integration_length.data == 2048
        assert updated.port_id == config.port_id
        assert updated.nco == config.nco

    def test_rejects_unaligned_integration_length(self):
        with pytest.raises(VerifyException, match="multiple of 4"):
            make_sequencer_config(integration_length=1023)

    def test_rejects_wrong_field_type(self):
        with pytest.raises(
            TypeError, match="Expected i1 IntegerAttr or boolean value, got IntAttr"
        ):
            NcoConfigAttr(prop_delay_comp_en=IntAttr(1))

    def test_rejects_empty_port_id(self):
        with pytest.raises(VerifyException, match="port_id must be non-empty"):
            SequencerConfigAttr(port_id="")

    @pytest.mark.parametrize(
        "connection_factory",
        [
            pytest.param(
                lambda: OutputPathConnectionAttr(-1, SignalPath.i),
                id="negative-output",
            ),
            pytest.param(
                lambda: AcquisitionPathConnectionAttr(-1, SignalPath.i),
                id="negative-input",
            ),
        ],
    )
    def test_rejects_negative_connection_port(self, connection_factory):
        with pytest.raises(VerifyException, match="must be non-negative"):
            connection_factory()

    @pytest.mark.parametrize(
        ("connection_factory", "expected"),
        [
            (lambda: ConnectionAttr(DirectionKind.output, []), "at least one"),
            (
                lambda: ConnectionAttr(DirectionKind.output, [-1]),
                "non-negative",
            ),
            (
                lambda: ConnectionAttr(DirectionKind.output, [0, 0]),
                "distinct",
            ),
            (
                lambda: ConnectionAttr(DirectionKind.input, [0, 1, 2]),
                "at most two",
            ),
            (
                lambda: ConnectionAttr(DirectionKind.output, [0, 1, 2]),
                "at most two",
            ),
            (
                lambda: ConnectionAttr(DirectionKind.io, [0, 1, 2]),
                "at most two",
            ),
        ],
    )
    def test_rejects_invalid_lossless_connections(self, connection_factory, expected):
        with pytest.raises(VerifyException, match=expected):
            connection_factory()

    def test_connection_accepts_typed_direction_and_port_array(self):
        connection = ConnectionAttr(
            DirectionKindAttr(DirectionKind.io),
            ArrayAttr([IntAttr(0), IntAttr(1)]),
        )

        assert connection.connection == "io0_1"

    def test_acquisition_connection_requires_path(self):
        with pytest.raises(TypeError, match="path is required"):
            AcquisitionPathConnectionAttr(0, None)

    def test_rejects_duplicate_acquisition_paths(self):
        with pytest.raises(VerifyException, match="duplicate acquisition"):
            SequencerConfigAttr(
                acquisition_path_connections=[
                    AcquisitionPathConnectionAttr(0, SignalPath.i),
                    AcquisitionPathConnectionAttr(1, SignalPath.i),
                ]
            )

    def test_rejects_multiple_paths_for_one_physical_output(self):
        with pytest.raises(VerifyException, match="same physical output"):
            SequencerConfigAttr(
                output_path_connections=[
                    OutputPathConnectionAttr(0, SignalPath.i),
                    OutputPathConnectionAttr(0, SignalPath.q),
                ]
            )

    def test_rejects_connected_and_disabled_output(self):
        with pytest.raises(VerifyException, match="both connected and disabled"):
            SequencerConfigAttr(
                output_path_connections=[OutputPathConnectionAttr(0, SignalPath.iq)],
                disabled_outputs=[0],
            )

    def test_rejects_overlapping_bulk_output_connections(self):
        with pytest.raises(VerifyException, match="overlapping output"):
            SequencerConfigAttr(
                connections=[
                    ConnectionAttr(DirectionKind.output, [0]),
                    ConnectionAttr(DirectionKind.output, [0, 1]),
                ]
            )

    def test_rejects_bulk_connected_and_disabled_output(self):
        with pytest.raises(VerifyException, match="both connected and disabled"):
            SequencerConfigAttr(
                connections=[ConnectionAttr(DirectionKind.output, [0])],
                disabled_outputs=[0],
            )

    def test_rejects_connected_and_disabled_acquisition_path(self):
        with pytest.raises(VerifyException, match="both connected and disabled"):
            SequencerConfigAttr(
                acquisition_path_connections=[
                    AcquisitionPathConnectionAttr(0, SignalPath.i)
                ],
                disabled_acquisition_paths=[SignalPath.i],
            )

    def test_rejects_disabled_acquisition_with_input_connection(self):
        with pytest.raises(VerifyException, match="cannot be disabled"):
            SequencerConfigAttr(
                connections=[ConnectionAttr(DirectionKind.input, [0])],
                acquisition_disabled=True,
            )

    def test_rejects_enabled_and_disabled_acquisition(self):
        with pytest.raises(VerifyException, match="both enabled and disabled"):
            SequencerConfigAttr(
                acquisition_enabled=True,
                acquisition_disabled=True,
            )

    @pytest.mark.parametrize("frequency", [-500e6, 0.0, 500e6])
    def test_accepts_nco_frequency_in_hardware_range(self, frequency):
        NcoConfigAttr(frequency=frequency).verify()

    @pytest.mark.parametrize("frequency", [-500e6 - 1, 500e6 + 1])
    def test_rejects_nco_frequency_outside_hardware_range(self, frequency):
        with pytest.raises(VerifyException, match="outside"):
            NcoConfigAttr(frequency=frequency)


class TestModuleConfigAttr:
    def test_construction(self):
        config = ModuleConfigAttr(
            7,
            "cluster0",
            QbloxModuleKind.qrm_rf,
            [OutputConfigAttr(0)],
            [InputConfigAttr(0)],
            [LocalOscillatorConfigAttr("lo0", 6_000_000_000)],
        )
        assert config.slot_idx.data == 7
        assert config.kind.data is QbloxModuleKind.qrm_rf
        assert DEFAULT_QBLOX_TARGET.module_spec(config.kind.data).output_count == 1
        assert config.outputs.data[0].output_id.data == 0
        assert config.inputs.data[0].input_id.data == 0

    def test_empty_module_defaults_to_no_lanes(self):
        config = ModuleConfigAttr(1, "cluster0", QbloxModuleKind.qcm)
        assert len(config.outputs) == 0
        assert len(config.inputs) == 0
        assert len(config.local_oscillators) == 0

    def test_construction_via_helper(self):
        config = make_module_config(
            7,
            "cluster0",
            QbloxModuleKind.qrm_rf,
            outputs=[OutputConfigAttr(0)],
            inputs=[InputConfigAttr(0)],
            local_oscillators=[LocalOscillatorConfigAttr("lo0", 6_000_000_000)],
        )

        assert config.slot_idx.data == 7
        assert config.instrument_id.data == "cluster0"
        assert config.kind.data is QbloxModuleKind.qrm_rf

    def test_rejects_empty_module_identity(self):
        with pytest.raises(VerifyException, match="instrument_id"):
            ModuleConfigAttr(1, "", QbloxModuleKind.qcm)

    @pytest.mark.parametrize(
        ("lane", "expected"),
        [
            pytest.param(OutputConfigAttr, "output_id must be non-negative", id="out"),
            pytest.param(InputConfigAttr, "input_id must be non-negative", id="in"),
        ],
    )
    def test_rejects_negative_lane_identity(self, lane, expected):
        with pytest.raises(VerifyException, match=expected):
            lane(-1)

    def test_rejects_empty_oscillator_identity(self):
        with pytest.raises(VerifyException, match="non-empty oscillator_id"):
            LocalOscillatorConfigAttr("", 4_000_000_000)

    def test_rejects_duplicate_oscillator_identities(self):
        oscillator = LocalOscillatorConfigAttr("lo0", 4_000_000_000)

        with pytest.raises(VerifyException, match="duplicate local oscillator"):
            ModuleConfigAttr(
                1,
                "cluster0",
                QbloxModuleKind.qcm_rf,
                local_oscillators=[oscillator, oscillator],
            )

    @pytest.mark.parametrize(
        ("kind", "outputs", "inputs", "expected"),
        [
            pytest.param(
                QbloxModuleKind.qcm,
                [OutputConfigAttr(4)],
                [],
                "output_id 4",
                id="qcm-output",
            ),
            pytest.param(
                QbloxModuleKind.qcm_rf,
                [OutputConfigAttr(2)],
                [],
                "output_id 2",
                id="qcm-rf-output",
            ),
            pytest.param(
                QbloxModuleKind.qrm,
                [],
                [InputConfigAttr(2)],
                "input_id 2",
                id="qrm-input",
            ),
            pytest.param(
                QbloxModuleKind.qrm_rf,
                [],
                [InputConfigAttr(1)],
                "input_id 1",
                id="qrm-rf-input",
            ),
            pytest.param(
                QbloxModuleKind.qrc,
                [OutputConfigAttr(6)],
                [],
                "output_id 6",
                id="qrc-output",
            ),
            pytest.param(
                QbloxModuleKind.qrc,
                [],
                [InputConfigAttr(2)],
                "input_id 2",
                id="qrc-input",
            ),
        ],
    )
    def test_rejects_lane_invalid_for_module_kind(self, kind, outputs, inputs, expected):
        with pytest.raises(VerifyException, match=f"{expected} is invalid"):
            ModuleConfigAttr(1, "cluster0", kind, outputs, inputs)

    @pytest.mark.parametrize(
        ("outputs", "inputs", "expected"),
        [
            pytest.param(
                [OutputConfigAttr(0), OutputConfigAttr(0)], [], "output_id", id="outputs"
            ),
            pytest.param(
                [], [InputConfigAttr(0), InputConfigAttr(0)], "input_id", id="inputs"
            ),
        ],
    )
    def test_rejects_duplicate_lanes(self, outputs, inputs, expected):
        with pytest.raises(VerifyException, match=f"duplicate {expected}"):
            ModuleConfigAttr(1, "cluster0", QbloxModuleKind.qrm, outputs, inputs)


class TestModuleSpecs:
    @pytest.mark.parametrize("kind", list(QbloxModuleKind))
    def test_spec_registered_for_every_module_kind(self, kind):
        module_spec = DEFAULT_QBLOX_TARGET.module_spec(kind)
        assert module_spec.kind is kind
        assert module_spec.sequencer_count == len(module_spec.sequencers)

    @pytest.mark.parametrize(
        ("kind", "control", "readout"),
        [
            pytest.param(QbloxModuleKind.qcm, list(range(6)), [], id="qcm"),
            pytest.param(QbloxModuleKind.qrm, [], list(range(6)), id="qrm"),
            pytest.param(
                QbloxModuleKind.qrc,
                list(range(8, 12)),
                list(range(8)),
                id="qrc",
            ),
        ],
    )
    def test_sequencer_types_follow_hardware_layout(self, kind, control, readout):
        module_spec = DEFAULT_QBLOX_TARGET.module_spec(kind)
        assert list(module_spec.sequencer_indices(Q1SequencerType.control)) == control
        assert list(module_spec.sequencer_indices(Q1SequencerType.readout)) == readout


class TestIoConnectionLanes:
    """Qblox ``ioX_Y`` drives and acquires on every lane it names."""

    @pytest.mark.parametrize(
        ("direction", "port_ids", "outputs", "inputs"),
        [
            (DirectionKind.output, [0, 1], (0, 1), ()),
            (DirectionKind.input, [1], (), (1,)),
            (DirectionKind.input, [0, 1], (), (0, 1)),
            (DirectionKind.io, [0, 1], (0, 1), (0, 1)),
            (DirectionKind.io, [2], (2,), (2,)),
        ],
    )
    def test_connection_lanes_follow_their_direction(
        self, direction, port_ids, outputs, inputs
    ):
        connection = ConnectionAttr(direction, port_ids)

        assert connection.output_ids == outputs
        assert connection.input_ids == inputs

    def test_io_lanes_do_not_collide_with_unrelated_lanes(self):
        config = SequencerConfigAttr(
            connections=[
                ConnectionAttr(DirectionKind.io, [0, 1]),
                ConnectionAttr(DirectionKind.output, [2]),
                ConnectionAttr(DirectionKind.input, [3]),
            ]
        )

        config.verify()

    @pytest.mark.parametrize(
        ("other", "expected"),
        [
            (
                ConnectionAttr(DirectionKind.output, [0]),
                "overlapping output connections: [0]",
            ),
            (
                ConnectionAttr(DirectionKind.output, [1]),
                "overlapping output connections: [1]",
            ),
            (
                ConnectionAttr(DirectionKind.input, [0]),
                "overlapping input connections: [0]",
            ),
            (
                ConnectionAttr(DirectionKind.input, [1]),
                "overlapping input connections: [1]",
            ),
        ],
    )
    def test_io_lanes_collide_with_either_direction_of_every_lane(self, other, expected):
        with pytest.raises(VerifyException, match=re.escape(expected)):
            SequencerConfigAttr(
                connections=[ConnectionAttr(DirectionKind.io, [0, 1]), other]
            )
