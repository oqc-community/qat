# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import json
from pathlib import Path

import pytest

from qat.backend.qblox.target_data import QbloxTargetData
from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    CanonicalSystemData,
    ChannelData,
    ExternalResourceData,
    OscillatorData,
    PortData,
)
from qat.experimental.system_data.hardware.qblox import (
    MODULE_SPECS,
    QbloxHardwareView,
    QbloxModuleType,
    QbloxReadoutGenerator,
    QBloxReadoutSequencer,
    _module_type_from_id,
    _resolve_instrument_id,
    _sequencers_per_output,
    _slot_from_id,
)
from qat.experimental.system_data.hardware.qblox_config import (
    AcquireConfig,
    AwgConfig,
    MarkerSwitchConfig,
    NcoConfig,
    OutputSignalConfig,
)
from qat.experimental.system_data.materialisers.boundary import materialise

_CALIBRATION_DIR = Path(__file__).resolve().parents[4] / "files" / "calibrations"
_CALIBRATION_QBLOX = sorted(_CALIBRATION_DIR.glob("*.json"))[0]


@pytest.fixture(scope="module")
def canonical_data():
    """Materialise the canonical system data from the reference Qblox calibration."""
    source_payload = json.loads(_CALIBRATION_QBLOX.read_text())
    source_additional_data = {"target_data": QbloxTargetData()}
    return materialise(
        source_payload=source_payload,
        source_additional_data=source_additional_data,
    )


@pytest.fixture(scope="module")
def view(canonical_data):
    """Derive a Qblox hardware view from the materialised canonical data."""
    return QbloxHardwareView.derive(canonical_data)


class TestDeriveFromCalibration:
    """Derivation of a :class:`QbloxHardwareView` from a materialised calibration."""

    def test_load_canonical_hardware_config(self, canonical_data, view):
        # The acquire limit is drawn straight from the canonical system data.
        assert view.acquire_limit == canonical_data.acquire_limit

        # Modules are grouped by physical slot; the calibration has a QCM-RF and a QRM-RF.
        module_types = {module.module_type for module in view.modules}
        assert QbloxModuleType.QCM_RF in module_types
        assert QbloxModuleType.QRM_RF in module_types

        # Every canonical channel becomes exactly one single-frequency generator.
        assert len(view.generators) == len(canonical_data.channels)

    def test_generators_by_port_groups_every_generator(self, view):
        # The flattened generators are grouped by their output port.
        grouped = view.generators_by_port
        assert sum(len(gens) for gens in grouped.values()) == len(view.generators)
        for port_id, generators in grouped.items():
            assert all(generator.port_id == port_id for generator in generators)

    def test_modules_and_slot_indices_derive_from_canonical_data(self, view):
        for module in view.modules:
            # The Cluster slot is derived from the port baseband data.
            assert module.slot_idx is not None

            for generator in module.generators:
                assert generator.sequencer is not None
                # Generators on an RF module carry the connected local oscillator.
                if module.local_oscillators:
                    assert generator.local_oscillator is not None


class TestModuleResolutionErrors:
    """Error handling when module metadata cannot be resolved from canonical data."""

    def test_derive_raises_when_module_type_cannot_be_resolved(self):
        # A port whose identifier encodes no module type cannot be resolved.
        canonical = CanonicalSystemData(
            acquire_limit=10,
            ports=(PortData(id="mystery-port", sample_time=1000),),
        )
        with pytest.raises(ValueError, match="module type"):
            QbloxHardwareView.derive(canonical)

    def test_derive_raises_when_slot_index_cannot_be_resolved(self):
        # The module type resolves from the identifier, but the trailing slot segment does
        # not.
        canonical = CanonicalSystemData(
            acquire_limit=10,
            ports=(PortData(id="CH-QCM-RF-x", sample_time=1000),),
        )
        with pytest.raises(ValueError, match="slot index"):
            QbloxHardwareView.derive(canonical)


class TestSequencerRouting:
    """Routing of canonical channels onto the module's fixed sequencer set."""

    def test_module_sequencers_are_the_full_fixed_set_from_spec(self, view):
        for module in view.modules:
            spec = module.spec
            # The module exposes the full fixed sequencer set defined by its spec.
            assert len(module.sequencers) == (
                spec.number_of_control_sequencers + spec.number_of_readout_sequencers
            )
            # Every generator maps onto one of the module's sequencers.
            for generator in module.generators:
                assert generator.sequencer in module.sequencers

    def test_readout_ports_use_readout_sequencers(self, canonical_data, view):
        ports_by_id = {port.id: port for port in canonical_data.ports}

        for generator in view.generators:
            port = ports_by_id[generator.port_id]
            if port.acquire_allowed:
                # Acquisition ports use readout generators backed by readout sequencers.
                assert generator.sequencer.kind() == "readout"
                assert isinstance(generator, QbloxReadoutGenerator)
                assert isinstance(generator.sequencer, QBloxReadoutSequencer)
            else:
                assert generator.sequencer.kind() == "control"
                assert not isinstance(generator, QbloxReadoutGenerator)

    def test_qcm_rf_control_sequencers_route_to_outputs_by_default(self, view):
        qcm_module = next(
            module
            for module in view.modules
            if module.module_type == QbloxModuleType.QCM_RF
        )

        # Order the module's outputs by the order their generators first appear.
        ordered_ports: list[str] = []
        for generator in qcm_module.generators:
            if generator.port_id not in ordered_ports:
                ordered_ports.append(generator.port_id)
        # A QCM-RF exposes two outputs, each with its own local oscillator.
        assert len(ordered_ports) == 2

        # The six control sequencers split into two default routing blocks of three:
        # sequencers 0-2 drive the first output / LO and 3-5 the second.
        block = qcm_module.spec.number_of_control_sequencers // len(ordered_ports)
        assert block == 3

        output_local_oscillators = []
        for output_index, port_id in enumerate(ordered_ports):
            lo_ids = set()
            for generator in qcm_module.generators:
                if generator.port_id != port_id:
                    continue
                index = generator.sequencer.fields["index"]
                # Each output only drives sequencers from its default routing block.
                assert output_index * block <= index < (output_index + 1) * block
                assert generator.local_oscillator is not None
                lo_ids.add(generator.local_oscillator.id)
            # Every generator on an output shares that output's single local oscillator.
            assert len(lo_ids) == 1
            output_local_oscillators.append(lo_ids.pop())

        # The two outputs are driven by distinct local oscillators.
        assert len(set(output_local_oscillators)) == 2


class TestLocalOscillators:
    """Derivation and sharing of local oscillators."""

    def test_local_oscillators_derive_from_canonical_data(self, canonical_data, view):
        oscillator_frequencies = {
            osc.id: osc.frequency for osc in canonical_data.oscillators
        }
        channels_by_id = {channel.id: channel for channel in canonical_data.channels}

        for module in view.modules:
            for module_lo in module.local_oscillators:
                # Every derived local oscillator frequency comes from the canonical data.
                assert module_lo.frequency == oscillator_frequencies[module_lo.id]

        # A known drive channel's generator carries its channel's local oscillator.
        drive_channel = channels_by_id["Q0.drive"]
        drive_generators = [
            gen
            for gen in view.generators
            if gen.port_id == drive_channel.port_id
            and gen.local_oscillator is not None
            and gen.local_oscillator.id == drive_channel.oscillator_reference
        ]
        assert drive_generators
        expected_lo = oscillator_frequencies[drive_channel.oscillator_reference]
        assert drive_generators[0].local_oscillator.frequency == expected_lo

    def test_generators_sharing_an_oscillator_share_the_same_instance(
        self, canonical_data, view
    ):
        channels_by_id = {channel.id: channel for channel in canonical_data.channels}

        # Q0.drive and Q0.second_state share a single local oscillator on the same port.
        drive_channel = channels_by_id["Q0.drive"]
        second_channel = channels_by_id["Q0.second_state"]
        assert drive_channel.oscillator_reference == second_channel.oscillator_reference

        shared_generators = [
            gen
            for gen in view.generators
            if gen.port_id == drive_channel.port_id
            and gen.local_oscillator is not None
            and gen.local_oscillator.id == drive_channel.oscillator_reference
        ]
        assert len(shared_generators) >= 2
        # Sharing is implied by the generators mapping onto the same oscillator instance.
        first = shared_generators[0].local_oscillator
        assert all(gen.local_oscillator is first for gen in shared_generators)


class TestGeneratorConfiguration:
    """Configuration carried by generators and their sequencers."""

    def test_mixer_config_is_derived_from_channel_calibration(self, canonical_data, view):
        # Every generator's mixer correction matches one of the channel calibrations.
        expected_mixers = {
            (channel.phase_offset, channel.imbalance) for channel in canonical_data.channels
        }
        for generator in view.generators:
            mixer = generator.mixer_config
            assert (mixer.phase_offset, mixer.gain_ratio) in expected_mixers

    def test_readout_sequencer_carries_acquisition_and_control_config(
        self, canonical_data, view
    ):
        ports_by_id = {port.id: port for port in canonical_data.ports}

        for generator in view.generators:
            port = ports_by_id[generator.port_id]
            sequencer = generator.sequencer
            # All sequencers carry the inherited control-side configuration.
            assert sequencer.enable_sync is True
            assert sequencer.nco_config is not None
            assert sequencer.awg_config is not None
            if port.acquire_allowed:
                # Readout sequencers additionally carry the acquisition configuration.
                assert isinstance(sequencer, QBloxReadoutSequencer)
                assert sequencer.square_weight_acquire is not None
                assert sequencer.threshold_acquire_config is not None
                # The acquire config is default-constructed and only carries the
                # bin-increment and demodulation-enable flags.
                assert sequencer.acquire_config == AcquireConfig()
                assert sequencer.acquire_config.auto_bin_incr_en is None
                assert sequencer.acquire_config.demod_en_acq is None
                # Readout generators expose input and scope configuration.
                assert isinstance(generator, QbloxReadoutGenerator)
                assert generator.input_signal_config is not None
                assert generator.scope_acquire_config is not None

    def test_output_and_input_ids_derive_from_sequencer_connection(
        self, canonical_data, view
    ):
        for module in view.modules:
            for generator in module.generators:
                # The derived output is a valid physical output of the module.
                assert 0 <= generator.output_id < module.spec.number_of_outputs
                if isinstance(generator, QbloxReadoutGenerator):
                    # The derived input is a valid physical input of the module.
                    assert 0 <= generator.input_id < module.spec.number_of_inputs

    def test_output_id_is_consistent_per_local_oscillator(self, view):
        # A given local oscillator always pairs with a single output, so every generator
        # sharing that LO reports the same output.
        for module in view.modules:
            lo_to_output: dict[str, int] = {}
            for generator in module.generators:
                lo = generator.local_oscillator
                if lo is None:
                    continue
                assert lo_to_output.setdefault(lo.id, generator.output_id) == (
                    generator.output_id
                )

    def test_qcm_rf_distinct_oscillators_pair_with_distinct_outputs(self, view):
        # On a QCM-RF each of the two outputs has its own local oscillator, so the two
        # distinct LOs map to two distinct outputs.
        qcm_module = next(
            module
            for module in view.modules
            if module.module_type == QbloxModuleType.QCM_RF
        )
        lo_to_output = {
            generator.local_oscillator.id: generator.output_id
            for generator in qcm_module.generators
            if generator.local_oscillator is not None
        }
        assert len(lo_to_output) == 2
        assert len(set(lo_to_output.values())) == 2


class TestConfigDerivationFromAttributes:
    """Derivation of the new config dataclasses from a legacy ``QbloxConfig`` payload.

    The legacy config is materialised into the port external-resource attributes under
    ``baseband.config``; :meth:`QbloxHardwareView.derive` reads it back and populates the
    module- and sequencer-level configuration.
    """

    @staticmethod
    def _canonical_with_config(*, object_type, port_id, acquire_allowed, config):
        resource = ExternalResourceData(
            id="cluster-module",
            object_type=object_type,
            attributes=(
                AttributeEntry(
                    key="baseband",
                    value={
                        "slot_idx": 2,
                        "instrument_id": "clusterA",
                        "config": config,
                    },
                ),
            ),
        )
        oscillator = OscillatorData(id="lo", frequency=4_000_000_000)
        port = PortData(
            id=port_id,
            sample_time=1000,
            acquire_allowed=acquire_allowed,
            external_resource_id=resource.id,
        )
        channel = ChannelData(
            id="Q0.chan",
            port_id=port.id,
            frequency=4_240_000_000,
            oscillator_reference=oscillator.id,
        )
        return CanonicalSystemData(
            acquire_limit=5,
            oscillators=(oscillator,),
            ports=(port,),
            channels=(channel,),
            external_resources=(resource,),
        )

    def test_control_sequencer_and_output_config_derive_from_qblox_config(self):
        config = {
            "module": {
                "attenuation": {"out0": 10},
                "offset": {"out0_path0": 1.0, "out0_path1": 2.0},
            },
            "sequencers": {
                "0": {
                    "sync_en": False,
                    "marker_ovr_en": True,
                    "marker_ovr_value": 3,
                    "nco": {"phase_offs": 5.0, "prop_delay_comp_en": True},
                    "awg": {"gain_path0": 0.5, "mod_en": False},
                },
            },
        }
        canonical = self._canonical_with_config(
            object_type="QCM-RF",
            port_id="A-CH-QCM-RF-2",
            acquire_allowed=False,
            config=config,
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        generator = next(
            gen for gen in module.generators if gen.sequencer.fields["index"] == 0
        )

        assert generator.output_signal_config.attenuation == 10
        assert generator.output_signal_config.offset_path_0 == 1.0
        assert generator.output_signal_config.offset_path_1 == 2.0

        sequencer = generator.sequencer
        assert sequencer.enable_sync is False
        assert sequencer.nco_config.phase_offs == 5.0
        assert sequencer.nco_config.prop_delay_comp_en is True
        assert sequencer.awg_config.gain_path0 == 0.5
        assert sequencer.awg_config.mod_en is False
        assert sequencer.marker_switch_config.marker_ovr_en is True
        assert sequencer.marker_switch_config.marker_ovr_value == 3

    def test_readout_input_scope_and_acquisition_config_derive_from_qblox_config(self):
        config = {
            "module": {
                "attenuation": {"in0": 4},
                "gain": {"in0": 2},
                "offset": {"in0_path0": 0.5, "in0_path1": 0.25},
                "scope_acq": {"avg_mode_en_path0": False},
            },
            "sequencers": {
                "0": {
                    "demod_en_acq": True,
                    "square_weight_acq": {"integration_length": 2048},
                    "thresholded_acq": {"rotation": 30.0, "threshold": 1.5},
                    "ttl_acq": {"auto_bin_incr_en": True},
                },
            },
        }
        canonical = self._canonical_with_config(
            object_type="QRM-RF",
            port_id="A-CH-QRM-RF-14",
            acquire_allowed=True,
            config=config,
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        (generator,) = module.generators
        assert isinstance(generator, QbloxReadoutGenerator)

        assert generator.input_signal_config.attenuation == 4
        assert generator.input_signal_config.gain == 2
        assert generator.input_signal_config.offset_path_0 == 0.5
        assert generator.input_signal_config.offset_path_1 == 0.25
        assert generator.scope_acquire_config.enable_average_mode is False

        sequencer = generator.sequencer
        assert isinstance(sequencer, QBloxReadoutSequencer)
        assert sequencer.square_weight_acquire.integration_length == 2048
        assert sequencer.threshold_acquire_config.rotation == 30.0
        assert sequencer.threshold_acquire_config.threshold == 1.5
        assert sequencer.acquire_config.demod_en_acq is True
        assert sequencer.acquire_config.auto_bin_incr_en is True

    def test_missing_config_leaves_all_config_at_defaults(self):
        canonical = self._canonical_with_config(
            object_type="QCM-RF",
            port_id="A-CH-QCM-RF-2",
            acquire_allowed=False,
            config={},
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        (generator,) = module.generators

        assert generator.output_signal_config == OutputSignalConfig()
        assert generator.sequencer.enable_sync is True
        assert generator.sequencer.nco_config == NcoConfig()
        assert generator.sequencer.awg_config == AwgConfig()
        assert generator.sequencer.marker_switch_config == MarkerSwitchConfig()
        # With no connection routing the output falls back to the default routing block.
        assert generator.output_id == 0

    def test_output_id_derives_from_sequencer_connection(self):
        # The connection routes sequencer 0 to ``out1`` even though it is the only (and so
        # by default ``out0``) control output.
        config = {
            "module": {"attenuation": {"out1": 9}},
            "sequencers": {"0": {"connection": {"bulk_value": ["out1"]}}},
        }
        canonical = self._canonical_with_config(
            object_type="QCM-RF",
            port_id="A-CH-QCM-RF-2",
            acquire_allowed=False,
            config=config,
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        generator = next(
            gen for gen in module.generators if gen.sequencer.fields["index"] == 0
        )

        assert generator.output_id == 1
        # The output-signal conditioning follows the resolved output (``out1``).
        assert generator.output_signal_config.attenuation == 9

    def test_output_and_input_ids_derive_from_bulk_connection_for_readout(self):
        config = {
            "module": {},
            "sequencers": {"0": {"connection": {"bulk_value": ["out0", "in1"]}}},
        }
        canonical = self._canonical_with_config(
            object_type="QRM-RF",
            port_id="A-CH-QRM-RF-14",
            acquire_allowed=True,
            config=config,
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        (generator,) = module.generators

        assert isinstance(generator, QbloxReadoutGenerator)
        assert generator.output_id == 0
        assert generator.input_id == 1

    def test_generators_sharing_an_lo_share_the_same_output(self):
        # Two channels on one port share a single local oscillator, but their (synthetic)
        # sequencer allocation routes them to different outputs in the connection config.
        # The LO pairing must win: both generators report the LO's single output.
        config = {
            "module": {},
            "sequencers": {
                "0": {"connection": {"bulk_value": ["out0"]}},
                "1": {"connection": {"bulk_value": ["out1"]}},
            },
        }
        resource = ExternalResourceData(
            id="cluster-module",
            object_type="QCM-RF",
            attributes=(
                AttributeEntry(
                    key="baseband",
                    value={
                        "slot_idx": 2,
                        "instrument_id": "clusterA",
                        "config": config,
                    },
                ),
            ),
        )
        oscillator = OscillatorData(id="lo", frequency=4_000_000_000)
        port = PortData(
            id="A-CH-QCM-RF-2",
            sample_time=1000,
            external_resource_id=resource.id,
        )
        channels = tuple(
            ChannelData(
                id=channel_id,
                port_id=port.id,
                frequency=4_240_000_000,
                oscillator_reference=oscillator.id,
            )
            for channel_id in ("Q0.drive", "Q0.second_state")
        )
        canonical = CanonicalSystemData(
            acquire_limit=5,
            oscillators=(oscillator,),
            ports=(port,),
            channels=channels,
            external_resources=(resource,),
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules

        assert len(module.generators) == 2
        # Both generators share the one local oscillator and therefore one output.
        assert {gen.local_oscillator.id for gen in module.generators} == {oscillator.id}
        assert {gen.output_id for gen in module.generators} == {0}

    def test_input_id_falls_back_to_acquisition_lane_routing(self):
        # With no ``in*`` entry in ``bulk_value`` the acquisition lane routing is used.
        config = {
            "module": {},
            "sequencers": {"0": {"connection": {"bulk_value": ["out0"], "acq_I": "in1"}}},
        }
        canonical = self._canonical_with_config(
            object_type="QRM-RF",
            port_id="A-CH-QRM-RF-14",
            acquire_allowed=True,
            config=config,
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        (generator,) = module.generators

        assert generator.input_id == 1

    def test_input_id_defaults_to_output_id_when_connection_has_no_input_lane(self):
        # The readout input shares its LO with an output, so with no ``in*`` lane in the
        # connection the input pairs with the resolved output (``in1`` shares ``out1``'s LO).
        config = {
            "module": {},
            "sequencers": {"0": {"connection": {"bulk_value": ["out1"]}}},
        }
        canonical = self._canonical_with_config(
            object_type="QRM-RF",
            port_id="A-CH-QRM-RF-14",
            acquire_allowed=True,
            config=config,
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        (generator,) = module.generators

        assert generator.output_id == 1
        # The input shares the output's LO and so defaults to the same index.
        assert generator.input_id == 1
        # The adapter collapses the shared config to a reference stub on repeated ports; the
        # module still resolves its config from a sibling port that kept the full payload.
        full_config = {"module": {"attenuation": {"out0": 7}}, "sequencers": {}}
        stub_resource = ExternalResourceData(
            id="module-stub",
            object_type="QCM-RF",
            attributes=(
                AttributeEntry(
                    key="baseband",
                    value={
                        "slot_idx": 2,
                        "instrument_id": "clusterA",
                        "config": {"_adapter_reference": "mapping"},
                    },
                ),
            ),
        )
        full_resource = ExternalResourceData(
            id="module-full",
            object_type="QCM-RF",
            attributes=(
                AttributeEntry(
                    key="baseband",
                    value={
                        "slot_idx": 2,
                        "instrument_id": "clusterA",
                        "config": full_config,
                    },
                ),
            ),
        )
        ports = (
            PortData(
                id="A-CH-QCM-RF-2",
                sample_time=1000,
                external_resource_id=stub_resource.id,
            ),
            PortData(
                id="B-CH-QCM-RF-2",
                sample_time=1000,
                external_resource_id=full_resource.id,
            ),
        )
        channels = tuple(
            ChannelData(id=f"ch-{port.id}", port_id=port.id, frequency=4_640_000_000)
            for port in ports
        )
        canonical = CanonicalSystemData(
            acquire_limit=1,
            ports=ports,
            channels=channels,
            external_resources=(stub_resource, full_resource),
        )

        view = QbloxHardwareView.derive(canonical)
        (module,) = view.modules
        # Recovery succeeded when the non-stub output attenuation reaches a generator; the
        # two ports map to distinct outputs, so only the ``out0`` generator carries it.
        assert any(
            generator.output_signal_config.attenuation == 7
            for generator in module.generators
        )


class TestSyntheticDerivation:
    """Derivation from hand-built canonical data exercising the resolution paths."""

    def test_module_type_and_slot_resolve_from_external_resource(self):
        # A resource sets the module type via ``object_type`` and the slot/instrument via
        # its ``baseband`` attribute rather than through parseable identifiers.
        resource = ExternalResourceData(
            id="cluster-module",
            object_type="QRM_RF",
            attributes=(
                AttributeEntry(
                    key="baseband",
                    value={"slot_idx": 7, "instrument_id": "clusterA"},
                ),
            ),
        )
        oscillator = OscillatorData(id="ro-lo", frequency=6_000_000_000)
        port = PortData(
            id="readout-port",
            sample_time=500,
            acquire_allowed=True,
            external_resource_id=resource.id,
        )
        channel = ChannelData(
            id="Q0.measure",
            port_id=port.id,
            frequency=6_240_000_000,
            oscillator_reference=oscillator.id,
        )
        canonical = CanonicalSystemData(
            acquire_limit=5,
            oscillators=(oscillator,),
            ports=(port,),
            channels=(channel,),
            external_resources=(resource,),
        )

        view = QbloxHardwareView.derive(canonical)

        (module,) = view.modules
        assert module.module_type == QbloxModuleType.QRM_RF
        assert module.slot_idx == 7
        # The readout port yields a readout generator carrying its local oscillator.
        (generator,) = module.generators
        assert isinstance(generator, QbloxReadoutGenerator)
        assert generator.local_oscillator.id == oscillator.id

    def test_baseband_channel_has_no_local_oscillator(self):
        # A baseband QCM channel references no oscillator, so its generator has none.
        port = PortData(id="A-CH-QCM-3", sample_time=1000)
        channel = ChannelData(id="Q0.drive", port_id=port.id, frequency=100_000_000)
        canonical = CanonicalSystemData(
            acquire_limit=1,
            ports=(port,),
            channels=(channel,),
        )

        view = QbloxHardwareView.derive(canonical)

        (module,) = view.modules
        assert module.module_type == QbloxModuleType.QCM
        assert module.local_oscillators == ()
        (generator,) = module.generators
        assert generator.local_oscillator is None
        assert not isinstance(generator, QbloxReadoutGenerator)

    def test_ports_split_into_modules_by_instrument_and_slot(self):
        # Two resources share a slot index but sit on different instruments, so they form
        # two distinct modules keyed by ``(instrument_id, slot)``.
        resources = tuple(
            ExternalResourceData(
                id=f"module-{instrument}",
                object_type="QCM-RF",
                attributes=(
                    AttributeEntry(
                        key="baseband",
                        value={"slot_idx": 2, "instrument_id": instrument},
                    ),
                ),
            )
            for instrument in ("clusterA", "clusterB")
        )
        ports = tuple(
            PortData(
                id=f"port-{resource.id}",
                sample_time=1000,
                external_resource_id=resource.id,
            )
            for resource in resources
        )
        channels = tuple(
            ChannelData(id=f"ch-{port.id}", port_id=port.id, frequency=4_640_000_000)
            for port in ports
        )
        canonical = CanonicalSystemData(
            acquire_limit=1,
            ports=ports,
            channels=channels,
            external_resources=resources,
        )

        view = QbloxHardwareView.derive(canonical)

        assert len(view.modules) == 2
        assert all(module.module_type == QbloxModuleType.QCM_RF for module in view.modules)


class TestModuleSpecs:
    """Static module specifications and their sequencer construction."""

    @pytest.mark.parametrize("module_type", list(QbloxModuleType))
    def test_module_spec_registered_for_every_module_type(self, module_type):
        spec = MODULE_SPECS[module_type]
        assert spec.module_type == module_type

    @pytest.mark.parametrize("module_type", list(QbloxModuleType))
    def test_build_sequencers_matches_spec_counts_and_order(self, module_type):
        spec = MODULE_SPECS[module_type]
        sequencers = spec.build_sequencers()

        # The full fixed set is control sequencers followed by readout sequencers.
        assert len(sequencers) == (
            spec.number_of_control_sequencers + spec.number_of_readout_sequencers
        )
        control = sequencers[: spec.number_of_control_sequencers]
        readout = sequencers[spec.number_of_control_sequencers :]
        assert all(not isinstance(seq, QBloxReadoutSequencer) for seq in control)
        assert all(isinstance(seq, QBloxReadoutSequencer) for seq in readout)
        # Sequencers are indexed contiguously from zero.
        assert [seq.fields["index"] for seq in sequencers] == list(range(len(sequencers)))


class TestHelperFunctions:
    """Pure identifier/metadata resolution helpers."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A-CH-QCM-RF-2", QbloxModuleType.QCM_RF),
            ("A-LO-0-QRM-RF-14", QbloxModuleType.QRM_RF),
            ("something-QCM-3", QbloxModuleType.QCM),
            ("x-QRM-1", QbloxModuleType.QRM),
            ("y-QRC-0", QbloxModuleType.QRC),
            ("no-module-here", None),
            ("", None),
            (None, None),
        ],
    )
    def test_module_type_from_id(self, text, expected):
        assert _module_type_from_id(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A-CH-QCM-RF-2", 2),
            ("A-LO-0-QRM-RF-14", 14),
            ("CH-QCM-RF-x", None),
            ("", None),
            (None, None),
        ],
    )
    def test_slot_from_id(self, text, expected):
        assert _slot_from_id(text) == expected

    @pytest.mark.parametrize(
        ("sequencer_count", "output_count", "expected"),
        [
            (6, 2, 3),
            (8, 2, 4),
            (6, 4, 1),
            (6, 0, 6),
            (0, 0, 1),
        ],
    )
    def test_sequencers_per_output(self, sequencer_count, output_count, expected):
        assert _sequencers_per_output(sequencer_count, output_count) == expected

    def test_resolve_instrument_id_from_baseband(self):
        resource = ExternalResourceData(
            id="module",
            attributes=(
                AttributeEntry(key="baseband", value={"instrument_id": "clusterA"}),
            ),
        )
        assert _resolve_instrument_id(resource) == "clusterA"

    def test_resolve_instrument_id_returns_none_without_baseband(self):
        assert _resolve_instrument_id(None) is None
        assert _resolve_instrument_id(ExternalResourceData(id="module")) is None
