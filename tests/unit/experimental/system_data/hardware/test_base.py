# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import dataclass

import pytest

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    OscillatorData,
    PortData,
)
from qat.experimental.system_data.hardware.base import (
    Generator,
    HardwareViewInterface,
    LocalOscillator,
    Sequencer,
)


@dataclass(frozen=True, slots=True)
class DummySequencer(Sequencer):
    @classmethod
    def kind(cls) -> str:
        return "dummy"


@dataclass(frozen=True, slots=True)
class DummyConcreteHardwareView(HardwareViewInterface):
    acquire_limit: int
    generators: tuple[Generator, ...]

    @classmethod
    def derive(cls, parent: CanonicalSystemData, **_kwargs) -> "DummyConcreteHardwareView":
        sequencers_by_port = {
            port.id: (
                DummySequencer(
                    fields={
                        "sample_time": port.sample_time * port.block_size,
                    },
                    min_values={"if_frequency": -500e6},
                    max_values={"if_frequency": 500e6},
                ),
            )
            for port in parent.ports
        }
        local_oscillators = tuple(
            LocalOscillator(id=oscillator.id, frequency=oscillator.frequency)
            for oscillator in parent.oscillators
        )
        oscillators_by_port = {port.id: local_oscillators for port in parent.ports}
        generators = tuple(
            Generator(
                port_id=port.id,
                sample_time=port.sample_time * port.block_size,
                sequencers=sequencers_by_port.get(port.id, ()),
                local_oscillators=oscillators_by_port.get(port.id, ()),
            )
            for port in parent.ports
        )
        return cls(
            acquire_limit=parent.acquire_limit,
            generators=generators,
        )


@pytest.fixture
def canonical_core() -> CanonicalSystemData:
    return CanonicalSystemData(
        acquire_limit=50,
        ports=(
            PortData(id="p0", sample_time=4000, block_size=1),
            PortData(id="p1", sample_time=500, block_size=3),
        ),
    )


@pytest.fixture
def view_core(canonical_core: CanonicalSystemData) -> DummyConcreteHardwareView:
    return DummyConcreteHardwareView.derive(canonical_core)


@pytest.fixture
def canonical_with_oscillators() -> CanonicalSystemData:
    return CanonicalSystemData(
        ports=(
            PortData(id="p0", sample_time=4000),
            PortData(id="p1", sample_time=4000),
        ),
        oscillators=(
            OscillatorData(id="osc0", frequency=5_000_000_000),
            OscillatorData(id="osc1", frequency=6_000_000_000),
        ),
    )


@pytest.fixture
def view_with_oscillators(
    canonical_with_oscillators: CanonicalSystemData,
) -> DummyConcreteHardwareView:
    return DummyConcreteHardwareView.derive(canonical_with_oscillators)


class TestHardwareViewDerive:
    def test_returns_expected_types(self, view_core: DummyConcreteHardwareView):
        assert isinstance(view_core, HardwareViewInterface)
        assert isinstance(view_core, DummyConcreteHardwareView)

    def test_populates_acquire_limit(self, view_core: DummyConcreteHardwareView):
        assert view_core.acquire_limit == 50

    def test_maps_ports_to_generators(self, view_core: DummyConcreteHardwareView):
        assert len(view_core.generators) == 2
        assert view_core.generators[0].port_id == "p0"
        assert view_core.generators[1].port_id == "p1"

    def test_populates_generator_sample_times(self, view_core: DummyConcreteHardwareView):
        assert view_core.generators[0].sample_time == 4000
        assert view_core.generators[1].sample_time == 1500

    def test_populates_sequencer_sample_time_fields(
        self,
        view_core: DummyConcreteHardwareView,
    ):
        assert view_core.generators[0].sequencers[0].fields["sample_time"] == 4000
        assert view_core.generators[1].sequencers[0].fields["sample_time"] == 1500

    def test_attaches_oscillators_to_each_generator(
        self,
        view_with_oscillators: DummyConcreteHardwareView,
    ):
        assert len(view_with_oscillators.generators) == 2
        assert len(view_with_oscillators.generators[0].local_oscillators) == 2
        assert view_with_oscillators.generators[0].local_oscillators == (
            LocalOscillator(id="osc0", frequency=5_000_000_000),
            LocalOscillator(id="osc1", frequency=6_000_000_000),
        )
        assert view_with_oscillators.generators[1].local_oscillators == (
            LocalOscillator(id="osc0", frequency=5_000_000_000),
            LocalOscillator(id="osc1", frequency=6_000_000_000),
        )


def test_sequencer_allows_arbitrary_fields_and_optional_bounds():
    """Base sequencer contract supports arbitrary fields and optional bounds."""
    sequencer = DummySequencer(
        fields={"if_frequency": 240e6, "iq_bias": 0.01 + 0.02j, "label": "seq0"},
        min_values={"if_frequency": -500e6},
        max_values={"if_frequency": 500e6},
    )

    assert sequencer.kind() == "dummy"
    assert sequencer.fields["if_frequency"] == 240e6
    assert sequencer.fields["label"] == "seq0"
    assert sequencer.min_values["if_frequency"] == -500e6
    assert sequencer.max_values["if_frequency"] == 500e6
