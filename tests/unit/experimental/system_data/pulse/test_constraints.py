# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests the :class:`PulseLevelConstraints` class and its associated methods."""

import pytest

from qat.experimental.dialect.pulse.ir import (
    GaussianWaveformOp,
    SinWaveformOp,
    SquareWaveformOp,
)
from qat.experimental.system_data.canonical.schema import CanonicalSystemData, PortData
from qat.experimental.system_data.pulse.constraints import (
    PortConstraints,
    PulseLevelConstraints,
)


class TestPortConstraints:
    """Tests the access methods on the :class:`PortConstraints` class."""

    def test_sample_time_in_seconds_gives_correct_value(self):
        """Tests that the sample time in seconds is correctly calculated."""
        constraints = PortConstraints(
            sample_time_ps=8000,
            min_duration_ps=0,
            max_duration_ps=None,
            native_waveform_shapes=(),
            acquire_allowed=True,
        )
        assert constraints.sample_time_s == 8e-9

    @pytest.mark.parametrize("min_duration_ps, min_duration_s", [(0, 0.0), (8000, 8e-9)])
    def test_min_pulse_duration_in_seconds_gives_correct_value(
        self, min_duration_ps, min_duration_s
    ):
        """Tests that the minimum pulse duration in seconds is correctly calculated."""
        constraints = PortConstraints(
            sample_time_ps=0,
            min_duration_ps=min_duration_ps,
            max_duration_ps=None,
            native_waveform_shapes=(),
            acquire_allowed=True,
        )
        assert constraints.min_pulse_duration_s == min_duration_s

    @pytest.mark.parametrize(
        "max_duration_ps, max_duration_s", [(None, None), (8000, 8e-9)]
    )
    def test_max_pulse_duration_in_seconds_gives_correct_value(
        self, max_duration_ps, max_duration_s
    ):
        """Tests that the maximum pulse duration in seconds is correctly calculated."""
        constraints = PortConstraints(
            sample_time_ps=0,
            min_duration_ps=0,
            max_duration_ps=max_duration_ps,
            native_waveform_shapes=(),
            acquire_allowed=True,
        )
        assert constraints.max_pulse_duration_s == max_duration_s

    @pytest.mark.parametrize(
        "waveform_shape, response",
        [
            (SquareWaveformOp, True),
            (GaussianWaveformOp, True),
            (SinWaveformOp, False),
        ],
    )
    def test_supports_waveform_shape_gives_correct_value(self, waveform_shape, response):
        """Tests that the supports_waveform_shape method gives the correct value."""
        constraints = PortConstraints(
            sample_time_ps=0,
            min_duration_ps=0,
            max_duration_ps=None,
            native_waveform_shapes=(SquareWaveformOp, GaussianWaveformOp),
            acquire_allowed=True,
        )
        assert constraints.supports_waveform_shape(waveform_shape) == response


class TestPulseLevelConstraints:
    """Tests the access methods on the :class:`PulseLevelConstraints` class."""

    def test_granularity_in_seconds_gives_correct_value(self):
        """Tests that the granularity in seconds is correctly calculated."""
        constraints = PulseLevelConstraints(
            ports={},
            granularity_ps=8000,
        )
        assert constraints.granularity_s == 8e-9


class TestBuildPulseLevelConstraints:
    """Tests the ``PulseLevelConstraints.from_canonical`` class method."""

    @pytest.fixture
    def canonical_data(self):
        """Returns canonical data with two ports."""

        port_1 = PortData(
            id="port_1",
            sample_time=1000,
            block_size=8,
            min_blocks=1,
            max_blocks=10,
            native_waveform_shapes=("square", "gaussian"),
            acquire_allowed=True,
        )
        port_2 = PortData(
            id="port_2",
            sample_time=500,
            block_size=16,
            min_blocks=1,
            max_blocks=-1,  # No limit on duration
            native_waveform_shapes=("square",),
            acquire_allowed=False,
        )

        return CanonicalSystemData(
            ports=(port_1, port_2),
        )

    def test_building_gives_the_correct_granularity(self, canonical_data):
        """Tests that the granularity is correctly calculated from canonical data."""
        pulse_constraints = PulseLevelConstraints.from_canonical(canonical_data)
        assert pulse_constraints.granularity_ps == 8000
        assert pulse_constraints.granularity_s == 8e-9

    def test_building_gives_correct_sample_times(self, canonical_data):
        """Tests that the sample times are correctly calculated from canonical data."""
        pulse_constraints = PulseLevelConstraints.from_canonical(canonical_data)
        assert pulse_constraints.ports["port_1"].sample_time_ps == 1000
        assert pulse_constraints.ports["port_2"].sample_time_ps == 500
        assert pulse_constraints.ports["port_1"].sample_time_s == 1e-9
        assert pulse_constraints.ports["port_2"].sample_time_s == 5e-10

    def test_building_gives_correct_min_durations(self, canonical_data):
        """Tests that the minimum durations are correctly calculated from canonical data."""
        pulse_constraints = PulseLevelConstraints.from_canonical(canonical_data)
        assert pulse_constraints.ports["port_1"].min_duration_ps == 8000
        assert pulse_constraints.ports["port_2"].min_duration_ps == 8000
        assert pulse_constraints.ports["port_1"].min_pulse_duration_s == 8e-9
        assert pulse_constraints.ports["port_2"].min_pulse_duration_s == 8e-9

    def test_building_gives_correct_max_durations(self, canonical_data):
        """Tests that the maximum durations are correctly calculated from canonical data."""
        pulse_constraints = PulseLevelConstraints.from_canonical(canonical_data)
        assert pulse_constraints.ports["port_1"].max_duration_ps == 80000
        assert pulse_constraints.ports["port_2"].max_duration_ps is None
        assert pulse_constraints.ports["port_1"].max_pulse_duration_s == 8e-8
        assert pulse_constraints.ports["port_2"].max_pulse_duration_s is None

    def test_building_gives_correct_native_waveform_shapes(self, canonical_data):
        """Tests that native waveform shapes are correctly calculated from canonical
        data."""
        pulse_constraints = PulseLevelConstraints.from_canonical(canonical_data)
        assert pulse_constraints.ports["port_1"].native_waveform_shapes == (
            SquareWaveformOp,
            GaussianWaveformOp,
        )
        assert pulse_constraints.ports["port_2"].native_waveform_shapes == (
            SquareWaveformOp,
        )

    def test_building_gives_correct_acquire_allowed(self, canonical_data):
        """Tests that acquire_allowed is correctly calculated from canonical data."""
        pulse_constraints = PulseLevelConstraints.from_canonical(canonical_data)
        assert pulse_constraints.ports["port_1"].acquire_allowed is True
        assert pulse_constraints.ports["port_2"].acquire_allowed is False


class TestBuildPulseLevelConstraintsErrors:
    """Tests the ``PulseLevelConstraints.from_canonical`` error cases."""

    def test_different_granularities_raises_value_error(self):
        """Tests that a ValueError is raised if ports have different granularities."""
        port_1 = PortData(
            id="port_1",
            sample_time=1000,
            block_size=8,
            min_blocks=1,
            max_blocks=10,
            native_waveform_shapes=("square", "gaussian"),
            acquire_allowed=True,
        )
        port_2 = PortData(
            id="port_2",
            sample_time=1000,
            block_size=16,
            min_blocks=1,
            max_blocks=-1,
            native_waveform_shapes=("square",),
            acquire_allowed=False,
        )
        canonical_data = CanonicalSystemData(ports=(port_1, port_2))
        with pytest.raises(ValueError, match="The port port_2 has a different granularity"):
            PulseLevelConstraints.from_canonical(canonical_data)

    def test_no_ports_raises_value_error(self):
        """Tests that a ValueError is raised if there are no ports in canonical data."""
        canonical_data = CanonicalSystemData(ports=())
        with pytest.raises(ValueError, match="No ports were found in the canonical data"):
            PulseLevelConstraints.from_canonical(canonical_data)

    def test_unknown_waveform_shape_raises_value_error(self):
        """Tests that a ValueError is raised if a port has an unknown waveform shape."""
        port = PortData(
            id="port_1",
            sample_time=1000,
            block_size=8,
            min_blocks=1,
            max_blocks=10,
            native_waveform_shapes=("banana",),
            acquire_allowed=True,
        )
        canonical_data = CanonicalSystemData(ports=(port,))
        with pytest.raises(ValueError, match="unknown waveform shape"):
            PulseLevelConstraints.from_canonical(canonical_data)

    def test_duplicate_port_ids_raises_value_error(self):
        """Tests that a ValueError is raised if duplicate port IDs are present."""
        port_1 = PortData(
            id="port_1",
            sample_time=1000,
            block_size=8,
            min_blocks=1,
            max_blocks=10,
            native_waveform_shapes=("square",),
            acquire_allowed=True,
        )
        port_2 = PortData(
            id="port_1",
            sample_time=1000,
            block_size=8,
            min_blocks=1,
            max_blocks=10,
            native_waveform_shapes=("square",),
            acquire_allowed=True,
        )
        canonical_data = CanonicalSystemData(ports=(port_1, port_2))
        with pytest.raises(ValueError, match="is defined multiple times"):
            PulseLevelConstraints.from_canonical(canonical_data)
