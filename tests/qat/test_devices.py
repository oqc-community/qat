# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.compiler.hardware_models import PhysicalBaseband, PhysicalChannel


class TestPulseChannel:
    @pytest.mark.parametrize("imbalance", [0, 1e-03, 1])
    @pytest.mark.parametrize("phase_offset", [0, 3.2e-03, 1.6])
    def test_default_pulse_channel(self, imbalance, phase_offset):
        bb = PhysicalBaseband(f"LO1", 5.5e9, if_frequency=250e6)
        phys_channel = PhysicalChannel(
            f"CH1", 1.0e-9, bb, 1, imbalance=imbalance, phase_offset=phase_offset
        )

        pulse_channel = phys_channel.create_pulse_channel("PulseCH1")
        # Default behaviour retrieves the imbalance/phase_offset from the physical channel.
        assert pulse_channel.imbalance == imbalance
        assert pulse_channel.imbalance == pulse_channel.physical_channel.imbalance
        assert pulse_channel.phase_offset == phase_offset
        assert pulse_channel.phase_offset == pulse_channel.physical_channel.phase_offset

        # Default behaviour assigns new imbalance/phase_offset to the physical channel.
        new_imbalance = imbalance + 1e-02
        new_phase_offset = phase_offset - 3e-05
        pulse_channel.imbalance = new_imbalance
        assert pulse_channel.imbalance == new_imbalance
        assert pulse_channel.imbalance != pulse_channel.physical_channel.imbalance

        pulse_channel.phase_offset = new_phase_offset
        assert pulse_channel.phase_offset == new_phase_offset
        assert pulse_channel.phase_offset != pulse_channel.physical_channel.phase_offset

    @pytest.mark.parametrize("imbalance", [1.0, -0.1, 3.0])
    def test_pulse_channel_imbalance(self, imbalance):
        bb = PhysicalBaseband(f"LO1", 5.5e9, if_frequency=250e6)
        phys_channel = PhysicalChannel(f"CH1", 1.0e-9, bb, 1, imbalance=1.1)
        pulse_channel = phys_channel.create_pulse_channel("PulseCH1", imbalance=imbalance)

        assert pulse_channel.imbalance != phys_channel.imbalance
        assert pulse_channel.imbalance == imbalance

        new_imbalance = imbalance + 1.0
        pulse_channel.imbalance = new_imbalance
        assert pulse_channel.imbalance == new_imbalance
        assert pulse_channel.imbalance != pulse_channel.physical_channel.imbalance

    @pytest.mark.parametrize("phase_offset", [0, -0.25, 1.0])
    def test_pulse_channel_phase_offset(self, phase_offset):
        bb = PhysicalBaseband(f"LO1", 5.5e9, if_frequency=250e6)
        phys_channel = PhysicalChannel(f"CH1", 1.0e-9, bb, 1, phase_offset=100)
        pulse_channel = phys_channel.create_pulse_channel(
            "PulseCH1", phase_offset=phase_offset
        )

        assert pulse_channel.phase_offset != phys_channel.phase_offset
        assert pulse_channel.phase_offset == phase_offset

        new_phase_offset = phase_offset + 1.0
        pulse_channel.phase_offset = new_phase_offset
        assert pulse_channel.phase_offset == new_phase_offset
        assert pulse_channel.phase_offset != pulse_channel.physical_channel.phase_offset
