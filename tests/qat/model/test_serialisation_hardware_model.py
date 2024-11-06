from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from qat.model.component import make_refdict
from qat.model.device import (
    PhysicalBaseband,
    PhysicalChannel,
    PulseChannel,
    Qubit,
    Resonator,
)
from qat.model.hardware_model import VERSION, QuantumHardwareModel
from qat.purr.compiler.devices import ChannelType


def make_Hardware(count=10, connections=3, seed=42):
    rng = np.random.default_rng(seed)
    pick = lambda L, size=3: make_refdict(*rng.choice(L, size=size))
    pick_pulse_channels = lambda pulse_channels, physical_channel: make_refdict(
        *[
            pulse_ch
            for pulse_ch in pulse_channels
            if pulse_ch.physical_channel == physical_channel
        ]
    )
    physical_basebands = [
        PhysicalBaseband(
            frequency=np.random.uniform(1e05, 1e07),
            if_frequency=np.random.uniform(1e05, 1e07),
        )
        for _ in range(count)
    ]
    physical_channels = [
        PhysicalChannel(
            baseband=list(pick(physical_basebands, 1).values())[0],
            sample_time=np.random.uniform(1e-10, 1e-08),
            block_size=np.random.randint(1, 10),
            phase_iq_offset=np.random.uniform(0.0, 1.0),
            bias=np.random.uniform(-1.0, 1.0),
            acquire_allowed=rng.choice([True, False], size=1)[0],
        )
        for _ in range(count)
    ]
    pulse_channels = [
        PulseChannel(
            physical_channel=physical_channel,
            frequency=np.random.uniform(1e08, 1e10),
            bias=np.random.uniform(-1, 1) + 1.0j * np.random.uniform(-1, 1),
            scale=np.random.uniform(-10, 10) + 1.0j * np.random.uniform(-10, 10),
            fixed_if=rng.choice([True, False], size=1)[0],
            channel_type=rng.choice(list(ChannelType), size=1)[0],
            auxiliary_qubits=[],
        )
        for physical_channel in physical_channels
    ]
    resonators = [
        Resonator(
            physical_channel=physical_channel,
            pulse_channels=pick_pulse_channels(
                pulse_channels=pulse_channels, physical_channel=physical_channels[0]
            ),
            default_channel_type=rng.choice(list(ChannelType), size=1)[0],
        )
        for physical_channel in list(pick(physical_channels, count // 2).values())
    ]
    qubits = [
        Qubit(
            index=np.random.randint(0, count),
            drive_amp=np.random.uniform(-10, 10),
            physical_channel=physical_channel,
            pulse_channels=pick_pulse_channels(
                pulse_channels=pulse_channels, physical_channel=physical_channels[0]
            ),
            measure_device=list(pick(resonators, 1).values())[0],
            default_channel_type=rng.choice(list(ChannelType), size=1)[0],
            coupled_qubits=[],
        )
        for physical_channel in list(pick(physical_channels, count // 2).values())
    ]

    for pulse_channel in pulse_channels:
        pulse_channel.auxiliary_qubits = list(pick(qubits, 3).values())

    for qubit in qubits:
        qubit.coupled_qubits = [q for q in pick(qubits, 1).values() if qubit != q]

    return QuantumHardwareModel(
        physical_basebands=make_refdict(*physical_basebands),
        physical_channels=make_refdict(*physical_channels),
        pulse_channels=make_refdict(*pulse_channels),
        qubits=make_refdict(*qubits),
        resonators=make_refdict(*resonators),
    )


class Test_HW_Serialise:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_dump_load_eq(self, seed):
        O1 = make_Hardware(seed=seed)
        blob = O1.model_dump()
        O2 = QuantumHardwareModel(**blob)

        assert O1._deepequals(O2)

        O3 = make_Hardware(seed=6353234234)
        assert not O1._deepequals(O3)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_dump_eq(self, seed):
        O1 = make_Hardware(seed=seed)
        blob = O1.model_dump()

        O2 = QuantumHardwareModel(**blob)
        blob2 = O2.model_dump()

        O3 = make_Hardware(seed=6353234234)
        blob3 = O3.model_dump()

        assert blob == blob2
        assert blob != blob3

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_deep_equals(self, seed):
        O1 = make_Hardware(seed=seed)
        O2 = deepcopy(O1)

        assert O2._deepequals(O1)
        pc = list(O2.pulse_channels.values())[0]
        pc.frequency = -1
        assert not O2._deepequals(O1)

    def test_deserialise_version(self):
        O1 = make_Hardware()
        assert O1.version == VERSION

        O2 = QuantumHardwareModel(**O1.model_dump())
        assert O2.version == VERSION
