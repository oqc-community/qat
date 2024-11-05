from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from qat.model.component import make_refdict
from qat.model.device import PhysicalBaseband, PhysicalChannel, PulseChannel
from qat.model.hardware_model import QuantumHardwareModel


def make_Hardware(count=10, connections=3, seed=42):
    rng = np.random.default_rng(seed)
    pick = lambda L, size=3: make_refdict(*rng.choice(L, size=size))
    physical_basebands = [PhysicalBaseband() for _ in range(count)]
    physical_channels = [
        PhysicalChannel(baseband=list(pick(physical_basebands, 1).values())[0])
        for _ in range(count)
    ]
    pulse_channels = [
        PulseChannel(
            physical_channel=list(pick(physical_channels, 1).values())[0],
            auxiliary_qubits=[],
            some_val=1,
        )
        for _ in range(count)
    ]
    qubits = []  # finish this
    resonators = []  # finish this

    # for pulse_channel in pulse_channels:
    #    pulse_channel.auxiliary_qubits = list(pick(qubits, 3).values())

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
        pc.some_val = -1
        assert not O2._deepequals(O1)
