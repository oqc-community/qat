import random

import numpy as np
import pytest

from qat.model.builder import QuantumHardwareModelBuilder
from qat.model.hardware_model import QuantumHardwareModel


def build_hardware(n_qubits=4, connectivity=None):

    builder = QuantumHardwareModelBuilder()

    for q_index in range(n_qubits):

        baseband_r = builder.add_physical_baseband(
            frequency=np.random.uniform(1e05, 1e07),
            if_frequency=np.random.uniform(1e05, 1e07),
        )

        baseband_q = builder.add_physical_baseband(
            frequency=np.random.uniform(1e05, 1e07),
            if_frequency=np.random.uniform(1e05, 1e07),
        )

        physical_channel_r = builder.add_physical_channel(
            baseband=baseband_r,
            sample_time=np.random.uniform(1e-10, 1e-08),
            phase_iq_offset=np.random.uniform(0.0, 1.0),
            bias=np.random.uniform(-1.0, 1.0),
            acquire_allowed=True,
        )

        physical_channel_q = builder.add_physical_channel(
            baseband=baseband_q,
            sample_time=np.random.uniform(1e-10, 1e-08),
            phase_iq_offset=np.random.uniform(0.0, 1.0),
            bias=np.random.uniform(-1.0, 1.0),
            acquire_allowed=False,
        )

        resonator = builder.add_resonator(
            frequency=np.random.uniform(1e06, 200e06), physical_channel=physical_channel_r
        )

        qubit = builder.add_qubit(
            index=q_index,
            frequency=np.random.uniform(1e08, 5e09),
            physical_channel=physical_channel_q,
            measure_device=resonator,
        )

    builder.add_connectivity(connectivity=connectivity)

    return builder.model


def all_pairs(lst):
    if len(lst) < 2:
        yield []
        return
    if len(lst) % 2 == 1:
        # Handle odd length list
        for i in range(len(lst)):
            for result in all_pairs(lst[:i] + lst[i + 1 :]):
                yield result
    else:
        a = lst[0]
        for i in range(1, len(lst)):
            pair = (a, lst[i])
            for rest in all_pairs(lst[1:i] + lst[i + 1 :]):
                yield [pair] + rest


class Test_HW_Builder:
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 10, 20, 32, 100])
    def test_built_model_serialises(self, n_qubits):
        qubits = list(range(0, n_qubits))
        random.shuffle(qubits)

        connectivity = [
            (q1, q2)
            for q1, q2 in zip(qubits[: len(qubits) // 2], qubits[len(qubits) // 2 :])
        ]

        hw1 = build_hardware(n_qubits=n_qubits, connectivity=connectivity)
        hw2 = QuantumHardwareModel(**hw1.model_dump())

        hw1._deepequals(hw2)
