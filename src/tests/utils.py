# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import random
import re
from typing import List

import numpy as np

from qat.purr.backends.echo import EchoEngine
from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    Qubit,
    Resonator,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel


def apply_setup_to_hardware(hw, qubit_indices: list):
    """Apply the default echo hardware setup to the passed-in hardware."""
    qubit_devices = []
    resonator_devices = []
    channel_index = 1
    for primary_index in qubit_indices:
        bb1 = PhysicalBaseband(f"LO{channel_index}", 5.5e9)
        bb2 = PhysicalBaseband(f"LO{channel_index + 1}", 8.5e9)
        hw.add_physical_baseband(bb1, bb2)

        ch1 = PhysicalChannel(f"CH{channel_index}", 1.0e-9, bb1, 1)
        ch2 = PhysicalChannel(
            f"CH{channel_index + 1}", 1.0e-9, bb2, 1, acquire_allowed=True
        )
        hw.add_physical_channel(ch1, ch2)

        resonator = Resonator(f"R{primary_index}", ch2)
        resonator.create_pulse_channel(ChannelType.measure, frequency=8.5e9)
        resonator.create_pulse_channel(ChannelType.acquire, frequency=8.5e9)

        # As our main system is a ring architecture we just attach every qubit in the
        # ring to the one on either side.
        # 2 has a connection to 1 and 3. This number wraps around, so we also have
        # 10-0-1 linkages.
        qubit = Qubit(primary_index, resonator, ch1)
        qubit.create_pulse_channel(ChannelType.drive, frequency=5.5e9)

        qubit_devices.append(qubit)
        resonator_devices.append(resonator)
        channel_index = channel_index + 2

    # TODO: For backwards compatability cross resonance pulse channels are fully
    #   connected but coupled qubits are only in a ring architecture. I think it would be
    #   more approriate for cross resonace channels to also be a ring architecture but
    #   that can be done in a later PR.
    def _cross_channels(q1, q2):
        """Create cross resonance channels for q2 on q1."""
        try:
            q1.create_pulse_channel(
                auxiliary_devices=[q2],
                channel_type=ChannelType.cross_resonance,
                frequency=5.5e9,
                scale=50,
            )
            q1.create_pulse_channel(
                auxiliary_devices=[q2],
                channel_type=ChannelType.cross_resonance_cancellation,
                frequency=5.5e9,
                scale=0.0,
            )
        except KeyError:
            pass

    def _couple_qubits(q1, q2):
        def _directional_couple_qubits(q1, q2):
            if q2 in q1.coupled_qubits:
                return
            _cross_channels(q1, q2)
            q1.add_coupled_qubit(q2)

        _directional_couple_qubits(q1, q2)
        _directional_couple_qubits(q2, q1)

    for i, qubit in enumerate(qubit_devices):
        for other_qubit in qubit_devices:
            if qubit != other_qubit:
                _cross_channels(qubit, other_qubit)
        other_qubit = qubit_devices[(i + 1) % len(qubit_indices)]
        _couple_qubits(qubit, other_qubit)

    hw.add_quantum_device(*qubit_devices, *resonator_devices)
    hw.is_calibrated = True
    return hw


def get_jagged_echo_hardware(
    qubit_count: int = 4, qubit_indices: List[int] = []
) -> "QuantumHardwareModel":
    model = QuantumHardwareModel()
    if len(qubit_indices) == 0:
        qubit_indices = random.sample(range(1, qubit_count + 5), qubit_count)
    qubit_indices.sort()
    return apply_setup_to_hardware(model, qubit_indices)


def update_qubit_indices(program: str, qubit_indices: List[int]) -> str:
    """Adjust the physical qubit references in the program to use ones from qubit_indices."""
    patterns = [
        r"(\$)(?P<dollar_index>\d+)",
        r"(q)(?P<control>\d+)(_q)(?P<target>\d+)(_cross_resonance)",
        r"(q|r)(?P<index>\d+)(_drive|_acquire|_measure)",
    ]
    refs = []
    for pattern in patterns:
        refs.extend(list(re.finditer(pattern, program)))
    refs.sort(key=lambda r: r.start())
    new_program = ""
    old_end = 0
    for ref in refs:
        keys = ref.groupdict().keys()
        if "dollar_index" in keys:
            new_program = (
                new_program
                + program[old_end : ref.start()]
                + ref.group(1)
                + str(qubit_indices[int(ref.group("dollar_index"))])
            )
            old_end = ref.end()
        elif ("control" and "target") in keys:
            new_program = (
                new_program
                + program[old_end : ref.start()]
                + ref.group(1)
                + str(qubit_indices[int(ref.group("control"))])
                + ref.group(3)
                + str(qubit_indices[int(ref.group("target"))])
                + ref.group(5)
            )
            old_end = ref.end()
        elif "index" in keys:
            new_program = (
                new_program
                + program[old_end : ref.start()]
                + ref.group(1)
                + str(qubit_indices[int(ref.group("index"))])
                + ref.group(3)
            )
            old_end = ref.end()

    new_program += program[old_end:]
    return new_program


class ListReturningEngine(EchoEngine):
    """EchoEngine which is forced to return results in list format."""

    def _execute_on_hardware(self, *args, **kwargs):
        results = super()._execute_on_hardware(*args, **kwargs)
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results[k] = v.tolist()
        return results
