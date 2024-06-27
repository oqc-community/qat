# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import random
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.devices import (
    ChannelType,
    PhysicalBaseband,
    PhysicalChannel,
    Qubit,
    QubitCoupling,
    Resonator,
)
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import QuantumExecutionEngine, SweepIterator
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import AcquireMode, PostProcessing
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt


def apply_setup_to_hardware(
    hw, qubit_count: int = 4, connectivity: Optional[List[Tuple[int, int]]] = None
):
    """Apply the default echo hardware setup to the passed-in hardware."""
    qubit_devices = []
    resonator_devices = []
    channel_index = 1
    for primary_index in range(qubit_count):
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
    qubits_by_index = {qb.index: qb for qb in qubit_devices}
    if connectivity is None:
        for i, qubit in enumerate(qubit_devices):
            for other_qubit in qubit_devices:
                if qubit != other_qubit:
                    qubit.create_pulse_channel(
                        auxiliary_devices=[other_qubit],
                        channel_type=ChannelType.cross_resonance,
                        frequency=5.5e9,
                        scale=50,
                    )
                    qubit.create_pulse_channel(
                        auxiliary_devices=[other_qubit],
                        channel_type=ChannelType.cross_resonance_cancellation,
                        frequency=5.5e9,
                        scale=0.0,
                    )
                qubit.add_coupled_qubit(qubit_devices[(i + 1) % qubit_count])
                qubit.add_coupled_qubit(qubit_devices[(i - 1) % qubit_count])
    else:
        for connection in connectivity:
            left_index, right_index = connection
            qubit_left = qubits_by_index.get(left_index)
            qubit_right = qubits_by_index.get(right_index)
            qubit_right.create_pulse_channel(
                auxiliary_devices=[qubit_left],
                channel_type=ChannelType.cross_resonance,
                frequency=5.5e9,
                scale=50,
            )
            qubit_right.create_pulse_channel(
                auxiliary_devices=[qubit_left],
                channel_type=ChannelType.cross_resonance_cancellation,
                frequency=5.5e9,
                scale=0.0,
            )
            qubit_left.create_pulse_channel(
                auxiliary_devices=[qubit_right],
                channel_type=ChannelType.cross_resonance,
                frequency=5.5e9,
                scale=50,
            )
            qubit_left.create_pulse_channel(
                auxiliary_devices=[qubit_right],
                channel_type=ChannelType.cross_resonance_cancellation,
                frequency=5.5e9,
                scale=0.0,
            )
            qubit_left.add_coupled_qubit(qubit_right)
            qubit_right.add_coupled_qubit(qubit_left)

    hw.add_quantum_device(*qubit_devices, *resonator_devices)
    hw.is_calibrated = True

    return hw


def generate_connectivity(con_type, qubit_count):
    if con_type == Connectivity.Ring:
        if qubit_count == 1:
            return []
        if qubit_count == 2:
            return [(0, 1)]
        return [(i, (i + 1) % qubit_count) for i in range(qubit_count)]
    return None


def add_direction_couplings_to_hardware(model, connectivity):
    for edge in connectivity:
        model.qubit_direction_couplings.append(
            QubitCoupling(edge, quality=random.randrange(1, 20))
        )
    return model


class Connectivity(Enum):
    Ring = auto()


def get_default_echo_hardware(
    qubit_count=4,
    connectivity: Optional[Union[Connectivity, List[Tuple[int, int]]]] = None,
) -> "QuantumHardwareModel":
    """
    Generate a default echo backend optionally providing the type of connectivity. Either you pass a pre-defined connectivity as
    defined in the Connectivity enum or a specific connectivity list of which qubits connect to which.
    """
    model = QuantumHardwareModel()
    if isinstance(connectivity, Connectivity):
        connectivity = generate_connectivity(connectivity, qubit_count)

    return apply_setup_to_hardware(model, qubit_count, connectivity)


class EchoEngine(QuantumExecutionEngine):
    """
    A backend that just returns default values. Primarily used for testing and
    no-backend situations.
    """

    def run_calibrations(self, qubits_to_calibrate=None):
        pass

    def optimize(self, instructions):
        instructions = super().optimize(instructions)
        for instruction in instructions:
            if isinstance(instruction, PostProcessing):
                acq = instruction.acquire
                if acq.mode == AcquireMode.INTEGRATOR:
                    acq.mode = self.model.default_acquire_mode

        return instructions

    def _execute_on_hardware(
        self,
        sweep_iterator: SweepIterator,
        package: QatFile,
        interrupt: Interrupt = NullInterrupt(),
    ) -> Dict[str, np.ndarray]:
        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            metadata = {"sweep_iteration": sweep_iterator.get_current_sweep_iteration()}
            interrupt.if_triggered(metadata, throw=True)

            position_map = self.create_duration_timeline(package.instructions)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            repeats = package.repeat.repeat_count
            for channel_id, aqs in aq_map.items():
                for aq in aqs:
                    # just echo the output pulse back for now
                    response = buffers[aq.physical_channel.full_id()][
                        aq.start : aq.start + aq.samples
                    ]
                    if aq.mode != AcquireMode.SCOPE:
                        if repeats > 0:
                            response = np.tile(response, repeats).reshape((repeats, -1))

                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(
                            pp, response, response_axis
                        )

                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            sweep_iterator.get_results_shape(response.shape),
                            response.dtype,
                        ),
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)

        return results
