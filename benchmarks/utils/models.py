# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.purr.backends.echo import apply_setup_to_hardware
from qat.purr.backends.live import LiveDeviceEngine
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.hardware_models import QuantumHardwareModel


def get_mock_live_hardware(num_qubits):
    """
    Returns a general hardware model that is set to create a mock live execution engine.
    """
    return apply_setup_to_hardware(MockLiveHardwareModel(), num_qubits)


class MockLiveHardwareModel(QuantumHardwareModel):
    """
    A hardware model that will return a mock live execution engine to allow us to
    test and benchmark features exclusive to the live engine.
    """

    def create_engine(self):
        return MockQuantumExecution(self)


class MockQuantumExecution(LiveDeviceEngine):
    """
    An execution engine that
    """

    baseband_frequencies = {}
    buffers = {}

    def startup(self):
        pass

    def shutdown(self):
        pass

    def _execute_on_hardware(self, sweep_iterator, package: QatFile, interrupt=None):
        self.buffers = {}
        self.baseband_frequencies = {}

        results = {}
        increment = 0
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package.instructions)
            pulse_channel_buffers = self.build_pulse_channel_buffers(position_map, True)
            buffers = self.build_physical_channel_buffers(pulse_channel_buffers)
            baseband_freqs = self.build_baseband_frequencies(pulse_channel_buffers)
            aq_map = self.build_acquire_list(position_map)

            self.buffers[increment] = buffers
            self.baseband_frequencies[increment] = baseband_freqs
            for channel, aqs in aq_map.items():
                for aq in aqs:
                    dt = aq.physical_channel.sample_time
                    start = round(aq.start + aq.delay / dt)
                    response = self.buffers[increment][aq.physical_channel.full_id()][
                        start : start + aq.samples
                    ]

                    response_axis = get_axis_map(aq.mode, response)
                    for pp in package.get_pp_for_variable(aq.output_variable):
                        response, response_axis = self.run_post_processing(
                            pp, response, response_axis
                        )

                    var_result = results.setdefault(
                        aq.output_variable,
                        np.empty(
                            (sweep_iterator.get_results_shape(response.shape)),
                            response.dtype,
                        ),
                    )
                    sweep_iterator.insert_result_at_sweep_position(var_result, response)

            increment += 1
        return results
