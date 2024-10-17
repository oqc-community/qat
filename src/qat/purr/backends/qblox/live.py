from typing import Dict

import numpy as np

from qat.purr.backends.live import LiveDeviceEngine, LiveHardwareModel
from qat.purr.backends.live_devices import ControlHardware
from qat.purr.backends.qblox.codegen import QbloxEmitter
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.compiler.interrupt import NullInterrupt


class QbloxLiveHardwareModel(LiveHardwareModel):
    def __init__(self, control_hardware: ControlHardware = None):
        super().__init__(control_hardware)
        self._reverse_coercion()

    def create_engine(self):
        return QbloxLiveEngine(self)

    def _reverse_coercion(self):
        """
        HwM pickler coerces primitive types into strings. This is a temporary
        workaround to support in-memory, pickled, and un-pickled dict key types
        of the qblox config
        """
        for channel in self.physical_channels.values():
            config = channel.config
            config.sequencers = {int(k): v for k, v in config.sequencers.items()}


class QbloxLiveEngine(LiveDeviceEngine):
    def startup(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.connect()

    def shutdown(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.disconnect()

    def _execute_on_hardware(
        self, sweep_iterator: SweepIterator, package: QatFile, interrupt=NullInterrupt()
    ):
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")

        results = {}
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            position_map = self.create_duration_timeline(package.instructions)

            qblox_packages = QbloxEmitter().emit(package)
            aq_map = self.build_acquire_list(position_map)
            self.model.control_hardware.set_data(qblox_packages)

            repetitions = package.repeat.repeat_count
            repetition_time = package.repeat.repetition_period

            for aqs in aq_map.values():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel in one sweep step"
                    )
                for aq in aqs:
                    physical_channel = aq.physical_channel
                    dt = physical_channel.sample_time
                    physical_channel.readout_start = aq.start * dt + (
                        aq.delay if aq.delay else 0.0
                    )
                    physical_channel.readout_length = aq.samples * dt
                    physical_channel.acquire_mode_integrator = (
                        aq.mode == AcquireMode.INTEGRATOR
                    )

            playback_results: Dict[str, np.ndarray] = (
                self.model.control_hardware.start_playback(
                    repetitions=repetitions, repetition_time=repetition_time
                )
            )

            for channel, aqs in aq_map.items():
                if len(aqs) > 1:
                    raise ValueError(
                        "Multiple acquisitions are not supported on the same channel in one sweep step"
                    )
                for aq in aqs:
                    response = playback_results[aq.output_variable]
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
