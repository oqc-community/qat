from typing import List

import numpy as np

from qat.purr.backends.live import LiveDeviceEngine
from qat.purr.backends.qblox.rnd.codegen import QbloxEmitter
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.emitter import InstructionEmitter, QatFile
from qat.purr.compiler.instructions import AcquireMode, Instruction, Repeat
from qat.purr.compiler.interrupt import NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


class QbloxLiveEngine(LiveDeviceEngine):
    def startup(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.connect()

    def shutdown(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.disconnect()

    def _generate_batch_repeats(self, repeat: Repeat):
        batch_size = self.model.repeat_limit
        if batch_size == -1:
            return [repeat]

        # Rebuild repeat list if the hardware can't support the current setup.
        if repeat.repeat_count <= self.model.repeat_limit:
            return [repeat]

        log.info(
            f"Running {repeat.repeat_count} shots at once is "
            f"unsupported on the current hardware. Batching execution."
        )
        quotient = repeat.repeat_count // batch_size
        remainder = repeat.repeat_count % batch_size
        batches = [Repeat(batch_size, repeat.repetition_period)] * quotient
        if remainder > 0:
            batches.append(Repeat(remainder, repeat.repetition_period))

        return batches

    def _common_execute(self, instructions, interrupt=NullInterrupt()):
        """Executes this qat file against this current hardware."""
        self._model_exists()

        with log_duration("QPU returned results in {} seconds."):
            # This is where we'd send the instructions off to the compiler for processing,
            # for now do ad-hoc processing.
            qat_file = InstructionEmitter().emit(instructions, self.model)
            batch_repeats = self._generate_batch_repeats(qat_file.repeat)

            results = {}
            for i, batch_repeat in enumerate(batch_repeats):
                metadata = {"batch_iteration": i}
                interrupt.if_triggered(metadata, throw=True)
                batch_results = self._execute_batch(qat_file, batch_repeat)
                results = self._accumulate_results(results, batch_results)

            # Process metadata assign/return values to make sure the data is in the right form.
            # TODO - prepare a (tweaked) qat file
            results = self._process_results(results, qat_file)
            results = self._process_assigns(results, qat_file)

            return results

    def _execute_batch(self, package: QatFile, batch_repeat: Repeat):
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")

        results = {}
        qblox_packages = QbloxEmitter(package).emit(package)
        self.model.control_hardware.set_data(qblox_packages)
        playback_results = self.model.control_hardware.start_playback()

        position_map = self.create_duration_timeline(package)
        aq_map = self.build_acquire_list(position_map)
        for channel, aqs in aq_map.items():
            if len(aqs) > 1:
                raise ValueError(
                    "Multiple acquisitions are not supported on the same channel in one sweep step"
                )

            for aq in aqs:
                physical_channel = aq.physical_channel
                dt = physical_channel.sample_time
                physical_channel.readout_start = aq.start * dt + aq.delay
                physical_channel.readout_length = aq.samples * dt
                physical_channel.acquire_mode_integrator = (
                    aq.mode == AcquireMode.INTEGRATOR
                )

            for aq in aqs:
                response = playback_results[aq.physical_channel.id]
                response_axis = get_axis_map(aq.mode, response)
                for pp in package.get_pp_for_variable(aq.output_variable):
                    response, response_axis = self.run_post_processing(
                        pp, response, response_axis
                    )

        return results
