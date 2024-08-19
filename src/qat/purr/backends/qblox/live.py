from typing import Dict, List

import numpy as np

from qat.ir.pass_base import InvokerMixin, PassManager, PassResultSet
from qat.purr.backends.analysis_passes import TriagePass
from qat.purr.backends.codegen import CodegenResultType
from qat.purr.backends.live import LiveDeviceEngine, LiveHardwareModel
from qat.purr.backends.live_devices import ControlHardware
from qat.purr.backends.optimisation_passes import (
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
    SweepDecomposition,
)
from qat.purr.backends.qblox.codegen import QbloxEmitter
from qat.purr.backends.qblox.fast.codegen import FastQbloxEmitter
from qat.purr.backends.utilities import get_axis_map
from qat.purr.backends.verification_passes import (
    RepeatSanitisationValidation,
    ReturnSanitisationValidation,
    ScopeSanitisationValidation,
)
from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import SweepIterator, _binary_average, _numpy_array_to_list
from qat.purr.compiler.instructions import AcquireMode, IndexAccessor, Instruction, Variable
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.compiler.runtime import NewQuantumRuntime
from qat.purr.utils.logging_utils import log_duration


class QbloxLiveHardwareModel(LiveHardwareModel):
    def __init__(self, control_hardware: ControlHardware = None):
        super().__init__(control_hardware)
        self._reverse_coercion()

    def create_engine(self, startup_engine: bool = True):
        return FastQbloxLiveEngine(self, startup_engine)

    def create_runtime(self, existing_engine=None):
        engine = existing_engine or self.create_engine()
        return NewQuantumRuntime(engine)

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
                    physical_channel.readout_start = aq.start * dt + aq.delay
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


class FastQbloxLiveEngine(LiveDeviceEngine, InvokerMixin):
    def build_pass_pipeline(self, *args, **kwargs):
        pipeline = PassManager()
        pipeline.add(SweepDecomposition())
        pipeline.add(RepeatSanitisation(self.model))
        pipeline.add(ScopeSanitisation())
        pipeline.add(ReturnSanitisation())
        pipeline.add(ScopeSanitisationValidation())
        pipeline.add(ReturnSanitisationValidation())
        pipeline.add(RepeatSanitisationValidation())
        pipeline.add(TriagePass())
        return pipeline

    def optimize(self, instructions):
        pass

    def validate(self, instructions: List[Instruction]):
        pass

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        self._model_exists()

        with log_duration("QPU returned results in {} seconds."):
            analyses = self.run_pass_pipeline(builder)
            packages = FastQbloxEmitter(analyses).emit_packages(builder)
            self.model.control_hardware.set_data(packages)
            playback_results: Dict[str, np.ndarray] = (
                self.model.control_hardware.start_playback(None, None)
            )

            # Post execution step needs a lot of work
            # TODO - Robust batching analysis (as a pass !)
            # TODO - Lowerability analysis pass
            # TODO - A generic loop nest model. Sth similar to SweepIterator but does not mixin injection stuff

            results = {}
            acquire_map = analyses.get_result(CodegenResultType.ACQUIRE_MAP)
            pp_map = analyses.get_result(CodegenResultType.PP_MAP)
            sweeps = analyses.get_result(CodegenResultType.SWEEPS)

            def create_sweep_iterator():
                switerator = SweepIterator()
                for sweep in sweeps:
                    switerator.add_sweep(sweep)
                return switerator

            for t, acquires in acquire_map.items():
                big_response = playback_results[t.physical_channel.id]
                switerator = create_sweep_iterator()
                sweep_splits = np.split(big_response, switerator.length)
                for acq in acquires:
                    switerator.reset_iteration()
                    while not switerator.is_finished():
                        switerator.do_sweep(
                            []
                        )  # just to advance iteration, no need for injection
                        response = sweep_splits[switerator.current_iteration]
                        response_axis = get_axis_map(acq.mode, response)
                        for pp in pp_map[acq.output_variable]:
                            response, response_axis = self.run_post_processing(
                                pp, response, response_axis
                            )
                            handle = results.setdefault(
                                acq.output_variable,
                                np.empty(
                                    switerator.get_results_shape(response.shape),
                                    response.dtype,
                                ),
                            )
                            switerator.insert_result_at_sweep_position(handle, response)

            results = self._process_results(results, analyses)
            results = self._process_assigns(results, analyses)

            return results

    def _process_results(self, results, analyses: PassResultSet):
        """
        Process any software-driven results transformation, such as taking a raw
        waveform result and turning it into a bit, or something else.
        """
        rp_map = analyses.get_result(CodegenResultType.RP_MAP)

        for inst in rp_map.values():
            target_values = results.get(inst.variable, None)
            if target_values is None:
                raise ValueError(f"Variable {inst.variable} not found in results output.")

            if (
                InlineResultsProcessing.Raw in inst.results_processing
                and InlineResultsProcessing.Binary in inst.results_processing
            ):
                raise ValueError(
                    f"Raw and Binary processing attempted to be applied "
                    f"to {inst.variable}. Only one should be selected."
                )

            # Strip numpy arrays if we're set to do so.
            if InlineResultsProcessing.NumpyArrays not in inst.results_processing:
                target_values = _numpy_array_to_list(target_values)

            # Transform to various formats if required.
            if InlineResultsProcessing.Binary in inst.results_processing:
                target_values = _binary_average(target_values)

            results[inst.variable] = target_values

        return results

    def _process_assigns(self, results, analyses: PassResultSet):
        """
        As assigns are classical instructions they are not processed as a part of the
        quantum execution (right now).
        Read through the results dictionary and perform the assigns directly, return the
        results.
        """

        def recurse_arrays(results_map, value):
            """Recurse through assignment lists and fetch values in sequence."""
            if isinstance(value, List):
                return [recurse_arrays(results_map, val) for val in value]
            elif isinstance(value, Variable):
                if value.name not in results_map:
                    raise ValueError(
                        f"Attempt to assign variable that doesn't exist {value.name}."
                    )

                if isinstance(value, IndexAccessor):
                    return results_map[value.name][value.index]
                else:
                    return results_map[value.name]
            else:
                return value

        assigns = analyses.get_result(CodegenResultType.ASSIGNS)
        ret_inst = analyses.get_result(CodegenResultType.RETURN)
        assigned_results = dict(results)
        for assign in assigns:
            assigned_results[assign.name] = recurse_arrays(assigned_results, assign.value)

        return {key: assigned_results[key] for key in ret_inst.variables}
