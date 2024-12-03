from typing import Dict, List

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.backend.analysis_passes import (
    BindingPass,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.backend.transform_passes import (
    DesugaringPass,
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.ir.pass_base import InvokerMixin, PassManager, QatIR
from qat.ir.result_base import ResultManager
from qat.purr.backends.live import LiveDeviceEngine, LiveHardwareModel
from qat.purr.backends.live_devices import ControlHardware
from qat.purr.backends.qblox.analysis_passes import QbloxLegalisationPass
from qat.purr.backends.qblox.codegen import NewQbloxEmitter, QbloxEmitter
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import (
    DeviceInjectors,
    SweepIterator,
    _binary_average,
    _numpy_array_to_list,
)
from qat.purr.compiler.instructions import (
    AcquireMode,
    DeviceUpdate,
    IndexAccessor,
    Instruction,
    Variable,
)
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.compiler.runtime import NewQuantumRuntime
from qat.purr.utils.logging_utils import log_duration
from qat.utils.algorithm import stable_partition


class QbloxLiveHardwareModel(LiveHardwareModel):
    def __init__(self, control_hardware: ControlHardware = None):
        super().__init__(control_hardware)
        self._reverse_coercion()

    def create_engine(self, startup_engine: bool = True):
        return QbloxLiveEngineAdapter(self, startup_engine)

    def create_runtime(self, existing_engine=None):
        existing_engine = existing_engine or self.create_engine()
        return NewQuantumRuntime(existing_engine)

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

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        """
        Wrapper override that accepts an InstructionBuilder
        """

        return super()._common_execute(builder.instructions, interrupt)

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


class NewQbloxLiveEngine(LiveDeviceEngine, InvokerMixin):
    def startup(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.connect()

    def shutdown(self):
        if self.model.control_hardware is None:
            raise ValueError(f"Please add a control hardware first!")
        self.model.control_hardware.disconnect()

    def optimize(self, instructions):
        pass

    def validate(self, instructions: List[Instruction]):
        pass

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation()
            | ScopeSanitisation()
            | ReturnSanitisation()
            | DesugaringPass()
            | TriagePass()
            | BindingPass()
            | TILegalisationPass()
            | QbloxLegalisationPass()
        )

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        self._model_exists()

        # TODO - A skeptical usage of DeviceInjectors on static device updates
        # TODO - Figure out what they mean w/r to scopes and control flow
        static_dus, builder.instructions = stable_partition(
            builder.instructions,
            lambda inst: isinstance(inst, DeviceUpdate)
            and not isinstance(inst.value, Variable),
        )
        injectors = DeviceInjectors(static_dus)

        try:
            injectors.inject()
            with log_duration("Codegen run in {} seconds."):
                res_mgr = ResultManager()
                ir = QatIR(builder)
                self.run_pass_pipeline(ir, res_mgr, self.model)
                packages = NewQbloxEmitter().emit_packages(ir, res_mgr, self.model)

            with log_duration("QPU returned results in {} seconds."):
                self.model.control_hardware.set_data(packages)
                playback_results: Dict[str, np.ndarray] = (
                    self.model.control_hardware.start_playback(None, None)
                )

                # Post execution step needs a lot of work
                # TODO - Robust batching analysis (as a pass !)
                # TODO - Lowerability analysis pass

                triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
                acquire_map = triage_result.acquire_map
                pp_map = triage_result.pp_map
                sweeps = triage_result.sweeps

                def create_sweep_iterator():
                    switerator = SweepIterator()
                    for sweep in sweeps:
                        switerator.add_sweep(sweep)
                    return switerator

                results = {}
                for t, acquires in acquire_map.items():
                    switerator = create_sweep_iterator()
                    for acq in acquires:
                        big_response = playback_results[acq.output_variable]
                        sweep_splits = np.split(big_response, switerator.length)
                        switerator.reset_iteration()
                        while not switerator.is_finished():
                            # just to advance iteration, no need for injection
                            # TODO - A generic loop nest model. Sth similar to SweepIterator but does not mixin injection stuff
                            switerator.do_sweep([])
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

                results = self._process_results(results, triage_result)
                results = self._process_assigns(results, triage_result)

                return results
        finally:
            injectors.revert()

    def _process_results(self, results, triage_result: TriageResult):
        """
        Process any software-driven results transformation, such as taking a raw
        waveform result and turning it into a bit, or something else.
        """

        rp_map = triage_result.rp_map

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

    def _process_assigns(self, results, triage_result: TriageResult):
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

        assigns = triage_result.assigns
        ret_inst = next(iter(triage_result.returns))
        assigned_results = dict(results)
        for assign in assigns:
            assigned_results[assign.name] = recurse_arrays(assigned_results, assign.value)

        return {key: assigned_results[key] for key in ret_inst.variables}


class QbloxLiveEngineAdapter(LiveDeviceEngine):

    model: QbloxLiveHardwareModel

    def __init__(
        self,
        model: QbloxLiveHardwareModel,
        startup_engine: bool = True,
        enable_hax=False,
    ):
        super().__init__(model, startup_engine)
        self._legacy_engine = QbloxLiveEngine(model, False)
        self._new_engine = NewQbloxLiveEngine(model, False)
        self.enable_hax = enable_hax

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        if self.enable_hax:
            return self._new_engine._common_execute(builder, interrupt)
        return self._legacy_engine._common_execute(builder, interrupt)
