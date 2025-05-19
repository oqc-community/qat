# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from functools import reduce
from itertools import groupby
from typing import Dict, List

from compiler_config.config import InlineResultsProcessing

from qat.purr.backends.live import LiveDeviceEngine, LiveHardwareModel
from qat.purr.backends.live_devices import ControlHardware
from qat.purr.backends.qblox.acquisition import Acquisition
from qat.purr.backends.qblox.algorithm import stable_partition
from qat.purr.backends.qblox.analysis_passes import (
    BindingPass,
    QbloxLegalisationPass,
    TILegalisationPass,
    TriagePass,
    TriageResult,
)
from qat.purr.backends.qblox.codegen import NewQbloxEmitter, QbloxEmitter
from qat.purr.backends.qblox.metrics_base import MetricsManager
from qat.purr.backends.qblox.pass_base import InvokerMixin, PassManager, QatIR
from qat.purr.backends.qblox.result_base import ResultManager
from qat.purr.backends.qblox.transform_passes import (
    RepeatSanitisation,
    ReturnSanitisation,
    ScopeSanitisation,
)
from qat.purr.backends.utilities import software_post_process_linear_map_complex_to_real
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.execution import (
    DeviceInjectors,
    _binary_average,
    _numpy_array_to_list,
)
from qat.purr.compiler.instructions import (
    AcquireMode,
    DeviceUpdate,
    IndexAccessor,
    Instruction,
    PostProcessType,
    Variable,
)
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.compiler.runtime import NewQuantumRuntime
from qat.purr.utils.logging_utils import log_duration


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


class AbstractQbloxLiveEngine(LiveDeviceEngine, InvokerMixin):
    def startup(self):
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")
        self.model.control_hardware.connect()

    def shutdown(self):
        if self.model.control_hardware is None:
            raise ValueError("Please add a control hardware first!")
        self.model.control_hardware.disconnect()

    def optimize(self, instructions):
        pass

    def validate(self, instructions: List[Instruction]):
        pass

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

    @staticmethod
    def combine_playbacks(playbacks: Dict[PulseChannel, List[Acquisition]]):
        """
        Combines acquisition objects from multiple acquire instructions in multiple readout targets.
        Notice that :meth:`groupby` preserves (original) relative order, which makes it honour
        the (sequential) lexicographical order of the loop nest:

        playback[target]["acq_0"] contains (potentially) a list of acquisitions collected in the same
        order as the order in which the packages were sent to the FPGA.

        Although acquisition names are enough for unicity in practice, the playback's structure
        distinguishes different (multiple) acquisitions per readout target, thus making it more robust.
        """

        playback: Dict[PulseChannel, Dict[str, Acquisition]] = {}
        for target, acquisitions in playbacks.items():
            groups_by_name = groupby(acquisitions, lambda acquisition: acquisition.name)
            playback[target] = {
                name: reduce(
                    lambda acq1, acq2: Acquisition.accumulate(acq1, acq2),
                    acqs,
                    Acquisition(),
                )
                for name, acqs in groups_by_name
            }

        return playback

    def process_playback(
        self, playback: Dict[PulseChannel, Dict[str, Acquisition]], res_mgr: ResultManager
    ):
        """
        Now that the combined playback is ready, we can compute and process results as required
        by customers. This requires loop nest information as well as post-processing and shaping
        requirements.
        """

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        acquire_map = triage_result.acquire_map
        sweeps = triage_result.sweeps
        pp_map = triage_result.pp_map
        loop_nest_shape = tuple(
            (list(len(next(iter(s.variables.values()))) for s in sweeps))
        ) or (1, -1)

        results = {}
        for target, acquisitions in playback.items():
            acquires = acquire_map[target]
            for name, acquisition in acquisitions.items():
                scope_data = acquisition.acq_data.scope
                integ_data = acquisition.acq_data.bins.integration
                thrld_data = acquisition.acq_data.bins.threshold

                acquire = next((inst for inst in acquires if inst.output_variable == name))
                if acquire.mode in [AcquireMode.SCOPE, AcquireMode.RAW]:
                    response = (scope_data.i.data + 1j * scope_data.q.data).reshape(
                        loop_nest_shape
                    )
                elif acquire.mode == AcquireMode.INTEGRATOR:
                    response = (integ_data.i + 1j * integ_data.q).reshape(loop_nest_shape)
                else:
                    raise ValueError(f"Unrecognised acquire mode {acquire.mode}")

                post_procs, axes = pp_map[name], {}
                for pp in post_procs:
                    if pp.process == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                        response, _ = software_post_process_linear_map_complex_to_real(
                            pp.args, response, axes
                        )
                    elif pp.process == PostProcessType.DISCRIMINATE:
                        # f: {0, 1}  --->   {-1, 1}
                        #      x    |--->   2x - 1
                        response = (2 * thrld_data - 1).reshape(loop_nest_shape)

                results[name] = response

        results = self._process_results(results, triage_result)
        results = self._process_assigns(results, triage_result)

        return results


class QbloxLiveEngine(AbstractQbloxLiveEngine):
    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation(self.model)
            | ReturnSanitisation()
            | TriagePass()
        )

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        self._model_exists()

        with log_duration("Codegen run in {} seconds."):
            res_mgr = ResultManager()
            met_mgr = MetricsManager()
            ir = QatIR(builder)
            self.run_pass_pipeline(ir, res_mgr, met_mgr)
            iter2packages = QbloxEmitter().emit_packages(ir, res_mgr, met_mgr)

        with log_duration("QPU returned results in {} seconds."):
            playbacks: Dict[PulseChannel, List[Acquisition]] = defaultdict(list)
            for packages in iter2packages.values():
                self.model.control_hardware.set_data(packages)
                payback: Dict[PulseChannel, List[Acquisition]] = (
                    self.model.control_hardware.start_playback(None, None)
                )
                for target, acquisitions in payback.items():
                    playbacks[target] += acquisitions

            playback: Dict[PulseChannel, Dict[str, Acquisition]] = self.combine_playbacks(
                playbacks
            )
            results = self.process_playback(playback, res_mgr)
            return results


class NewQbloxLiveEngine(AbstractQbloxLiveEngine):
    """
    Unlike vanilla QbloxLiveEngine, this engine does not use static iteration and injection mechanism.
    It leverages Q1's programming model to accelerate a handful of pulse-level algorithms.

    Not all algorithms fall within Q1's capabilities. While this remains an analysis issue we saw great
    flexibility in forking out this engine to allow R&D without compromising existing execution environment.
    Nevertheless, there should be a hybrid JIT-like reconciliation in the future that can run both IR features
    that still need static iteration while being able to accelerate programs whenever possible.

    See QbloxLiveEngine and QbloxLiveEngineAdapter.
    """

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation(self.model)
            | ScopeSanitisation()
            | ReturnSanitisation()
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
                met_mgr = MetricsManager()
                ir = QatIR(builder)
                self.run_pass_pipeline(ir, res_mgr, met_mgr)
                packages = NewQbloxEmitter().emit_packages(ir, res_mgr, met_mgr)
        finally:
            injectors.revert()

        with log_duration("QPU returned results in {} seconds."):
            self.model.control_hardware.set_data(packages)
            playbacks: Dict[PulseChannel, List[Acquisition]] = (
                self.model.control_hardware.start_playback(None, None)
            )

            playback: Dict[PulseChannel, Dict[str, Acquisition]] = self.combine_playbacks(
                playbacks
            )
            results = self.process_playback(playback, res_mgr)
            return results


class QbloxLiveEngineAdapter(LiveDeviceEngine):
    """
    A manual adapter of the new and legacy engines. Users can switch on and off HW acceleration by using
    the `enable_hax` flag.

    There should be a proper hybrid, dynamic, JIT-like engine in the future that can leverage the HW to accelerate
    programs whenever possible.
    """

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
