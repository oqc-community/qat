# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from functools import reduce
from itertools import groupby
from typing import Dict, List

from compiler_config.config import InlineResultsProcessing

from qat.backend.passes.purr.analysis import TriageResult
from qat.backend.qblox.acquisition import Acquisition
from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.visualisation import plot_playback, plot_program
from qat.engines import NativeEngine
from qat.instrument.base import InstrumentConcept
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.backends.utilities import (
    software_post_process_linear_map_complex_to_real,
)
from qat.purr.compiler.execution import _binary_average, _numpy_array_to_list
from qat.purr.compiler.instructions import (
    AcquireMode,
    IndexAccessor,
    PostProcessType,
    ProcessAxis,
    Variable,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class QbloxEngine(NativeEngine[QbloxProgram]):
    def __init__(self, instrument: InstrumentConcept, model: QbloxLiveHardwareModel):
        self.instrument: InstrumentConcept = instrument
        self.model: QbloxLiveHardwareModel = model

        self.plot_program = False
        self.dump_program = False
        self.plot_playback = False

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
    def combine_playbacks(playbacks: Dict[str, List[Acquisition]]):
        """
        Combines acquisition objects from multiple acquire instructions in multiple readout targets.
        Notice that :meth:`groupby` preserves (original) relative order, which makes it honour
        the (sequential) lexicographical order of the loop nest:

        playback[target]["acq_0"] contains (potentially) a list of acquisitions collected in the same
        order as the order in which the packages were sent to the FPGA.

        Although acquisition names are enough for unicity in practice, the playback's structure
        distinguishes different (multiple) acquisitions per readout target, thus making it more robust.
        """

        playback: Dict[str, Dict[str, Acquisition]] = {}
        for pulse_channel_id, acquisitions in playbacks.items():
            groups_by_name = groupby(acquisitions, lambda acquisition: acquisition.name)
            playback[pulse_channel_id] = {
                name: reduce(
                    lambda acq1, acq2: acq1 + acq2,
                    acqs,
                    Acquisition(),
                )
                for name, acqs in groups_by_name
            }

        return playback

    def process_playback(
        self,
        playback: Dict[str, Dict[str, Acquisition]],
        triage_result: TriageResult,
    ):
        """
        Now that the combined playback is ready, we can compute and process results as required
        by customers. This requires loop nest information as well as post-processing and shaping
        requirements.
        """

        acquire_map = triage_result.acquire_map
        sweeps = triage_result.sweeps
        repeats = triage_result.repeats
        pp_map = triage_result.pp_map
        sweep_counts = list(len(next(iter(s.variables.values()))) for s in sweeps)
        repeat_counts = list(r.repeat_count for r in repeats)

        results = {}
        for pulse_channel_id, acquisitions in playback.items():
            target = self.model.get_pulse_channel_from_id(pulse_channel_id)
            acquires = acquire_map[target]
            for name, acquisition in acquisitions.items():
                scope_data = acquisition.acquisition.scope
                integ_data = acquisition.acquisition.bins.integration
                thrld_data = acquisition.acquisition.bins.threshold

                acquire = next((inst for inst in acquires if inst.output_variable == name))
                if acquire.mode in [AcquireMode.SCOPE, AcquireMode.RAW]:
                    repeat_counts.clear()
                    response = scope_data.path0.data + 1j * scope_data.path1.data
                elif acquire.mode == AcquireMode.INTEGRATOR:
                    response = integ_data.path0 + 1j * integ_data.path1
                else:
                    raise ValueError(f"Unrecognised acquire mode {acquire.mode}")

                post_procs, axes = pp_map[name], {}
                for pp in post_procs:
                    if (
                        pp.process == PostProcessType.MEAN
                        and ProcessAxis.SEQUENCE in pp.axes
                    ):
                        repeat_counts.clear()
                    elif pp.process == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
                        response, _ = software_post_process_linear_map_complex_to_real(
                            pp.args, response, axes
                        )
                    elif pp.process == PostProcessType.DISCRIMINATE:
                        # f : {0, 1} ----> {-1, 1}
                        #       x    |---> 1 - 2x
                        response = 1 - 2 * thrld_data

                loop_nest_shape = tuple((sweep_counts or [1]) + (repeat_counts or [-1]))
                response = response.reshape(loop_nest_shape)
                if sweep_counts:
                    response = response.squeeze()
                results[name] = response

        results = self._process_results(results, triage_result)
        results = self._process_assigns(results, triage_result)

        return results

    def execute(self, programs: list[QbloxProgram], triage_result: TriageResult) -> Dict:
        playbacks: Dict[str, List[Acquisition]] = defaultdict(list)
        for program in programs:
            if self.plot_program:
                plot_program(program)

            if self.dump_program:
                for pulse_channel_id, pkg in program.packages.items():
                    filename = f"schedules/target_{pulse_channel_id}_@_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.json"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    with open(filename, "w") as f:
                        f.write(json.dumps(asdict(pkg.sequence)))

            self.instrument.setup(program)
            self.instrument.playback()
            payback: Dict[str, List[Acquisition]] = self.instrument.collect()
            for pulse_channel_id, acquisitions in payback.items():
                playbacks[pulse_channel_id] += acquisitions
        playback = self.combine_playbacks(playbacks)

        if self.plot_playback:
            plot_playback(playbacks)

        results = self.process_playback(playback, triage_result)
        return results
