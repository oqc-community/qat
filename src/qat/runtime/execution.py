from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, Iterable, List, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing
from pydantic import Field

from qat.model.model import QuantumHardwareModel
from qat.purr.backends.utilities import get_axis_map
from qat.purr.compiler.emitter import InstructionEmitter, QatFile
from qat.purr.compiler.execution import DeviceInjectors, SweepIterator
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    IndexAccessor,
    Instruction,
    ResultsProcessing,
    Variable,
)
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration
from qat.runtime.live_devices import ControlHardware, InstrumentConnectionManager
from qat.utils.pydantic import WarnOnExtraFieldsModel

log = get_default_logger()


class InstructionExecutionEngine(WarnOnExtraFieldsModel, ABC):
    """
    Base class for an engine that can execute instructions.

    Attributes:
        model: The hardware model.
        max_instruction_len: The max length an instruction can have.
    """

    model: QuantumHardwareModel
    max_instruction_len: int = Field(ge=1, default=2000000)

    @abstractmethod
    def execute(self, instructions: List[Instruction]) -> Dict[str, Any]: ...


class QuantumExecutionEngine(InstructionExecutionEngine):
    """
    Engine that can execute instructions on a quantum hardware model.
    """

    def execute(self, instructions: QatFile):
        """Executes this qat file against this current hardware."""
        return self._common_execute(instructions)

    ### !!! This method is probably going to be refactored according to the new pass manager feature.
    def _common_execute(
        self, instructions: List[Instruction], interrupt: Interrupt = NullInterrupt()
    ):
        """Executes this qat file against this current hardware."""
        with log_duration("QPU returned results in {} seconds."):
            # This is where we'd send the instructions off to the compiler for
            # processing, for now do ad-hoc processing.
            qat_file = InstructionEmitter().emit(instructions, self.model)

            # Rebuild repeat list if the hardware can't support the current setup.
            repeat_count = qat_file.repeat.repeat_count
            if repeat_count > self.model.repeat_limit:
                log.info(
                    f"Running {repeat_count} shots at once is "
                    f"unsupported on the current hardware. Batching execution."
                )
            batches = _generate_repeat_batches(repeat_count, self.model.repeat_limit)

            # Set up the sweep and device injectors, passing all appropriate data to
            # the specific execute methods and deal with clean-up afterwards.
            results = {}
            for i, batch_count in enumerate(batches):
                metadata = {"batch_iteration": i}
                interrupt.if_triggered(metadata, throw=True)
                if batch_count <= 0:
                    continue

                # Reset iterators/injectors per cycle.
                switerator = SweepIterator.from_qfile(qat_file)
                dinjectors = DeviceInjectors(qat_file.instructions)

                # Assign shortened repeat for this execution then reset it.
                qat_file.repeat.repeat_count = batch_count
                try:
                    dinjectors.inject()
                    batch_results = self._execute_on_hardware(
                        switerator, qat_file, interrupt
                    )
                finally:
                    switerator.revert(qat_file.instructions)
                    dinjectors.revert()
                    qat_file.repeat.repeat_count = repeat_count

                results = self._accumulate_results(results, batch_results)

            # Process metadata assign/return values to make sure the data is in the
            # right form.
            results = self._process_results(results, qat_file)
            results = self._process_assigns(results, qat_file)

            return results

    @abstractmethod
    def _execute_on_hardware(
        self, sweep_iterator, package: QatFile, interrupt: Interrupt = NullInterrupt()
    ) -> Union[Dict[str, List[float]], Dict[str, np.ndarray]]: ...

    def _execute_with_interrupt(
        self, instructions: QatFile, interrupt: Interrupt = NullInterrupt()
    ):
        """Executes this qat file against this current hardware.
        Execution allows for interrupts triggered by events.
        """
        return self._common_execute(instructions, interrupt)

    def _process_assigns(self, results, qfile: QatFile):
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

        assigns = dict(results)
        for assign in qfile.meta_instructions:
            if not isinstance(assign, Assign):
                continue

            assigns[assign.name] = recurse_arrays(assigns, assign.value)

        return {key: assigns[key] for key in qfile.return_.variables}

    def _process_results(self, results, qfile: QatFile):
        """
        Process any software-driven results transformation, such as taking a raw
        waveform result and turning it into a bit, or something else.
        """
        results_processing = {
            val.variable: val
            for val in qfile.meta_instructions
            if isinstance(val, ResultsProcessing)
        }

        missing_results = {
            val.output_variable
            for val in qfile.instructions
            if isinstance(val, Acquire) and val.output_variable not in results_processing
        }

        # For any acquires that are raw, assume they're experiment results.
        for missing_var in missing_results:
            results_processing[missing_var] = ResultsProcessing(
                missing_var, InlineResultsProcessing.Experiment
            )

        for inst in results_processing.values():
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

    @staticmethod
    def _accumulate_results(results: Dict, batch_results: Dict):
        if not any(batch_results):
            return results

        if any(results):
            if results.keys() != batch_results.keys():
                raise ValueError(
                    f"Dictionaries' keys mismatch, {results.keys()} != {batch_results.keys()}."
                )

            for key in results:
                existing = results[key]
                appending = batch_results[key]

                if type(existing) is not type(appending):
                    raise ValueError(
                        f"Expected objects with the same type, got {type(existing)} and {type(appending)} instead."
                    )

                if isinstance(existing, np.ndarray):
                    # TODO: Certain postprocessing instructions result in
                    # strange concatenation, e.g. sequential mean.
                    results[key] = np.concatenate(
                        (existing, appending), axis=existing.ndim - 1
                    )
                elif isinstance(existing, List):

                    def combine_lists(exi, new):
                        if len(new) > 0 and not isinstance(new[0], list):
                            exi += new
                            return
                        for i, item in enumerate(new):
                            combine_lists(exi[i], item)

                    combine_lists(existing, appending)
                else:
                    raise ValueError(
                        f"Cannot combine objects of unsupported type {type(existing)}"
                    )
        else:
            results = batch_results

        return results

    def __repr__(self):
        return f"{type(self).__name__}(model={str(self.model)})"


class EchoEngine(QuantumExecutionEngine):
    """
    A backend that just returns default values.
    Primarily used for testing and no-backend situations.
    """

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


class LiveDeviceEngine(QuantumExecutionEngine):
    """
    Backend that hooks up the logical hardware model to our QPU's, currently hardcoded to particular fridges.
    This will only work when run on a machine physically connected to a QPU.
    """

    instruments: InstrumentConnectionManager
    startup_engine: bool = True
    control_hardware: ControlHardware | None = None

    def __init__(self, **data):
        super.__init__(**data)
        if self.startup_engine:
            self.startup()

    def startup(self):
        connected = self.instruments.connect()
        return connected

    def shutdown(self):
        connected = self.instruments.disconnect()
        return connected

    def execute(self):
        pass


def _complex_to_binary(number: complex):
    """Base calculation for changing a complex measurement to binary form."""
    return 0 if number.real > 0 else 1


def _binary(results_list):
    """Changes all measurements to binary format."""
    if not isinstance(results_list, Iterable):
        return [results_list]

    results = []
    for item in results_list:
        if isinstance(item, Iterable):
            results.append(_binary(item))
        elif isinstance(item, complex):  # If we're a flat register, just append.
            results.append(_complex_to_binary(item))
        elif isinstance(item, float):
            results.append(0 if item > 0 else 1)
        else:
            results.append(item)
    return results


def _binary_average(results_list):
    """
    Averages all repeat results and returns a definitive 1/0 for each qubit measurement.
    """
    # If we have many sweeps/repeats loop through all of them and sum.
    if all([isinstance(val, list) for val in results_list]):
        binary_results = [_binary_average(nested) for nested in results_list]
    else:
        binary_results = _binary(results_list)

    return 1 if sum(binary_results) >= (len(binary_results) / 2) else 0


def _numpy_array_to_list(array):
    """Transform numpy arrays to a normal list."""
    if isinstance(array, np.ndarray):
        numpy_list: List = array.tolist()
        if len(numpy_list) == 1:
            return numpy_list[0]
        return numpy_list
    elif isinstance(array, List):
        list_list = [_numpy_array_to_list(val) for val in array]
        if len(list_list) == 1:
            return list_list[0]
        return list_list
    else:
        return array


def _generate_repeat_batches(repeats, batch_limit):
    """
    Batches together executions if we have more than the amount the hardware can
    handle at once. A repeat limit of -1 disables this functionality and just runs
    everything at once.
    """
    if batch_limit == -1:
        return [repeats]

    list_expansion_ratio = int(
        (Decimal(repeats) / Decimal(batch_limit)).to_integral(ROUND_DOWN)
    )
    remainder = repeats % batch_limit
    batches = [batch_limit] * list_expansion_ratio
    if remainder > 0:
        batches.append(remainder)

    return batches
