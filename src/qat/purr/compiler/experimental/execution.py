from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Any, Dict, Iterable, List

import numpy as np
from compiler_config.config import InlineResultsProcessing
from pydantic import Field

from qat import qatconfig
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.experimental.devices import PulseChannel
from qat.purr.compiler.experimental.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    CustomPulse,
    IndexAccessor,
    Instruction,
    PhaseReset,
    PhaseShift,
    Pulse,
    ResultsProcessing,
    Sweep,
    Variable,
)
from qat.purr.utils.pydantic import WarnOnExtraFieldsModel


class InstructionExecutionEngine(WarnOnExtraFieldsModel, ABC):
    """
    Engine that can optimise, validate and execute instructions
    on a hardware model.

    Attributes:
        model: The hardware model.
        max_instruction_len: The max length an instruction can have.
    """

    model: QuantumHardwareModel
    max_instruction_len: int = Field(ge=1, default=2000000)

    @abstractmethod
    def execute(self, instructions: List[Instruction]) -> Dict[str, Any]: ...

    @abstractmethod
    def optimize(self, instructions: List[Instruction]) -> List[Instruction]: ...

    def validate(self, instructions: List[Instruction]):
        if (instruction_length := len(instructions)) > self.max_instruction_len:
            raise ValueError(
                f"Program with {instruction_length} instructions too large to be run in a single block on current hardware."
            )


class QuantumInstructionExecutionEngine(InstructionExecutionEngine):
    startup_engine: bool = True

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

    def _process_results(self, results: Dict, qfile: QatFile):
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
            if target_values := results.get(inst.variable, None) is None:
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

    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        """Runs optimization passes specific to this hardware."""
        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in instructions:
            if isinstance(instruction, PhaseShift) and isinstance(
                instruction.phase, Number
            ):
                if accum_phaseshift := accum_phaseshifts.get(instruction.channel, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.channel] = PhaseShift(
                        instruction.channel, instruction.phase
                    )
            elif isinstance(instruction, (Pulse, CustomPulse)):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, List):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if quantum_target in accum_phaseshifts:
                        optimized_instructions.append(accum_phaseshifts.pop(quantum_target))
                optimized_instructions.append(instruction)
            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    accum_phaseshifts.pop(channel, None)
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)
        return optimized_instructions

    def validate(self, instructions: List[Instruction]):
        """Validates this graph for execution on the current hardware."""
        super.validate(instructions)

        for inst in instructions:
            if isinstance(inst, Acquire) and not inst.channel.acquire_allowed:
                raise ValueError(
                    f"Cannot perform an acquire on the physical channel with id "
                    f"{inst.channel.physical_channel}."
                )
            if isinstance(inst, (Pulse, CustomPulse)):
                duration = inst.duration

                if not qatconfig.DISABLE_PULSE_DURATION_LIMITS:
                    min_pulse_duration = self.model.min_pulse_length
                    max_pulse_duration = self.model.max_pulse_length
                    if isinstance(duration, Number):
                        if duration < min_pulse_duration or duration > max_pulse_duration:
                            raise ValueError(
                                f"Waveform duration {inst.duration} s is not within the limits {[min_pulse_duration, max_pulse_duration]}."
                            )
                    elif isinstance(duration, Variable):
                        values = next(
                            iter(
                                [
                                    sw.variables[duration.name]
                                    for sw in instructions
                                    if isinstance(sw, Sweep)
                                    and duration.name in sw.variables.keys()
                                ]
                            )
                        )
                        if (
                            np.min(values) < min_pulse_duration
                            or np.max(values) > max_pulse_duration
                        ):
                            raise ValueError(
                                f"Waveform durations {values} s are not within the limits {[min_pulse_duration, max_pulse_duration]}."
                            )


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
