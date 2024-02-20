# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import abc
from decimal import ROUND_DOWN, Decimal
from math import ceil
from numbers import Number
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from qat.purr.backends.utilities import (
    UPCONVERT_SIGN,
    PositionData,
    SimpleAcquire,
    evaluate_shape,
    software_post_process_discriminate,
    software_post_process_down_convert,
    software_post_process_linear_map_complex_to_real,
    software_post_process_mean,
)
from qat.purr.compiler.config import InlineResultsProcessing, ResultsFormatting
from qat.purr.compiler.devices import MaxPulseLength, MeasureChannel, PulseChannel
from qat.purr.compiler.emitter import InstructionEmitter, QatFile
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    Delay,
    DeviceUpdate,
    FrequencyShift,
    IndexAccessor,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    Pulse,
    Repeat,
    Reset,
    ResultsProcessing,
    Sweep,
    Synchronize,
    Variable,
    Waveform,
    is_generated_name,
)
from qat.purr.error_mitigation.readout_mitigation import (
    PostProcessingReadoutMitigation,
    get_readout_mitigation,
)
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


class InstructionExecutionEngine(abc.ABC):
    def __init__(self, model: QuantumHardwareModel = None):
        self.model: Optional[QuantumHardwareModel] = None
        if model is not None:
            self.load_model(model)

    def load_model(self, model: QuantumHardwareModel):
        """ Shuts down the current hardware, loads the new model then restarts. """
        if self.model is not None:
            self.shutdown()
        self.model = model
        self.startup()

    def startup(self):
        """ Starts up the underlying hardware or does nothing if already started. """

    @abc.abstractmethod
    def execute(self, instructions: List[Instruction],
                results_format=None) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        pass

    @abc.abstractmethod
    def validate(self, instructions: List[Instruction]):
        pass

    def shutdown(self):
        """
        Shuts down the underlying hardware when this instance is no longer in use.
        """


class QuantumExecutionEngine(InstructionExecutionEngine):
    def __init__(self, model: QuantumHardwareModel = None, instruction_limit: int = 200000):
        super().__init__(model)
        self.instruction_limit = instruction_limit

    def _model_exists(self):
        if self.model is None:
            raise ValueError("Requires a loaded hardware model.")

    def _process_assigns(self, results, qfile: "QatFile"):
        """
        As assigns are classical instructions they are not processed as a part of the
        quantum execution (right now). Read through the results dictionary and perform
        the assigns directly, return the results.
        """
        def recurse_arrays(results_map, value):
            """ Recurse through assignment lists and fetch values in sequence. """
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

    def _process_results(self, results, qfile: "QatFile"):
        """
        Process any software-driven results transformation, such as taking a raw
        waveform result and turning it into a bit, or something else.
        """
        results_processing = {
            val.variable: val for val in qfile.meta_instructions
            if isinstance(val, ResultsProcessing)
        }

        missing_results = {
            val.output_variable for val in qfile.instructions
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

            if InlineResultsProcessing.Raw in inst.results_processing \
                    and InlineResultsProcessing.Binary in inst.results_processing:
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

    def _apply_results_formatting(self, results, format_flags: ResultsFormatting, repeats):
        """
        Transform the raw results into the format that we've been asked to provide. Look at individual transformation
        documentation for descriptions on what they do.
        """
        if len(results) == 0:
            return []

        # If we have no flags at all just infer structure simplification.
        if format_flags is None:
            format_flags = ResultsFormatting.DynamicStructureReturn

        def simplify_results(simplify_target):
            """
            To facilitate backwards compatability and being able to run low-level experiments alongside quantum
            programs we make some assumptions based upon form of the results.

            If all results have default variable names then the user didn't care about value assignment or this was a
            low-level experiment - in both cases, it means we can throw away the names and simply return the results in
            the order they were defined in the instructions.

            If we only have one result after this, just return that list directly instead, as it's probably just a single
            experiment.
            """
            if all([is_generated_name(k) for k in simplify_target.keys()]):
                if len(simplify_target) == 1:
                    return list(simplify_target.values())[0]
                else:
                    squashed_results = list(simplify_target.values())
                    if all(isinstance(val, np.ndarray) for val in squashed_results):
                        return np.array(squashed_results)
                    return squashed_results
            else:
                return simplify_target

        if ResultsFormatting.BinaryCount in format_flags:
            results = {key: _binary_count(val, repeats) for key, val in results.items()}

        def squash_binary(value):
            if isinstance(value, int):
                return str(value)
            elif all(isinstance(val, int) for val in value):
                return ''.join([str(val) for val in value])

        if ResultsFormatting.SquashBinaryResultArrays in format_flags:
            results = {key: squash_binary(val) for key, val in results.items()}

        # Dynamic structure return is an ease-of-use flag to strip things that you know your
        # use-case won't use, such as variable names and nested lists.
        if ResultsFormatting.DynamicStructureReturn in format_flags:
            results = simplify_results(results)

        return results

    def _apply_error_mitigation(self, results, qfile: "QatFile"):
        """
        Process any software-driven results transformation, such as taking a raw waveform result
        and turning it into a bit, or something else.
        """
        readout_instructions = [
            inst for inst in qfile.meta_instructions
            if isinstance(inst, PostProcessingReadoutMitigation)
        ]
        if not any(readout_instructions):
            return results

        if len(results) > 1:
            # TODO: add support for multiple registers
            raise ValueError(
                "Cannot have multiple registers in conjunction with readout error mitigation."
            )

        for readout_type in readout_instructions:
            mitigator = get_readout_mitigation(readout_type)
            new_result = mitigator.apply_error_mitigation(results, qfile, self.model)
            results[mitigator.name] = new_result
        return results

    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        """ Runs optimization passes specific to this hardware. """
        self._model_exists()
        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in instructions:
            if isinstance(instruction, PhaseShift):
                if accum_phaseshift := accum_phaseshifts.get(instruction.channel, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.channel] \
                        = PhaseShift(instruction.channel, instruction.phase)
            elif isinstance(instruction, Pulse):
                for quantum_target in instruction.quantum_targets:
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
        """ Validates this graph for execution on the current hardware."""
        self._model_exists()
        instruction_length = len(instructions)
        if instruction_length > self.instruction_limit:
            raise ValueError(
                f"Program too large to be run in a single block on current hardware. "
                f"{instruction_length} instructions."
            )

        for inst in instructions:
            if isinstance(inst, Acquire) and not inst.channel.acquire_allowed:
                raise ValueError(
                    f"Cannot perform an acquire on the physical channel with id "
                    f"{inst.channel.physical_channel}"
                )
            if isinstance(inst, Pulse):
                duration = inst.duration
                if isinstance(duration, Number) and duration > MaxPulseLength:
                    raise ValueError(
                        f"Max Waveform width is {MaxPulseLength} s "
                        f"given: {inst.duration} s"
                    )
                elif isinstance(duration, Variable):
                    values = next(
                        iter([
                            sw.variables[duration.name] for sw in instructions if
                            isinstance(sw, Sweep) and duration.name in sw.variables.keys()
                        ])
                    )
                    if np.max(values) > MaxPulseLength:
                        raise ValueError(
                            f"Max Waveform width is {MaxPulseLength} s "
                            f"given: {values} s"
                        )

            if isinstance(inst, Repeat) \
                    and inst.repeat_count is not None \
                    and inst.repeat_count < 1:
                raise ValueError(
                    f"{inst.repeat_count} shots is invalid. Please use 1 and above."
                )

    def _generate_repeat_batches(self, repeats):
        """
        Batches together executions if we have more than the amount the hardware can
        handle at once. A repeat limit of -1 disables this functionality and just runs
        everything at once.
        """
        batch_limit = self.model.shot_limit
        if batch_limit == -1:
            return [repeats]

        list_expansion_ratio = int(
            (Decimal(repeats) / Decimal(batch_limit)).to_integral(ROUND_DOWN)
        )
        return ([batch_limit] * list_expansion_ratio) + [repeats % batch_limit]

    def execute(self, instructions, results_format=None):
        """ Executes this qat file against this current hardware. """
        self._model_exists()

        with log_duration("QPU returned results in {} seconds."):
            # This is where we'd send the instructions off to the compiler for
            # processing, for now do ad-hoc processing.
            qat_file = InstructionEmitter().emit(instructions, self.model)

            # Rebuild repeat list if the hardware can't support the current setup.
            if qat_file.repeat.repeat_count > self.model.shot_limit:
                log.info(
                    f"Running {qat_file.repeat.repeat_count} shots at once is "
                    f"unsupported on the current hardware. Batching execution."
                )
                batches = self._generate_repeat_batches(sum(qat_file.repeat.repeat_count))
            else:
                batches = [qat_file.repeat.repeat_count]

            # Set up the sweep and device injectors, passing all appropriate data to
            # the specific execute methods and deal with clean-up afterwards.
            switerator = SweepIterator.from_qfile(qat_file)
            dinjectors = DeviceInjectors(qat_file.instructions)
            results = {}
            for batch_count in batches:
                qat_file.repeat.repeat_count = batch_count
                dinjectors.inject()
                batch_results = self._execute_on_hardware(switerator, qat_file)
                switerator.revert(qat_file.instructions)
                dinjectors.revert()

                if not any(results):
                    results = batch_results
                else:
                    # As it's the same execution we can assume it has the same keys.
                    for key in results:
                        results[key].append(batch_results[key])

            # Process metadata assign/return values to make sure the data is in the
            # right form.
            results = self._process_results(results, qat_file)
            results = self._process_assigns(results, qat_file)

            repeats = qat_file.repeat.repeat_count if qat_file.repeat is not None else self.model.default_repeat_count
            results = self._apply_results_formatting(results, results_format, repeats)
            results = self._apply_error_mitigation(results, qat_file)
            return results

    @abc.abstractmethod
    def _execute_on_hardware(self, sweep_iterator, package: "QatFile"):
        pass

    def calculate_duration(self, instruction, return_samples: bool = True):
        """
        Calculates the duration of the instruction for this particular piece of
        hardware.
        """
        pulse_targets = [
            pulse for pulse in instruction.quantum_targets
            if isinstance(pulse, PulseChannel)
        ]
        if not any(pulse_targets):
            return 0

        # TODO: Allow for multiple pulse channel targets.
        if len(pulse_targets) > 1 and not isinstance(instruction, PhaseReset):
            get_default_logger().warning(
                f"Attempted to calculate duration of {str(instruction)} that has "
                "multiple target channels. We're arbitrarily using the duration of the "
                "first channel to calculate instruction duration."
            )

        pc = pulse_targets[0].physical_channel
        block_size = pc.block_size
        block_time = pc.block_time

        # round to remove floating point errors
        block_number = ceil(round(instruction.duration / block_time, 4))
        if return_samples:
            calc_sample = block_number * block_size
        else:
            calc_sample = block_number * block_time

        return calc_sample

    def create_duration_timeline(self, package: QatFile):
        """
        Builds a dictionary mapping channels to a list of instructions and at what
        precise sample time they should be associated with. It's important that the
        times are absolute here, not relative.

        :Example:

        .. code-block:: python

            {"drive": [
                (0, 1000, pulse),
                (1000, 2000, pulse),
                (1000, 2000, acquire)
                ...
            ]}
        """
        results: Dict[PulseChannel, List[PositionData]] = {}
        total_durations: Dict[PulseChannel, int] = dict()

        for instruction in package.instructions:
            for qtarget in instruction.quantum_targets:

                # TODO: Acquire is a special quantum target for post processing.
                #  This should probably be changed.
                if isinstance(qtarget, Acquire):
                    qtarget = qtarget.channel

                device_instructions: List[PositionData] = results.setdefault(qtarget, [])
                if not any(device_instructions):
                    sample_start = 0
                else:
                    sample_start = device_instructions[-1].end

                # For syncs we want to look at the currently-processed instructions on
                # the channels we target, get the max end time then align all of our
                # channels to that point in time.
                position_data = None
                if isinstance(instruction, Synchronize):
                    current_durations = {
                        qt: total_durations.setdefault(qt, 0)
                        for qt in instruction.quantum_targets
                    }
                    longest_length = max(current_durations.values(), default=0.0)
                    delay_time = longest_length - total_durations[qtarget]
                    if delay_time > 0:
                        instr = Delay(qtarget, delay_time)
                        position_data = PositionData(
                            sample_start,
                            sample_start + self.calculate_duration(instr),
                            instr
                        )
                else:
                    position_data = PositionData(
                        sample_start,
                        sample_start + self.calculate_duration(instruction),
                        instruction
                    )

                if position_data is not None:
                    device_instructions.append(position_data)

                    # Calculate running durations for sync/delay evaluation
                    current_duration = total_durations.setdefault(qtarget, 0)
                    total_durations[qtarget] \
                        = current_duration + position_data.instruction.duration

        # Strip timelines that only hold delays, since that just means nothing is
        # happening on this channel.
        for key, timeline in dict(results).items():
            if all(isinstance(pos_data.instruction, Delay) for pos_data in timeline):
                del results[key]

        return results

    def build_pulse_channel_buffers(
        self,
        position_map: Dict[PulseChannel, List[PositionData]],
        do_upconvert: bool = True
    ):
        buffers = {}
        for pulse_channel, positions in position_map.items():
            buffers[pulse_channel] = buffer = np.zeros(
                positions[-1].end if any(positions) else 0, dtype='cfloat'
            )

            phase = 0.0
            frequency = pulse_channel.frequency
            for pos in positions:
                if isinstance(pos.instruction, Waveform):
                    phase = self.process_pulse(
                        pos, buffer, pulse_channel, phase, frequency, do_upconvert
                    )
                elif isinstance(pos.instruction, PhaseShift):
                    phase = self.process_phaseshift(pos, phase)
                elif isinstance(pos.instruction, PhaseReset):
                    phase = self.process_phasereset(pos, phase)
                elif isinstance(pos.instruction, Reset):
                    self.process_reset(pos)
                elif isinstance(pos.instruction, FrequencyShift):
                    frequency = self.process_frequencyshift(pos, frequency, pulse_channel)

        return buffers

    def process_pulse(
        self,
        position: PositionData,
        buffer: List[float],
        pulse_channel: PulseChannel,
        phase: float,
        frequency: float,
        do_upconvert: bool
    ):
        dt = pulse_channel.sample_time

        samples = position.end - position.start
        length = (position.end - position.start) * dt
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * dt, stop=length - centre - 0.5 * dt, num=samples
        )
        half_sample = isinstance(pulse_channel, MeasureChannel)
        pulse = evaluate_shape(position.instruction, t, phase, half_sample)
        if not position.instruction.ignore_channel_scale:
            pulse *= pulse_channel.scale
        pulse += pulse_channel.bias

        if do_upconvert:
            t += centre - 0.5 * dt + position.start * dt
            pulse = self.do_upconvert(pulse, t, pulse_channel, frequency)

        buffer[position.start:position.end] = pulse

        return phase

    def process_phaseshift(self, position: PositionData, phase: float):
        return phase + position.instruction.phase

    def process_phasereset(self, position: PositionData, phase: float):
        """ Child hardware types might need to know the phase and instruction """
        return 0

    def process_reset(self, position: PositionData):
        pass

    def process_frequencyshift(
        self, position: PositionData, frequency: float, pulse_channel: PulseChannel
    ):
        # Check no pulse channels on this physical channel used a fixed if
        for channel in pulse_channel.physical_channel.related_pulse_channels:
            if channel.fixed_if:
                raise NotImplementedError(
                    f"Hardware does not currently support frequency shifts on the physical channel "
                    f"{pulse_channel.physical_channel}."
                )

        new_frequency = frequency + position.instruction.frequency
        if new_frequency < pulse_channel.min_frequency or new_frequency > pulse_channel.max_frequency:
            raise ValueError(
                f"Cannot shift pulse channel frequency from '{frequency}' to "
                f"'{new_frequency}'. The new frequency must be between the bounds "
                f"({pulse_channel.min_frequency}, {pulse_channel.max_frequency}) on "
                f"physical channel with id {pulse_channel.physical_channel_id}."
            )

        return new_frequency

    def do_upconvert(
        self,
        buffer: List[float],
        time: List[float],
        pulse_channel: PulseChannel,
        frequency: float
    ):
        tslip = pulse_channel.phase_offset
        imbalance = pulse_channel.imbalance
        if pulse_channel.fixed_if:
            freq = pulse_channel.baseband_if_frequency
        else:
            freq = frequency - pulse_channel.baseband_frequency
        buffer *= np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * time)
        if not tslip == 0.0:
            buffer_slip = buffer * np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * tslip)
            buffer.imag = buffer_slip.imag
        if not imbalance == 1.0:
            buffer.real /= imbalance**0.5
            buffer.imag *= imbalance**0.5

        return buffer

    def build_physical_channel_buffers(
        self, pulse_channel_buffers: Dict[PulseChannel, np.ndarray]
    ):
        # Add all pulse channel buffers belonging to the same physical channel together
        buffers = {}
        for physical_channel_id in self.model.physical_channels.keys():
            physical_channel_pulse_buffers = [
                buffer for pulse_channel,
                buffer in pulse_channel_buffers.items()
                if pulse_channel.physical_channel_id == physical_channel_id
            ]
            physical_channel_buffer = buffers.setdefault(
                physical_channel_id,
                np.zeros(
                    max(map(len, physical_channel_pulse_buffers), default=0),
                    dtype='cfloat'
                )
            )
            for buffer in physical_channel_pulse_buffers:
                physical_channel_buffer[0:len(buffer)] += buffer

        return buffers

    def build_acquire_list(self, position_map: Dict[PulseChannel, List[PositionData]]):
        buffers = {}
        for pulse_channel, positions in position_map.items():
            buffer = buffers.setdefault(pulse_channel.full_id(), [])
            for pos in positions:
                if isinstance(pos.instruction, Acquire):
                    if pos.instruction.filter is not None:
                        raise NotImplementedError(
                            "Hardware currently doesn't support custom filters."
                        )

                    samples = pos.end - pos.start
                    buffer.append(
                        SimpleAcquire(
                            pos.start,
                            samples,
                            pos.instruction.output_variable,
                            pulse_channel,
                            pulse_channel.physical_channel,
                            pos.instruction.mode,
                            pos.instruction.delay,
                            pos.instruction
                        )
                    )

        return buffers

    def run_post_processing(self, post_processing: PostProcessing, value, value_axis):
        """ Run post-processing on the response via software. """
        if post_processing.process == PostProcessType.DOWN_CONVERT:
            return software_post_process_down_convert(
                post_processing.args, post_processing.axes, value, value_axis
            )
        elif post_processing.process == PostProcessType.MEAN:
            return software_post_process_mean(post_processing.axes, value, value_axis)
        elif post_processing.process == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL:
            return software_post_process_linear_map_complex_to_real(
                post_processing.args, value, value_axis
            )
        elif post_processing.process == PostProcessType.DISCRIMINATE:
            return software_post_process_discriminate(
                post_processing.args, value, value_axis
            )

    def __repr__(self):
        if self.model is not None:
            return f"{self.__class__.__name__} with {len(self.model.qubits)} qubits"
        return "No hardware loaded."


def _binary_count(results_list, repeats):
    """
    Returns a dictionary of binary number: count. So for a two qubit register it'll return the various counts for
    00, 01, 10 and 11.
    """
    def flatten(res):
        """ Combine binary result from the QPU into composite key result. Aka '0110' or '0001' """
        if isinstance(res, Iterable):
            return ''.join([flatten(val) for val in res])
        else:
            return str(res)

    def get_tuple(res, index):
        return [val[index] if isinstance(val, (List, np.ndarray)) else val for val in res]

    binary_results = _binary(results_list)

    # If our results are a single qubit then pretend to be a register of one.
    if isinstance(next(iter(binary_results), None), Number) \
            and len(binary_results) == repeats:
        binary_results = [binary_results]

    result_count = dict()
    for qubit_result in [list(get_tuple(binary_results, i)) for i in range(repeats)]:
        key = flatten(qubit_result)
        value = result_count.get(key, 0)
        result_count[key] = value + 1

    return result_count


def _complex_to_binary(number: complex):
    """ Base calculation for changing a complex measurement to binary form. """
    return 0 if number.real > 0 else 1


def _binary(results_list):
    """ Changes all measurements to binary format. """
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
    """ Transform numpy arrays to a normal list. """
    if isinstance(array, np.ndarray):
        numpy_list: List = array.tolist()
        if len(numpy_list) == 1:
            return numpy_list[0]
        return numpy_list
    elif isinstance(array, List):
        return [_numpy_array_to_list(val) for val in array]
    else:
        return array


class DeviceInjector:
    """
    Injects a value into a device for the entirety of a sweeps duration. Analogous to
    setting a device value before execution.
    """
    def __init__(self, target: DeviceUpdate):
        super().__init__()
        self.updater = target
        self.revert_value = None

    def inject(self):
        if isinstance(self.updater.target, Dict):
            self.revert_value = self.updater.target[self.updater.attribute]
            self.updater.target[self.updater.attribute] = self.updater.value
        else:
            self.revert_value = getattr(self.updater.target, self.updater.attribute)
            setattr(self.updater.target, self.updater.attribute, self.updater.value)

    def revert(self):
        if isinstance(self.updater.target, Dict):
            self.updater.target[self.updater.attribute] = self.revert_value
        else:
            setattr(self.updater.target, self.updater.attribute, self.revert_value)

    def __repr__(self):
        return f"{self.updater.target.id}.{self.updater.attribute} = " \
            f"{str(self.updater.value)}"


class ValueReplacement(abc.ABC):
    """ Abstract class for values that inject during a sweep. """
    @abc.abstractmethod
    def replace(self, field_value, symbols, index):
        pass

    @abc.abstractmethod
    def revert(self):
        pass


class VariableInjector(ValueReplacement):
    """
    All injected values are assumed to be in a list of len(sweep_length), this just
    fetches out the particular value of a list to inject as we're sweeping.

    So calling this with ``name='frequency', index=5``, from:

    .. code-block:: python

        {
            'frequency': [1, 2, 4, 8, 16, 32, 64, ...],
            ...
        }

    Would inject 32 into this field, as we're on the 6th iteration of this sweep.
    """
    def __init__(self, var):
        self.var: Variable = var

    def replace(self, field_value, replacements, index):
        if self.var.name in replacements:
            return replacements[self.var.name][index]
        else:
            return field_value

    def revert(self):
        return self.var


class IteratorInjector(ValueReplacement):
    """
    Value injector for objects in lists, tuples and dictionaries.

    Currently when we inject an iteration into an iteration of the same type - resolving
    a ValueReplacer entry in a dictionary also returns a dictionary - it merges that
    value into the iteration instead of a simple insertion.

    So the list ``[1, 2, ValueReplacer(...)]`` that resolves into ``[1, 2, [3, 4, 5]]`` will become ``[1, 2, 3, 4, 5]``. But
    if it was ``[1, 2, (3, 4, 5)]`` it will stay as such.
    """
    def __init__(self, iteration):
        self.iteration = iteration

    def replace(self, field_value, replacements, index):
        if isinstance(self.iteration, (List, Tuple)):
            type_to_check = type(self.iteration)
            expanded = []
            for val in self.iteration:
                if isinstance(val, Variable):
                    replacement_value = replacements[val.name][index]
                    if isinstance(replacement_value, type_to_check):
                        expanded.extend(replacement_value)
                    else:
                        expanded.append(replacement_value)
                else:
                    expanded.append(val)
            return type_to_check(expanded)
        elif isinstance(self.iteration, Dict):
            return {
                key: replacements[value.name][index]
                if isinstance(value, Variable) and key in replacements else value for key,
                value in self.iteration.items()
            }
        else:
            return self.iteration

    def revert(self):
        return self.iteration


class InjectionMetadata:
    """
    A class which is injected into an instruction to hold data about its state before
    injection started, as well as objects that facilitate dynamic injection of values
    into fields.

    It'll replace fields in the object that hold ``Variable('X')`` with the value of ``X``. When revert is called it replaces
    the field with ``Variable('X')`` again, essentially reseting it for next execution.
    """

    field: str = "_$injection_metadata"

    def __init__(self):
        self.variables: Dict[str, ValueReplacement] = dict()

    def inject(self, node, replacements: Dict, index):
        for field, replacer in self.variables.items():
            setattr(
                node, field, replacer.replace(getattr(node, field), replacements, index)
            )

    def revert(self, node):
        for field, replacer in self.variables.items():
            setattr(node, field, replacer.revert())

    def is_empty(self):
        return len(self.variables) == 0


class DeviceInjectors:
    """
    Special sort of injector that is a sort of double-injection. It takes an object that
    needs to be modified during a sweep, replaces certain fields with Variable (and
    others) and lets normal injection do it's work, then at the end of the sweep resets
    the field with the original value.
    """
    def __init__(self, instructions):
        self.injectors = [
            DeviceInjector(val) for val in instructions if isinstance(val, DeviceUpdate)
        ]

    def inject(self):
        for dinject in self.injectors:
            dinject.inject()

    def revert(self):
        for dinject in self.injectors:
            dinject.revert()


class SweepIterator:
    """
    Acts as a controller for sweep-reliant values and iterates the amount of times
    designated by the sweep.

    Every time ``do_sweep`` is called it will inject the current sweep values into the variable, so if passed
    a sweep list with ``{'dave': [1, 2, 3, 4, 5]}`` it will iterate 5 times and inject the values ``1...5`` into
    objects that has ``Variable('dave')``.

    Nested sweeps act as inner loops, and iterates inner to outer. So with a nested sweep of ``{'sam': [5, 10, 15]}``
    and using our previous example we will sweep a total of 15 times, looking a little like this:

    .. code-block:: python

        dave = 1, sam = 5
        dave = 1, sam = 10
        dave = 1, sam = 15
        dave = 2, sam = 5
        ...
    """
    def __init__(self, sweep=None, nested_sweep=None):
        self.sweep: Sweep = sweep
        self.current_iteration = -1
        self.nested_sweep: "SweepIterator" = nested_sweep

        # An injector is an object that does more complicated value injection with a
        # revert at the end.
        self.injection_match_prefixs = {
            SweepIterator.__module__.split('.')[0], self.__module__.split('.')[0]
        }

    @property
    def accumulated_sweep_iteration(self):
        """
        The current number of total iterations this sweep as run as a whole, including
        nested sweeps.
        """
        if self.nested_sweep is not None:
            return self.current_iteration * self.nested_sweep.length \
                + self.nested_sweep.accumulated_sweep_iteration
        else:
            return self.current_iteration + 1

    @staticmethod
    def from_qfile(qfile: "QatFile"):
        """ Build a sweep iterator from a .qat file."""
        sweep_iterator = SweepIterator()
        for meta in qfile.meta_instructions:
            if isinstance(meta, Sweep):
                sweep_iterator.add_sweep(meta)

        return sweep_iterator

    def _revert_sweep_values(self, node):
        """
        Reverts all changes made by injectors during running to their pre-sweep states.
        The execution should be able to be re-run using the same Sweep object and
        results should be the same.
        """
        # TODO: Only a very small amount of code is needed by the revert, pull out into
        #   new method.
        self._inject_sweep_values(node, None, None, revert=True)

    def _inject_sweep_values(
        self, node, replacers, index, revert=False, recursion_guard=None
    ):
        """
        Injects values into objects, mostly making sure that Variable objects turn into
        their actual values.
        """
        # If we can't get its internal objects, skip.
        has_dict = getattr(node, "__dict__", None)
        if not has_dict:
            return

        if recursion_guard is None:
            recursion_guard = set()

        # If we've already seen this object and it's a recursive call, skip.
        if id(node) in recursion_guard:
            return

        # If we're not an object we can check the location of, skip.
        module = type(node).__dict__.get("__module__", None)
        if module is None or (
            self.injection_match_prefixs is not None
            and not any(module.startswith(val) for val in self.injection_match_prefixs)
        ):
            return

        # Add our object to the recursion guard so we don't recurse.
        recursion_guard.add(id(node))

        # Get injection metadata, and if don't have any generate it.
        injection_meta = getattr(node, InjectionMetadata.field, None)
        if injection_meta is None:
            injection_meta = InjectionMetadata()
            for field, value in filter(
                lambda kvp: not kvp[0].startswith('_'), node.__dict__.items()
            ):
                if isinstance(value, Variable):
                    injection_meta.variables[field] = VariableInjector(value)
                elif (isinstance(value, (Tuple, List)) and any([isinstance(val, Variable) for val in value]))\
                        or (isinstance(value, Dict) and any([isinstance(val, Variable) for val in value.values()])):
                    injection_meta.variables[field] = IteratorInjector(value)

            setattr(node, InjectionMetadata.field, injection_meta)

        # Perform injection on this node, then recurse into all objects taking into
        # account iterables.
        if revert:
            injection_meta.revert(node)
            delattr(node, InjectionMetadata.field)
        else:
            injection_meta.inject(node, replacers, index)

        for value in node.__dict__.values():
            if isinstance(value, InjectionMetadata):
                continue

            if isinstance(value, (Tuple, List, Dict)):
                for val in value:
                    self._inject_sweep_values(
                        val, replacers, index, revert, recursion_guard
                    )
            else:
                self._inject_sweep_values(value, replacers, index, revert, recursion_guard)

    def reset_iteration(self):
        """
        Resets iteration of this sweep and all children since we infer that if we've
        finished, then our children have too.
        """
        self.current_iteration = -1
        if self.nested_sweep is not None:
            self.nested_sweep.reset_iteration()

    def do_sweep(self, instructions):
        """
        Start/continue sweeping. Injects values into the instructions appropriate for
        this sweep.
        """
        if self.nested_sweep is not None:
            # Special-case where a parent sweep won't increment until its child is
            # complete, but we start at -1 so need to bump it forward a single value.
            if self.current_iteration == -1:
                self.current_iteration = 0

            if self.nested_sweep.is_finished():
                self.current_iteration += 1
                self.nested_sweep.reset_iteration()
        else:
            self.current_iteration += 1

        if self.sweep is None:
            return

        for inst in instructions:
            self._inject_sweep_values(inst, self.sweep.variables, self.current_iteration)

        if self.nested_sweep is not None:
            self.nested_sweep.do_sweep(instructions)

    def add_sweep(self, nested: Sweep):
        """
        Adds a sweep onto this iterator. If it already has a sweep, adds it as a nested
        one.
        """
        if self.sweep is None:
            self.sweep = nested
        elif self.nested_sweep is None:
            self.nested_sweep = SweepIterator(nested)
        else:
            self.nested_sweep.add_sweep(nested)

    def revert(self, instructions):
        """ Revert all instruction modifications to their pre-sweep state."""
        if self.sweep is None:
            return

        for inst in instructions:
            self._revert_sweep_values(inst)

        if self.nested_sweep is not None:
            self.nested_sweep.revert(instructions)

    def is_finished(self):
        """ Is this sweep finished. """
        # We're zero-index so take that into account.
        return self.current_iteration == (self.sweep.length-1 if self.sweep is not None else 0) \
            and (self.nested_sweep is None or self.nested_sweep.is_finished())

    def get_results_shape(self, shape: Tuple = None):
        """ Return a default array that mirrors the structure for this set of sweeps."""
        if self.nested_sweep is not None:
            return self.sweep.length, *self.nested_sweep.get_results_shape(shape)
        return (self.length, *shape) if shape is not None else (self.length, )

    @property
    def length(self):
        """Returns actual length of the entire sweep, so

        ``sweep_length * nested_length * nested_length ...``
        """
        if self.sweep is None:
            return 1

        return self.sweep.length * (
            self.nested_sweep.length if self.nested_sweep is not None else 1
        )

    def insert_result_at_sweep_position(self, results_array: np.array, value: np.array):
        """ Insert the response value into the appropriate nested array. """
        if self.nested_sweep is not None:
            self.nested_sweep.insert_result_at_sweep_position(
                results_array[self.current_iteration], value
            )
        else:
            results_array[self.current_iteration] = value


