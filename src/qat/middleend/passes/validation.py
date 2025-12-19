# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from functools import singledispatchmethod

import numpy as np
from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import InstructionBuilder, QuantumInstructionBuilder
from qat.ir.instructions import FrequencySet, FrequencyShift, Instruction, Repeat, Return
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    PostProcessing,
    ProcessAxis,
)
from qat.ir.waveforms import Pulse
from qat.middleend.passes.analysis import ActivePulseChannelResults
from qat.model.device import PhysicalChannel
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class InstructionValidation(ValidationPass):
    """
    Validates instructions against the hardware, including whether:
    - they fit in memory,
    - pulse durations are within the allowed limits and
    - acquire instructions are only performed on acquire channels..
    """

    def __init__(
        self,
        target_data: TargetData,
        model: PhysicalHardwareModel,
        pulse_duration_limits: bool | None = None,
        *args,
        **kwargs,
    ):
        """
        :param target_data: Target-related information.
        :param pulse_duration_limits: Whether to check the pulse duration limits.
            If None, uses the default from the QatConfig.
        :param instruction_memory_size: The maximum number of instructions that can
            be run in a single shot. If None, uses the default from QatConfig.
        """
        self.instruction_memory_size = target_data.instruction_memory_size
        self.pulse_duration_max = max(
            target_data.QUBIT_DATA.pulse_duration_max,
            target_data.RESONATOR_DATA.pulse_duration_max,
        )
        self.pulse_duration_min = min(
            target_data.QUBIT_DATA.pulse_duration_min,
            target_data.RESONATOR_DATA.pulse_duration_min,
        )
        self.pulse_duration_limits = (
            pulse_duration_limits
            if pulse_duration_limits is not None
            else get_config().INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS
        )
        self._channel_data = self._get_channel_data(model)

    @staticmethod
    def _get_channel_data(model):
        data = {"acquire": [], "non-acquire": {}}
        for q_id, qubit in model.qubits.items():
            data["acquire"].append(qubit.acquire_pulse_channel.uuid)
            for pulse_channel in qubit.all_pulse_channels + [qubit.measure_pulse_channel]:
                data["non-acquire"][pulse_channel.uuid] = [pulse_channel.pulse_type, q_id]
        return data

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        instruction_length = ir.number_of_instructions
        if instruction_length > self.instruction_memory_size:
            raise ValueError(
                f"Program with {instruction_length} instructions too large to be run "
                f"in a single block on current hardware."
            )

        for inst in ir.instructions:
            self._validate_instruction(inst)

        return ir

    @singledispatchmethod
    def _validate_instruction(self, instruction: Instruction):
        pass

    @_validate_instruction.register(Pulse)
    def _(self, instruction: Pulse):
        duration = instruction.duration
        if duration > self.pulse_duration_max or duration < self.pulse_duration_min:
            if self.pulse_duration_limits:
                # Do not throw error if we specifically disabled the limit checks.
                raise ValueError(
                    f"Waveform width must be between {self.pulse_duration_min} s "
                    f"and {self.pulse_duration_max} s. "
                    f"Given has a width: {instruction.duration} s"
                )

    @_validate_instruction.register(Acquire)
    def _(self, instruction: Acquire):
        if instruction.target not in self._channel_data["acquire"]:
            channel = self._channel_data["non-acquire"][instruction.target]
            raise ValueError(
                f"Cannot perform an acquire on the {channel[0]} pulse channel for qubit {channel[1]}"
            )


class DynamicFrequencyValidation(ValidationPass):
    """Validates the setting or shifting frequencies does not move the intermediate
    frequency of a pulse channel outside the allowed limits."""

    def __init__(self, model: PhysicalHardwareModel, target_data: TargetData):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model.
        :param target_data: Target-related information.
        """
        self.model = model
        self._is_resonator = self._create_resonator_map(model)

        # Extract limits from the target data.
        self.qubit_if_freq_limits = (
            target_data.QUBIT_DATA.pulse_channel_if_freq_min,
            target_data.QUBIT_DATA.pulse_channel_if_freq_max,
        )
        self.resonator_if_freq_limits = (
            target_data.RESONATOR_DATA.pulse_channel_if_freq_min,
            target_data.RESONATOR_DATA.pulse_channel_if_freq_max,
        )

    @staticmethod
    def _create_resonator_map(model: PhysicalHardwareModel) -> dict[str, bool]:
        """Creates a map to lookup if physical channels belong to resonators.."""
        res_map = {}
        for qubit in model.qubits.values():
            res_map[qubit.physical_channel.uuid] = False
            res_map[qubit.resonator.physical_channel.uuid] = True
        return res_map

    def _validate_frequency_shifts(self, physical_channel_id: str, ifs: list[float]):
        """Validates that the frequency shifts do not exceed the allowed limits."""

        if_limits = (
            self.qubit_if_freq_limits
            if not self._is_resonator[physical_channel_id]
            else self.resonator_if_freq_limits
        )

        return [
            if_value
            for if_value in ifs
            if not (if_limits[0] <= np.abs(if_value) <= if_limits[1])
        ]

    @singledispatchmethod
    def _calculate_frequency(
        self,
        instruction: Instruction,
        ir: QuantumInstructionBuilder,
        ifs: dict[str, list[float]],
        physical_channel_ids: dict[str, str],
    ):
        pass

    @_calculate_frequency.register(FrequencyShift)
    def _(
        self,
        instruction: FrequencyShift,
        ir: QuantumInstructionBuilder,
        ifs: dict[str, list[float]],
        physical_channel_ids: dict[str, str],
    ):
        frequencies = ifs[instruction.target]
        pulse_channel = ir.get_pulse_channel(instruction.target)
        physical_channel = self.model.physical_channel_with_id(
            pulse_channel.physical_channel_id
        )
        freq = (
            frequencies[-1]
            if len(frequencies) > 0
            else pulse_channel.frequency - physical_channel.baseband.frequency
        )
        frequencies.append(freq + instruction.frequency)
        physical_channel_ids[instruction.target] = physical_channel.uuid

    @_calculate_frequency.register(FrequencySet)
    def _(
        self,
        instruction: FrequencySet,
        ir: QuantumInstructionBuilder,
        ifs: dict[str, list[float]],
        physical_channel_ids: dict[str, str],
    ):
        physical_channel = self.model.physical_channel_with_id(
            ir.get_pulse_channel(instruction.target).physical_channel_id
        )
        ifs[instruction.target].append(
            instruction.frequency - physical_channel.baseband.frequency
        )
        physical_channel_ids[instruction.target] = physical_channel.uuid

    def run(
        self, ir: QuantumInstructionBuilder, *args, **kwargs
    ) -> QuantumInstructionBuilder:
        """:param ir: The list of instructions stored in an :class:`QuantumInstructionBuilder`."""

        ifs = defaultdict(list)
        physical_channel_ids = defaultdict(str)
        for instruction in ir.instructions:
            self._calculate_frequency(instruction, ir, ifs, physical_channel_ids)

        violations = []
        for pulse_channel_id, if_values in ifs.items():
            physical_channel_id = physical_channel_ids[pulse_channel_id]
            if if_violations := self._validate_frequency_shifts(
                physical_channel_id, if_values
            ):
                qubit = self.model.qubit_for_physical_channel_id(
                    physical_channel_ids[pulse_channel_id]
                )
                device_name = (
                    "Resonator " if self._is_resonator[physical_channel_id] else "Qubit "
                )
                device_name += str(self.model.index_of_qubit(qubit))

                if (
                    hw_pc := self.model.pulse_channel_with_id(pulse_channel_id)
                ) is not None:
                    pc_name = f"{hw_pc.pulse_type}"
                else:
                    pc_name = f"Custom({pulse_channel_id})"

                violations.append(
                    f"The IF of {device_name} {pc_name} pulse channel is set or "
                    f"shifted to values {if_violations} that exceed the allowed limits."
                )

        if len(violations) > 0:
            raise ValueError(
                "Dynamic frequency validation failed with the following violations:\n"
                + "\n".join(violations)
            )
        return ir


class NoMidCircuitMeasurementValidation(ValidationPass):
    """
    Validates that there are no mid-circuit measurements by checking that no qubit
    has an acquire instruction that is later followed by a pulse instruction.
    """

    def __init__(
        self,
        model: PhysicalHardwareModel,
        no_mid_circuit_measurement: bool | None = None,
        *args,
        **kwargs,
    ):
        """
        :param model: The hardware model.
        :param no_mid_circuit_measurement: Whether mid-circuit measurements are allowed.
            If None, uses the default from the QatConfig.
        """
        self.model = model
        self.no_mid_circuit_measurement = (
            no_mid_circuit_measurement
            if no_mid_circuit_measurement is not None
            else get_config().INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        )

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The intermediate representation (IR) :class:`InstructionBuilder`.
        """
        consumed_acquire_pc: set[str] = set()

        if not self.no_mid_circuit_measurement:
            return ir

        drive_acq_pc_map = {
            qubit.drive_pulse_channel.uuid: qubit.acquire_pulse_channel.uuid
            for qubit in self.model.qubits.values()
        }

        for instr in ir:
            if isinstance(instr, Acquire):
                consumed_acquire_pc.add(instr.target)

            # Check if we have a measure in the middle of the circuit somewhere.
            elif isinstance(instr, Pulse):
                acq_pc = drive_acq_pc_map.get(instr.target, None)

                if acq_pc and acq_pc in consumed_acquire_pc:
                    raise ValueError(
                        "Mid-circuit measurements currently unable to be used."
                    )
        return ir


class ReadoutValidation(ValidationPass):
    """Validates that the post-processing instructions do not have an invalid sequence.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.validate`.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        acquire_modes = {}
        for inst in ir:
            if isinstance(inst, Acquire):
                acquire_modes[inst.output_variable] = inst.mode
            if isinstance(inst, PostProcessing):
                acquire_mode = acquire_modes.get(inst.output_variable, None)
                self._post_processing_options_handling(inst, acquire_mode)
        return ir

    @staticmethod
    def _post_processing_options_handling(
        inst: PostProcessing, acquire_mode: AcquireMode | None
    ):
        if acquire_mode == AcquireMode.SCOPE and ProcessAxis.SEQUENCE in inst.axes:
            raise ValueError(
                "Invalid post-processing! Post-processing over SEQUENCE is "
                "not possible after the result is returned from hardware "
                "in SCOPE mode!"
            )
        elif acquire_mode == AcquireMode.INTEGRATOR and ProcessAxis.TIME in inst.axes:
            raise ValueError(
                "Invalid post-processing! Post-processing over TIME is not "
                "possible after the result is returned from hardware in "
                "INTEGRATOR mode!"
            )
        elif acquire_mode == AcquireMode.RAW:
            raise ValueError(
                "Invalid acquire mode! The live hardware doesn't support RAW acquire mode!"
            )
        elif acquire_mode is None:
            raise ValueError(
                f"No AcquireMode found with output variable {inst.output_variable},"
                f"ensure PostProcessing output_variable matches an Acquire output_variable with a"
                f"valid AcquireMode selected."
            )


class HardwareConfigValidity(ValidationPass):
    """Validates the :class:`CompilerConfig` against the hardware model."""

    def __init__(self, hardware_model: PhysicalHardwareModel, max_shots: int | None = None):
        """Instantiate the pass with a hardware model.

        :param hardware_model: The hardware model.
        :param max_shots: The maximum number of shots allowed for a single task.
                If None, uses the default from the QatConfig.
        """
        self.hardware_model = hardware_model
        self.max_shots = (
            max_shots if max_shots is not None else get_config().MAX_REPEATS_LIMIT
        )

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        self._validate_shots(compiler_config)
        self._validate_error_mitigation(self.hardware_model, compiler_config)
        return ir

    def _validate_shots(self, compiler_config: CompilerConfig):
        shots = compiler_config.repeats
        if shots is not None and shots > self.max_shots:
            raise ValueError(
                f"Number of shots in compiler config {compiler_config.repeats} exceeds max "
                f"number of shots {self.max_shots}."
            )

    def _validate_error_mitigation(
        self, hardware_model: PhysicalHardwareModel, compiler_config: CompilerConfig
    ):
        if (
            compiler_config.error_mitigation
            and compiler_config.error_mitigation != ErrorMitigationConfig.Empty
        ):
            if not hardware_model.error_mitigation.is_enabled:
                raise ValueError("Error mitigation not calibrated on this hardware model.")

            if ResultsFormatting.BinaryCount not in compiler_config.results_format:
                raise ValueError(
                    "BinaryCount format required for readout error mitigation."
                )


class FrequencySetupValidation(ValidationPass):
    """Validates the baseband frequencies and intermediate frequencies of pulse channels
    against the target data.
    """

    def __init__(self, model: PhysicalHardwareModel, target_data: TargetData):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model.
        :param target_data: Target-related information.
        """

        # Extract limits from the target data.
        self.qubit_lo_freq_limits = (
            target_data.QUBIT_DATA.pulse_channel_lo_freq_min,
            target_data.QUBIT_DATA.pulse_channel_lo_freq_max,
        )
        self.resonator_lo_freq_limits = (
            target_data.RESONATOR_DATA.pulse_channel_lo_freq_min,
            target_data.RESONATOR_DATA.pulse_channel_lo_freq_max,
        )
        self.qubit_if_freq_limits = (
            target_data.QUBIT_DATA.pulse_channel_if_freq_min,
            target_data.QUBIT_DATA.pulse_channel_if_freq_max,
        )
        self.resonator_if_freq_limits = (
            target_data.RESONATOR_DATA.pulse_channel_if_freq_min,
            target_data.RESONATOR_DATA.pulse_channel_if_freq_max,
        )
        self._physical_channel_data = self._get_physical_channel_data(model)
        self._model = model

    def _get_physical_channel_data(self, model: PhysicalHardwareModel) -> dict[str, dict]:
        """Extracts physical channel data from the model."""
        physical_channel_data = {}
        for q_index, qubit in model.qubits.items():
            physical_channel = qubit.physical_channel
            physical_channel_data[physical_channel.uuid] = (
                self._populate_physical_channel_data(physical_channel, False, q_index)
            )

            physical_channel = qubit.resonator.physical_channel
            physical_channel_data[physical_channel.uuid] = (
                self._populate_physical_channel_data(physical_channel, True, q_index)
            )

        return physical_channel_data

    def _populate_physical_channel_data(
        self, physical_channel: PhysicalChannel, is_resonator: bool, qubit_index: int
    ) -> dict:
        physical_channel_data = {
            "baseband_frequency": physical_channel.baseband.frequency,
            "is_resonator": is_resonator,
            "is_valid": self._validate_baseband_frequency(
                physical_channel.baseband.frequency, is_resonator
            ),
            "index": qubit_index,
        }
        return physical_channel_data

    def _validate_baseband_frequency(self, frequency, is_resonator: bool) -> bool:
        """Validates that the frequencies in the dictionary are within the specified range."""
        lower, upper = self._get_lo_freq_range(is_resonator)
        return bool(lower <= frequency <= upper)

    def _validate_pulse_channel_if(self, frequency: float, is_resonator: bool) -> bool:
        """Validates that the pulse channel frequencies are within the specified range."""
        lower, upper = self._get_if_freq_range(is_resonator)
        return bool(lower <= np.abs(frequency) <= upper)

    def _get_lo_freq_range(self, is_resonator: bool) -> tuple[float, float]:
        """Returns the frequency range for the given type of channel."""
        return (
            self.qubit_lo_freq_limits if not is_resonator else self.resonator_lo_freq_limits
        )

    def _get_if_freq_range(self, is_resonator: bool) -> tuple[float, float]:
        """Returns the IF frequency range for the given type of channel."""
        return (
            self.qubit_if_freq_limits if not is_resonator else self.resonator_if_freq_limits
        )

    def _find_physical_channel_violations(self, physical_channels) -> list[str] | None:
        """Finds violations in the physical channels based on their baseband frequencies."""
        violations = []
        for physical_channel in physical_channels:
            data = self._physical_channel_data.get(physical_channel, None)
            if not data["is_valid"]:
                freq_range = self._get_lo_freq_range(data["is_resonator"])
                baseband_freq = data["baseband_frequency"]
                name = "Resonator" if data["is_resonator"] else "Qubit"
                name += f" {data['index']}"
                violations.append(
                    f"Physical channel for {name} with baseband frequency "
                    f"{baseband_freq} is out of the valid "
                    f"range {freq_range}."
                )
        return violations

    def _find_pulse_channel_violations(
        self, pulse_channels: set[str], ir: QuantumInstructionBuilder
    ) -> list[str] | None:
        """Finds violations in the pulse channels based on their IF frequencies."""
        violations = []
        for pulse_channel_id in pulse_channels:
            pulse_channel = ir.get_pulse_channel(pulse_channel_id)
            physical_channel_id = pulse_channel.physical_channel_id
            physical_channel_data = self._physical_channel_data[physical_channel_id]

            if_freq = pulse_channel.frequency - physical_channel_data["baseband_frequency"]
            valid = self._validate_pulse_channel_if(
                if_freq, physical_channel_data["is_resonator"]
            )
            if not valid:
                name = "Resonator" if physical_channel_data["is_resonator"] else "Qubit"
                name += f" {physical_channel_data['index']} "
                if (
                    hw_pc := self._model.pulse_channel_with_id(pulse_channel_id)
                ) is not None:
                    pc_name = f"{hw_pc.pulse_type}"
                else:
                    pc_name = f"Custom({pulse_channel_id})"

                freq_range = self._get_if_freq_range(physical_channel_data["is_resonator"])
                violations.append(
                    f"The IF of {name} {pc_name} pulse channel has a value {if_freq}, "
                    f"which is outside of the the valid range {freq_range}."
                )

        return violations

    def run(
        self, ir: QuantumInstructionBuilder, res_mgr: ResultManager, *args, **kwargs
    ) -> QuantumInstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`QuantumInstructionBuilder`.
        :param res_mgr: The result manager containing the results of the analysis.
        """

        active_channel_res = res_mgr.lookup_by_type(ActivePulseChannelResults)
        pulse_channels = active_channel_res.targets
        physical_channels = set(
            ir.get_pulse_channel(pulse_channel).physical_channel_id
            for pulse_channel in pulse_channels
        )

        violations = self._find_physical_channel_violations(physical_channels)
        violations.extend(self._find_pulse_channel_violations(pulse_channels, ir))

        if len(violations) > 0:
            raise ValueError(
                "Frequency validation of the hardware model against the target data failed "
                "with the following violations: \n" + "\n".join(violations)
            )
        return ir


class ReturnSanitisationValidation(ValidationPass):
    """Validates that the IR has a :class:`Return` instruction."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions.")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions.")
        return ir


class RepeatSanitisationValidation(ValidationPass):
    """Checks if the builder has a :class:`Repeat` instruction and warns if none exists."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        for inst in ir.instructions:
            if isinstance(inst, Repeat):
                return ir

        log.warning("Could not find any repeat instructions.")
        return ir


PydInstructionValidation = InstructionValidation
PydReadoutValidation = ReadoutValidation
PydDynamicFrequencyValidation = DynamicFrequencyValidation
PydHardwareConfigValidity = HardwareConfigValidity
PydFrequencySetupValidation = FrequencySetupValidation
PydNoMidCircuitMeasurementValidation = NoMidCircuitMeasurementValidation
PydReturnSanitisationValidation = ReturnSanitisationValidation
PydRepeatSanitisationValidation = RepeatSanitisationValidation
