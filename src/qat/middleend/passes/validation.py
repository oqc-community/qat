# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from collections import defaultdict
from functools import singledispatchmethod

import numpy as np
from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.instructions import FrequencySet, FrequencyShift, Instruction, Repeat, Return
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    PostProcessing,
    ProcessAxis,
    Pulse,
)
from qat.middleend.passes.analysis import ActivePulseChannelResults
from qat.model.device import PhysicalChannel
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


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
        ifs: dict[str, list[float]],
        physical_channel_ids: dict[str, str],
    ):
        pass

    @_calculate_frequency.register(FrequencyShift)
    def _(
        self,
        instruction: FrequencyShift,
        ifs: dict[str, list[float]],
        physical_channel_ids: dict[str, str],
    ):
        frequencies = ifs[instruction.target]
        physical_channel = self.model.physical_channel_for_pulse_channel_id(
            instruction.target
        )
        freq = (
            frequencies[-1]
            if len(frequencies) > 0
            else (
                self.model.pulse_channel_with_id(instruction.target).frequency
                - physical_channel.baseband.frequency
            )
        )
        frequencies.append(freq + instruction.frequency)
        physical_channel_ids[instruction.target] = physical_channel.uuid

    @_calculate_frequency.register(FrequencySet)
    def _(
        self,
        instruction: FrequencySet,
        ifs: dict[str, list[float]],
        physical_channel_ids: dict[str, str],
    ):
        physical_channel = self.model.physical_channel_for_pulse_channel_id(
            instruction.target
        )
        ifs[instruction.target].append(
            instruction.frequency - physical_channel.baseband.frequency
        )
        physical_channel_ids[instruction.target] = physical_channel.uuid

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        ifs = defaultdict(list)
        physical_channel_ids = defaultdict(str)
        for instruction in ir.instructions:
            self._calculate_frequency(instruction, ifs, physical_channel_ids)

        violations = []
        for pulse_channel_id, if_values in ifs.items():
            physical_channel_id = physical_channel_ids[pulse_channel_id]
            if if_violations := self._validate_frequency_shifts(
                physical_channel_id, if_values
            ):
                pulse_channel = self.model.pulse_channel_with_id(pulse_channel_id)
                device = self.model.device_for_pulse_channel_id(pulse_channel_id)
                device_name = "Qubit "
                if self._is_resonator[physical_channel_id]:
                    device_name = "Resonator "
                    device = next(
                        filter(lambda qubit: qubit.resonator is device, self.model.qubits)
                    )
                device_name += str(self.model.index_of_qubit(device))
                violations.append(
                    f"The IF of {device_name} {pulse_channel.pulse_type} pulse channel is set or "
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

        self._pulse_channel_data = self._get_pulse_channel_data(model)
        self._physical_channel_data = self._get_physical_channel_data(model)

    def _get_pulse_channel_data(self, model: PhysicalHardwareModel) -> dict[str, dict]:
        """Extracts pulse channel data from the model."""
        pulse_channel_data = {}

        for qubit in model.qubits.values():
            pulse_channel_data = self._populate_pulse_channel_dict(
                pulse_channel_data, qubit.all_pulse_channels, qubit.physical_channel, False
            )
            pulse_channel_data = self._populate_pulse_channel_dict(
                pulse_channel_data,
                qubit.resonator.all_pulse_channels,
                qubit.resonator.physical_channel,
                True,
            )
        return pulse_channel_data

    def _populate_pulse_channel_dict(
        self,
        pulse_channel_data: dict,
        pulse_channels: list[Pulse],
        physical_channel: PhysicalChannel,
        is_resonator: bool,
    ) -> dict:
        for pulse_channel in pulse_channels:
            if_freq = pulse_channel.frequency - physical_channel.baseband.frequency
            pulse_channel_data[pulse_channel.uuid] = {
                "physical_channel": physical_channel.uuid,
                "is_valid": self._validate_pulse_channel_if(if_freq, is_resonator),
                "is_resonator": is_resonator,
                "if_frequency": if_freq,
                "pulse_type": pulse_channel.pulse_type,
            }
        return pulse_channel_data

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

    def _find_pulse_channel_violations(self, pulse_channels) -> list[str] | None:
        """Finds violations in the pulse channels based on their IF frequencies."""
        violations = []
        for pulse_channel in pulse_channels:
            data = self._pulse_channel_data[pulse_channel]

            # TODO: Add support for custom pulse channels COMPILER-698
            valid = data["is_valid"]

            if not valid:
                freq_range = self._get_if_freq_range(data["is_resonator"])
                if_freq = data["if_frequency"]
                index = self._physical_channel_data[data["physical_channel"]]["index"]
                name = "Resonator" if data["is_resonator"] else "Qubit"
                name += f" {index} {data['pulse_type']}"
                violations.append(
                    f"The IF of {name} pulse channel has a value {if_freq}, "
                    f"which is outside of the the valid range {freq_range}."
                )

        return violations

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager containing the results of the analysis.
        """

        active_channel_res = res_mgr.lookup_by_type(ActivePulseChannelResults)
        pulse_channels = active_channel_res.targets
        physical_channels = set(
            [
                self._pulse_channel_data[pulse_channel]["physical_channel"]
                for pulse_channel in pulse_channels
            ]
        )

        violations = self._find_physical_channel_violations(physical_channels)
        violations.extend(self._find_pulse_channel_violations(pulse_channels))

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


PydReadoutValidation = ReadoutValidation
PydDynamicFrequencyValidation = DynamicFrequencyValidation
PydHardwareConfigValidity = HardwareConfigValidity
PydFrequencySetupValidation = FrequencySetupValidation
PydNoMidCircuitMeasurementValidation = NoMidCircuitMeasurementValidation
PydReturnSanitisationValidation = ReturnSanitisationValidation
PydRepeatSanitisationValidation = RepeatSanitisationValidation
