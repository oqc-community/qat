# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from collections import defaultdict
from functools import singledispatchmethod

import numpy as np
from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.instructions import FrequencySet, FrequencyShift, Instruction, Repeat, Return
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    PostProcessing,
    ProcessAxis,
    Pulse,
)
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
        if compiler_config.repeats > self.max_shots:
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
PydNoMidCircuitMeasurementValidation = NoMidCircuitMeasurementValidation
PydReturnSanitisationValidation = ReturnSanitisationValidation
PydRepeatSanitisationValidation = RepeatSanitisationValidation
