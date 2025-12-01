# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from numbers import Number

import numpy as np
from compiler_config.config import CompilerConfig, QuantumResultsFormat, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
from qat.core.result_base import ResultManager
from qat.middleend.passes.purr.analysis import ActiveChannelResults
from qat.model.target_data import TargetData
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.backends.qiskit_simulator import QiskitBuilder
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, Qubit
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    FrequencyShift,
    PostProcessing,
    ProcessAxis,
    Pulse,
    PulseShapeType,
    Repeat,
    Return,
    Sweep,
    Synchronize,
    Variable,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PhysicalChannelAmplitudeValidation(ValidationPass):
    """
    Validates that the total amplitude of pulses on a physical channel does not exceed
    allowed levels.

    Data is stored in a nested dict structure

    .. code-block:: python

        { 'CH1': {
                      CH1.Q0.drive: 0.2+0.5j,
                      CH1.Q0.Q1.cross_resonsance: 0.4+0.0j,
                  },
          'CH2': { CH2.R0.measure: 0.9-0.6j },
        }

    This pass requires all :class:`Pulse` s to have been lowered to either :class:`PulseShapeType.SQUARE`
    or a :class:`CustomPulse` with the evaluated pulse `samples`. Any non lowered pulses with
    other types may cause the pipeline to fail.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The intermediate representation (IR) :class:`InstructionBuilder`.
        :raises ValueError: for :class:`Pulse` of non :class:`PulseShapeType.SQUARE` type.
        """
        # TODO: Evaluate if this would be better if operating on `PartitionedIR` class
        # instead of `InstructionBuilder`

        phys_chs_amps: dict[str, dict[str, complex]] = {}
        for pulse_channel in ir.model.pulse_channels.values():
            pulse_ch_amps = phys_chs_amps.setdefault(pulse_channel.physical_channel_id, {})
            pulse_ch_amps[pulse_channel] = 0.0

        for instruction in ir.instructions:
            if isinstance(instruction, Pulse | CustomPulse):
                pulse_ch_amps = phys_chs_amps[
                    phys_chan_id := instruction.channel.physical_channel_id
                ]
                pulse_chan = instruction.channel
                if isinstance(instruction, CustomPulse):
                    pulse_shape = np.array(instruction.samples)

                    # TODO: This may be part of the lowering or needed here.
                    # Needs to be determined and potentially adjusted as part of:
                    # COMPILER-???
                    # if not instruction.ignore_channel_scale:
                    #    pulse_shape *= instruction.channel.scale

                    pulse_ch_amps[pulse_chan] = complex(
                        max(np.abs(pulse_shape.real)) + 1j * max(np.abs(pulse_shape.imag))
                    )
                else:
                    if not instruction.shape == PulseShapeType.SQUARE:
                        raise ValueError(
                            f"Can not validate amplitude of un-lowered {instruction.shape} Pulse."
                        )
                    pulse_amp = complex(
                        instruction.scale_factor
                        * instruction.amp
                        * np.exp(1.0j * instruction.phase)
                    )
                    if not instruction.ignore_channel_scale:
                        pulse_amp *= instruction.channel.scale

                    pulse_ch_amps[pulse_chan] = pulse_amp

                total_amp = sum(pulse_ch_amps.values())

                if np.abs(np.real(total_amp)) > 1 or np.abs(np.imag(total_amp)) > 1:
                    raise ValueError(
                        f"Overflow error detect on {phys_chan_id}!"
                        " The sum of logical channels onto a physical channel is too high!"
                    )
            elif isinstance(instruction, Synchronize):
                handled_phys_chan_ids = []
                for target in instruction.quantum_targets:
                    phys_chan_id = target.physical_channel_id
                    if phys_chan_id in handled_phys_chan_ids:
                        continue

                    tmp = phys_chs_amps[phys_chan_id]
                    if all([chan in instruction.quantum_targets for chan in tmp.keys()]):
                        phys_chs_amps[phys_chan_id] = dict.fromkeys(tmp.keys(), 0.0)
                    handled_phys_chan_ids.append(phys_chan_id)

        return ir


class QiskitResultsFormatValidation(ValidationPass):
    """Validates the results format contains `BinaryCount`, and throws a warning if not."""

    def run(self, ir: QiskitBuilder, *args, compiler_config: CompilerConfig, **kwargs):
        """
        :param ir: The Qiskit instruction builder.
        :param compiler_config: The compiler config contains the results format.
        :return: The instruction builder, unaltered.
        """
        results_format = compiler_config.results_format
        format_flags = (
            results_format.transforms
            if isinstance(results_format, QuantumResultsFormat)
            else results_format
        )
        if format_flags is None or ResultsFormatting.BinaryCount not in format_flags:
            log.warning(
                "The results formatting `BinaryCount` was not found in the formatting "
                "flags. Please note that the Qiskit runtime only currently supports "
                "results returned as a binary count."
            )
        return ir


class InstructionValidation(ValidationPass):
    """Validates instructions against the hardware.

    Extracted from :mod:`qat.purr.compiler.execution.QuantumExecutionEngine.validate`.
    """

    def __init__(
        self,
        target_data: TargetData,
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
        self.pulse_duration_limits = (
            pulse_duration_limits
            if pulse_duration_limits is not None
            else get_config().INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS
        )

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        instruction_length = len(ir.instructions)
        if instruction_length > self.instruction_memory_size:
            raise ValueError(
                f"Program with {instruction_length} instructions too large to be run in a single block on current hardware."
            )

        for inst in ir.instructions:
            if isinstance(inst, Acquire) and not inst.channel.acquire_allowed:
                raise ValueError(
                    f"Cannot perform an acquire on the physical channel with id "
                    f"{inst.channel.physical_channel}"
                )
            if isinstance(inst, (Pulse, CustomPulse)):
                duration = inst.duration
                if isinstance(duration, Number) and duration > self.pulse_duration_max:
                    if self.pulse_duration_limits:
                        # Do not throw error if we specifically disabled the limit checks.
                        # TODO: Add a lower bound for the pulse duration limits as well in a later PR,
                        # which is specific to each hardware model and can be stored as a member variables there.
                        raise ValueError(
                            f"Max Waveform width is {self.pulse_duration_max} s "
                            f"given: {inst.duration} s"
                        )
                elif isinstance(duration, Variable):
                    values = next(
                        iter(
                            [
                                sw.variables[duration.name]
                                for sw in ir.instructions
                                if isinstance(sw, Sweep)
                                and duration.name in sw.variables.keys()
                            ]
                        )
                    )
                    if np.max(values) > self.pulse_duration_max:
                        if self.pulse_duration_limits:
                            raise ValueError(
                                f"Max Waveform width is {self.pulse_duration_max} s "
                                f"given: {values} s"
                            )
        return ir


class ReadoutValidation(ValidationPass):
    """Validates that there are no mid-circuit measurements, and that the post-processing
    instructions do not have an invalid sequence.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.validate`.
    """

    # TODO: break this down into smaller passes!

    def __init__(
        self,
        hardware: QuantumHardwareModel,
        no_mid_circuit_measurement: bool | None = None,
        *args,
        **kwargs,
    ):
        """
        :param hardware: The hardware model is needed to check for mid-circuit measurments.
        :param no_mid_circuit_measurement: Whether mid-circuit measurements are allowed.
            If None, uses the default from the QatConfig.
        """
        self.hardware = hardware
        self.no_mid_circuit_measurement = (
            no_mid_circuit_measurement
            if no_mid_circuit_measurement is not None
            else get_config().INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        )

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        # TODO -COMPILER-851 - Specify ctrl HW features in TargetData
        enable_hw_averaging = kwargs.get("enable_hw_averaging", False)

        model = self.hardware

        if not isinstance(model, LiveHardwareModel):
            return ir

        consumed_qubits: list[str] = []
        chanbits_map = {}
        for inst in ir.instructions:
            if isinstance(inst, PostProcessing):
                if (
                    inst.acquire.mode == AcquireMode.SCOPE
                    and ProcessAxis.SEQUENCE in inst.axes
                ):
                    if not enable_hw_averaging:
                        raise ValueError(
                            "Invalid post-processing! Post-processing over SEQUENCE is "
                            "not possible after the result is returned from hardware "
                            "in SCOPE mode!"
                        )
                elif (
                    inst.acquire.mode == AcquireMode.INTEGRATOR
                    and ProcessAxis.TIME in inst.axes
                ):
                    raise ValueError(
                        "Invalid post-processing! Post-processing over TIME is not "
                        "possible after the result is returned from hardware in "
                        "INTEGRATOR mode!"
                    )
                elif inst.acquire.mode == AcquireMode.RAW:
                    raise ValueError(
                        "Invalid acquire mode! The live hardware doesn't support "
                        "RAW acquire mode!"
                    )

            # Check if we've got a measure in the middle of the circuit somewhere.
            elif self.no_mid_circuit_measurement:
                if isinstance(inst, Acquire):
                    for qbit in model.qubits:
                        if qbit.get_acquire_channel() == inst.channel:
                            consumed_qubits.append(qbit)
                elif isinstance(inst, Pulse):
                    # Find target qubit from instruction and check whether it's been
                    # measured already.
                    acquired_qubits = [
                        (
                            (
                                chanbits_map[chanbit]
                                if chanbit in chanbits_map
                                else chanbits_map.setdefault(
                                    chanbit,
                                    model._resolve_qb_pulse_channel(chanbit)[0],
                                )
                            )
                            in consumed_qubits
                        )
                        for chanbit in inst.quantum_targets
                        if isinstance(chanbit, (Qubit, PulseChannel))
                    ]

                    if any(acquired_qubits):
                        raise ValueError(
                            "Mid-circuit measurements currently unable to be used."
                        )
        return ir


class HardwareConfigValidity(ValidationPass):
    """Validates the :class:`CompilerConfig` against the hardware model."""

    def __init__(self, hardware_model: QuantumHardwareModel):
        """Instantiate the pass with a hardware model.

        :param hardware_model: The hardware model.
        """
        self.hardware_model = hardware_model

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        """
        :param compiler_config: The config containing compilation settings provided as a
            keyword argument.
        """
        compiler_config.validate(self.hardware_model)
        return ir


class FrequencySetupValidation(ValidationPass):
    """Validates the baseband frequencies and intermediate frequencies of pulse channels
    against the target data.
    """

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
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

        # Calculate mappings for physical and pulse channels for quick lookup
        self._baseband_frequencies = self._create_baseband_frequency_map(model)
        self._is_resonator = self._create_resonator_map(model)
        self._pulse_channel_ifs = self._create_pulse_channel_if_map(model)
        self._pulse_to_physical_channel = (
            self._create_pulse_channel_to_physical_channel_map(model)
        )

        # Do validation prior to runtime
        self._baseband_valid = self._validate_baseband_frequencies(
            self._baseband_frequencies,
            self._is_resonator,
            self.qubit_lo_freq_limits,
            self.resonator_lo_freq_limits,
        )

        self._pulse_channels_valid = self._validate_pulse_channel_ifs(
            self._pulse_channel_ifs,
            self._is_resonator,
            self._pulse_to_physical_channel,
            self.qubit_if_freq_limits,
            self.resonator_if_freq_limits,
        )

    @staticmethod
    def _create_baseband_frequency_map(model: QuantumHardwareModel) -> dict[str, float]:
        """Creates a map of baseband frequencies for each physical channel in the model."""
        return {
            device.physical_channel.id: device.physical_channel.baseband_frequency
            for device in model.quantum_devices.values()
        }

    @staticmethod
    def _create_resonator_map(model: QuantumHardwareModel) -> dict[str, bool]:
        """Creates a map to lookup if physical channels belong to resonators.."""
        return {
            device.physical_channel.id: not isinstance(device, Qubit)
            for device in model.quantum_devices.values()
        }

    @staticmethod
    def _create_pulse_channel_if_map(model: QuantumHardwareModel) -> dict[str, float]:
        """Creates a map of pulse channel intermediate frequencies for each pulse channel
        in the model."""
        return {
            pulse_channel.partial_id(): (
                pulse_channel.frequency - pulse_channel.physical_channel.baseband_frequency
            )
            for pulse_channel in model.pulse_channels.values()
        }

    @staticmethod
    def _create_pulse_channel_to_physical_channel_map(
        model: QuantumHardwareModel,
    ) -> dict[str, str]:
        """Creates a map of pulse channels to their physical channels."""
        return {
            pulse_channel.partial_id(): pulse_channel.physical_channel.id
            for pulse_channel in model.pulse_channels.values()
        }

    @staticmethod
    def _validate_baseband_frequencies(
        frequencies: dict[str, float],
        is_resonator: dict[str, bool],
        qubit_lo_freq_limits: tuple[float, float],
        resonator_lo_freq_limits: tuple[float, float],
    ) -> dict[str, bool]:
        """Validates that the frequencies in the dictionary are within the specified range."""

        baseband_validation = {}
        for key, frequency in frequencies.items():
            lower, upper = (
                qubit_lo_freq_limits if not is_resonator[key] else resonator_lo_freq_limits
            )
            baseband_validation[key] = lower <= frequency <= upper
        return baseband_validation

    @staticmethod
    def _validate_pulse_channel_ifs(
        pulse_channel_ifs: dict[str, float],
        is_resonator: dict[str, bool],
        pulse_channel_to_physical_channel_map: dict[str, str],
        qubit_if_freq_limits: tuple[float, float],
        resonator_if_freq_limits: tuple[float, float],
    ) -> dict[str, bool]:
        """Validates that the pulse channel frequencies are within the specified range."""

        pulse_channel_validation = {}
        for pulse_channel, frequency in pulse_channel_ifs.items():
            lower, upper = (
                qubit_if_freq_limits
                if not is_resonator[pulse_channel_to_physical_channel_map[pulse_channel]]
                else resonator_if_freq_limits
            )
            pulse_channel_validation[pulse_channel] = bool(
                lower <= np.abs(frequency) <= upper
            )
        return pulse_channel_validation

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        active_channel_res = res_mgr.lookup_by_type(ActiveChannelResults)
        pulse_channels = active_channel_res.targets
        physical_channels = set(
            [pulse_channel.physical_channel for pulse_channel in pulse_channels]
        )

        violations = []
        for physical_channel in physical_channels:
            if not self._baseband_valid[physical_channel.id]:
                freq_range = (
                    self.qubit_lo_freq_limits
                    if not self._is_resonator[physical_channel.id]
                    else self.resonator_lo_freq_limits
                )
                violations.append(
                    f"Physical channel {physical_channel.full_id()} with baseband frequency "
                    f"{self._baseband_frequencies[physical_channel.id]} is out of the valid "
                    f"range {freq_range}."
                )

        for pulse_channel in pulse_channels:
            if pulse_channel.partial_id() in self._pulse_channels_valid:
                valid = self._pulse_channels_valid[pulse_channel.partial_id()]
                if_freq = self._pulse_channel_ifs[pulse_channel.partial_id()]
            else:
                # We have to handle custom pulse channels manually
                if_freq = (
                    pulse_channel.frequency
                    - pulse_channel.physical_channel.baseband_frequency
                )
                if self._is_resonator[pulse_channel.physical_channel.id]:
                    lower, upper = self.resonator_if_freq_limits
                else:
                    lower, upper = self.qubit_if_freq_limits
                valid = bool(lower <= np.abs(if_freq) <= upper)

            if not valid:
                freq_range = (
                    self.qubit_if_freq_limits
                    if not self._is_resonator[pulse_channel.physical_channel.id]
                    else self.resonator_if_freq_limits
                )
                violations.append(
                    f"Pulse channel {pulse_channel.full_id()} with IF {if_freq} is out of "
                    f"the valid range {freq_range}."
                )

        if len(violations) > 0:
            raise ValueError(
                "Frequency validation of the hardware model against the target data failed "
                "with the following violations: \n" + "\n".join(violations)
            )
        return ir


class DynamicFrequencyValidation(ValidationPass):
    """Validates the setting or shifting frequencies does not move the intermediate
    frequency of a pulse channel outside the allowed limits."""

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
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
    def _create_resonator_map(model: QuantumHardwareModel) -> dict[str, bool]:
        """Creates a map to lookup if physical channels belong to resonators.."""
        return {
            device.physical_channel.id: not isinstance(device, Qubit)
            for device in model.quantum_devices.values()
        }

    def _validate_frequency_shifts(self, pulse_channel: PulseChannel, ifs: list[float]):
        """Validates that the frequency shifts do not exceed the allowed limits."""

        if_limits = (
            self.qubit_if_freq_limits
            if not self._is_resonator[pulse_channel.physical_channel.id]
            else self.resonator_if_freq_limits
        )

        return [
            if_value
            for if_value in ifs
            if not (if_limits[0] <= np.abs(if_value) <= if_limits[1])
        ]

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        ifs = defaultdict(list)
        for instruction in ir.instructions:
            if isinstance(instruction, FrequencyShift):
                shifts = ifs[instruction.channel]
                freq = (
                    shifts[-1]
                    if len(shifts) > 0
                    else instruction.channel.frequency
                    - instruction.channel.physical_channel.baseband_frequency
                )
                shifts.append(freq + instruction.frequency)
            # TODO: Add support for frequency set (COMPILER-644)

        violations = []
        for pulse_channel, if_values in ifs.items():
            if if_violations := self._validate_frequency_shifts(pulse_channel, if_values):
                violations.append(
                    f"The IF of pulse channel {pulse_channel.full_id()} is frequency "
                    f"shifted to values {if_violations} that exceed the allowed limits."
                )

        if len(violations) > 0:
            raise ValueError(
                "Dynamic frequency validation failed with the following violations:\n"
                + "\n".join(violations)
            )
        return ir


class FixedIntermediateFrequencyValidation(ValidationPass):
    """Checks that no frequency shifts or sets are applied to pulse channels that have
    a fixed intermediate frequency, or share a physical channel with a pulse channel."""

    def __init__(self, model: QuantumHardwareModel):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model.
        """
        self.model = model
        self._fixed_ifs = self._create_fixed_if_map(model)

    @staticmethod
    def _create_fixed_if_map(model: QuantumHardwareModel) -> dict[str, bool]:
        """Creates a map of fixed intermediate frequencies for each pulse channel in the model."""
        fixed_ifs = {}
        for pulse_channel in model.pulse_channels.values():
            fixed_ifs[pulse_channel.physical_channel.id] = (
                pulse_channel.fixed_if
                or fixed_ifs.get(pulse_channel.physical_channel.id, False)
            )

        return fixed_ifs

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        violations = set()
        for instruction in ir.instructions:
            if isinstance(instruction, FrequencyShift):
                pulse_channel = instruction.channel
                if self._fixed_ifs[pulse_channel.physical_channel.id]:
                    violations.add(
                        f"Pulse channel {pulse_channel.full_id()} has a fixed IF and "
                        "cannot be frequency shifted."
                    )

        if len(violations) > 0:
            raise ValueError(
                "Fixed intermediate frequency validation failed with the following "
                "violations:\n".join(list(violations))
            )
        return ir


class ReturnSanitisationValidation(ValidationPass):
    """Validates that the IR has a :class:`Return` instruction."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions")
        return ir


class RepeatSanitisationValidation(ValidationPass):
    """Checks if the builder has a :class:`Repeat` instruction and warns if none exists."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        if not repeats:
            log.warning("Could not find any repeat instructions.")
        return ir
