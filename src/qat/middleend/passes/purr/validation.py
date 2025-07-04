# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import List

import numpy as np
from compiler_config.config import CompilerConfig, QuantumResultsFormat, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
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

        model = self.hardware

        if not isinstance(model, LiveHardwareModel):
            return ir

        consumed_qubits: List[str] = []
        chanbits_map = {}
        for inst in ir.instructions:
            if isinstance(inst, PostProcessing):
                if (
                    inst.acquire.mode == AcquireMode.SCOPE
                    and ProcessAxis.SEQUENCE in inst.axes
                ):
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


class FrequencyValidation(ValidationPass):
    """This validation pass checks two things:

    #. Frequency shifts do not move the frequency of a pulse channel outside of its
       allowed range.
    #. Frequency shifts do not occur on pulse channels that have a fixed IF, or share a
       physical channel with a pulse channel that has a fixed IF.
    """

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model.
        :param target_data: Target-related information.
        """

        # TODO: replace with new hardware models as our refactors mature.
        self.model = model
        qubit_freq_limits = {
            pulse_ch: (
                target_data.QUBIT_DATA.pulse_channel_lo_freq_min,
                target_data.QUBIT_DATA.pulse_channel_lo_freq_max,
            )
            for qubit in model.qubits
            for pulse_ch in qubit.pulse_channels.values()
        }
        res_freq_limits = {
            pulse_ch: (
                target_data.RESONATOR_DATA.pulse_channel_lo_freq_min,
                target_data.RESONATOR_DATA.pulse_channel_lo_freq_max,
            )
            for res in model.resonators
            for pulse_ch in res.pulse_channels.values()
        }
        self.freq_shift_limits = qubit_freq_limits | res_freq_limits

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        freqshifts = [inst for inst in ir.instructions if isinstance(inst, FrequencyShift)]
        targets = set([inst.channel for inst in freqshifts])
        self.validate_frequency_ranges(targets, freqshifts, self.freq_shift_limits)
        self.validate_no_freqshifts_on_fixed_if(targets)
        return ir

    @staticmethod
    def validate_frequency_ranges(
        targets: List[PulseChannel],
        freqshifts: List[FrequencyShift],
        freq_limits: dict[PulseChannel, tuple[int, int]],
    ):
        """Validates that a pulse channel remains within its allowed frequency range.

        :param targets: List of pulse channels to validate.
        :param freqshifts: List of frequency shift instructions.
        """
        for target in targets:
            freq_shifts = [inst.frequency for inst in freqshifts if inst.channel == target]
            freqs = target.frequency + np.cumsum(freq_shifts)
            min_freq, max_freq = freq_limits[target]
            violations = np.logical_or(freqs > max_freq, freqs < min_freq)
            if np.any(violations):
                raise ValueError(
                    f"Frequency shifts will change the pulse channel frequency to fall "
                    f"out of the allowed range between {min_freq} and "
                    f"{max_freq} for pulse channel {target.full_id()}."
                )

    def validate_no_freqshifts_on_fixed_if(self, targets: List[PulseChannel]):
        """Validates that frequency shifts do not occur on pulse channels with fixed IFs.

        :param targets: List of pulse channels to validate.
        """
        for target in targets:
            for channel in self.model.get_pulse_channels_from_physical_channel(
                target.physical_channel
            ):
                if channel.fixed_if:
                    raise NotImplementedError(
                        "Hardware does not currently support frequency shifts on the "
                        f"physical channel {target.physical_channel}."
                    )


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
