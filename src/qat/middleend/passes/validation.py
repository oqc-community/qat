# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import List

import numpy as np
from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
from qat.ir.measure import Acquire as PydAcquire
from qat.ir.measure import Pulse as PydPulse
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.model.target_data import TargetData
from qat.purr.backends.live import LiveHardwareModel
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
    Sweep,
    Variable,
)


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


class PydNoMidCircuitMeasurementValidation(ValidationPass):
    """
    Validates that there are no mid-circuit measurements by checking that no qubit
    has an acquire instruction that is later followed by a pulse instruction.
    """

    def __init__(
        self,
        model: PydHardwareModel,
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
            if isinstance(instr, PydAcquire):
                consumed_acquire_pc.add(instr.target)

            # Check if we have a measure in the middle of the circuit somewhere.
            elif isinstance(instr, PydPulse):
                acq_pc = drive_acq_pc_map.get(instr.target, None)

                if acq_pc and acq_pc in consumed_acquire_pc:
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


class PydHardwareConfigValidity(ValidationPass):
    """Validates the :class:`CompilerConfig` against the hardware model."""

    def __init__(self, hardware_model: PydHardwareModel, max_shots: int | None = None):
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
        self, hardware_model: PydHardwareModel, compiler_config: CompilerConfig
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
