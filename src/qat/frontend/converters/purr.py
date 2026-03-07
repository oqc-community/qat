# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from functools import cached_property, singledispatchmethod
from math import isclose
from warnings import warn

import qat.ir.instructions as Instructions
import qat.ir.measure as MeasureInstructions
import qat.ir.waveforms as WaveformInstructions
import qat.purr.compiler.instructions as PurrInstructions
from qat.ir.builder_factory import BuilderFactory
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.model.device import PhysicalChannel, ResonatorPhysicalChannel
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.validators import MismatchingHardwareModelException
from qat.purr.compiler.builders import InstructionBuilder as PurrInstructionBuilder
from qat.purr.compiler.devices import (
    ChannelType as PurrChannelType,
    PhysicalChannel as PurrPhysicalChannel,
    PulseChannel as PurrPulseChannel,
    PulseChannelView as PurrPulseChannelView,
    QuantumDevice as PurrQuantumDevice,
    Qubit as PurrQubit,
    Resonator as PurrResonator,
)
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import PulseShapeType
from qat.purr.utils.logger import get_default_logger

logger = get_default_logger()

waveform_map: dict[PulseShapeType, type[WaveformInstructions.AbstractWaveform]] = {
    PulseShapeType.SQUARE: WaveformInstructions.SquareWaveform,
    PulseShapeType.GAUSSIAN: WaveformInstructions.GaussianWaveform,
    PulseShapeType.SOFT_SQUARE: WaveformInstructions.SoftSquareWaveform,
    PulseShapeType.BLACKMAN: WaveformInstructions.BlackmanWaveform,
    PulseShapeType.SETUP_HOLD: WaveformInstructions.SetupHoldWaveform,
    PulseShapeType.SOFTER_SQUARE: WaveformInstructions.SofterSquareWaveform,
    PulseShapeType.EXTRA_SOFT_SQUARE: WaveformInstructions.ExtraSoftSquareWaveform,
    PulseShapeType.SOFTER_GAUSSIAN: WaveformInstructions.SofterGaussianWaveform,
    PulseShapeType.ROUNDED_SQUARE: WaveformInstructions.RoundedSquareWaveform,
    PulseShapeType.GAUSSIAN_DRAG: WaveformInstructions.DragGaussianWaveform,
    PulseShapeType.GAUSSIAN_ZERO_EDGE: WaveformInstructions.GaussianZeroEdgeWaveform,
    PulseShapeType.GAUSSIAN_SQUARE: WaveformInstructions.GaussianSquareWaveform,
    PulseShapeType.SECH: WaveformInstructions.SechWaveform,
    PulseShapeType.SIN: WaveformInstructions.SinWaveform,
    PulseShapeType.COS: WaveformInstructions.CosWaveform,
}


class HardwareModelMapper:
    """Helper class to deal with mapping of PuRR to Pydantic hardware models.

    It offers utilities to validate the physical properties of the hardware models to ensure
    compatibility, and to convert PuRR pulse channels into QAT IR pulse channels
    """

    def __init__(self, legacy_model: QuantumHardwareModel, model: PhysicalHardwareModel):
        self.legacy_model = legacy_model
        self.model = model
        self._pulse_channel_map: dict[str, str] = dict()

    def validate_physical_properties(self):
        """Validates the physical properties of the hardware model in the IR against the model
        provided to ensure compatibility."""

        self._validate_qubit_mapping()
        self._validate_coupling_mapping()

        errors_reported: list[str] = []
        for legacy_qubit in self.legacy_model.qubits:
            pydantic_qubit = self.model.qubit_with_index(legacy_qubit.index)
            self._validate_physical_channel_parity(
                errors_reported,
                legacy_qubit.physical_channel,
                pydantic_qubit.physical_channel,
            )
            self._validate_physical_channel_parity(
                errors_reported,
                legacy_qubit.measure_device.physical_channel,
                pydantic_qubit.resonator.physical_channel,
            )

        if len(errors_reported) > 0:
            raise MismatchingHardwareModelException(
                "The provided PuRR instruction builder is not compatible with the target "
                "hardware model, due to the following errors: \n"
                + "\n".join(errors_reported)
            )

    @cached_property
    def physical_channel_map(self) -> dict[str, str]:
        """Creates a mapping of physical channel full ids from the legacy model to the
        pydantic model, to allow for validation of pulse channel properties."""

        mapping = dict()
        for legacy_qubit in self.legacy_model.qubits:
            qubit = self.model.qubit_with_index(legacy_qubit.index)
            mapping[legacy_qubit.physical_channel.full_id()] = qubit.physical_channel.uuid
            mapping[legacy_qubit.measure_device.physical_channel.full_id()] = (
                qubit.resonator.physical_channel.uuid
            )
        return mapping

    def get_pulse_channel_id(self, pulse_channel: PurrPulseChannel):
        """Gets the id of the pulse channel to use in the IR, resolving it to the physical
        model if possible."""

        if (id_ := pulse_channel.partial_id()) not in self._pulse_channel_map:
            self._pulse_channel_map[id_] = self._resolve_pulse_channel(pulse_channel)
        return self._pulse_channel_map[id_]

    def pulse_channel_mapping_exists(self, pulse_channel: PurrPulseChannel):
        return pulse_channel.partial_id() in self._pulse_channel_map

    def _validate_qubit_mapping(self):
        legacy_qubits = self.legacy_model.qubits
        if len(legacy_qubits) != len(self.model.qubits):
            raise MismatchingHardwareModelException(
                f"Models have a different number of qubits, "
                f"{len(legacy_qubits)} does not equal {len(self.model.qubits)}."
            )

        qubit_indices = {qubit.index for qubit in legacy_qubits}
        if qubit_indices != set(self.model.qubits.keys()):
            raise MismatchingHardwareModelException(
                f"Models have different qubit indices, {qubit_indices} does not equal "
                f"{set(self.model.qubits.keys())}."
            )

    def _validate_coupling_mapping(self):
        legacy_couplings = set(
            coupling.direction for coupling in self.legacy_model.qubit_direction_couplings
        )
        couplings = set(
            (control, target)
            for control, targets in self.model.logical_connectivity.items()
            for target in targets
        )

        if legacy_couplings != couplings:
            raise MismatchingHardwareModelException(
                f"Models have different qubit couplings, {legacy_couplings} does not equal "
                f"{couplings}."
            )

    def _validate_physical_channel_parity(
        self,
        errors_reported: list[str],
        legacy_physical_channel: PurrPhysicalChannel,
        physical_channel: PhysicalChannel,
    ):
        """Validates the physical channels have equal properties."""

        if (
            isinstance(physical_channel, ResonatorPhysicalChannel)
            != legacy_physical_channel.acquire_allowed
        ):
            errors_reported.append(
                "Mismatch in the acquire allowed property of the physical channel "
                f"{legacy_physical_channel.full_id()}."
            )

        if not isclose(
            physical_channel.baseband.frequency, legacy_physical_channel.baseband.frequency
        ):
            errors_reported.append(
                "Mismatch in the baseband frequency of the physical channel "
                f"{legacy_physical_channel.full_id()}. "
                f"{physical_channel.baseband.frequency} does not equal "
                f"{legacy_physical_channel.baseband.frequency}."
            )

    def _resolve_pulse_channel(
        self,
        target: PurrPulseChannel | PurrPulseChannelView,
    ) -> str:
        """Matches the pulse channel to the physical model, if possible, and returns the
        id. Otherwise, keeps the channel naming."""

        if not isinstance(target, PurrPulseChannelView):
            return target.partial_id()

        legacy_devices = self.legacy_model.get_devices_from_pulse_channel(target)

        if len(legacy_devices) == 0:
            return target.partial_id()

        if len(legacy_devices) > 1:
            logger.warning(
                f"Multiple devices found for target {target.full_id()}; can't match pulse "
                "channels."
            )
            return target.partial_id()

        legacy_device = legacy_devices[0]
        if isinstance(legacy_device, PurrResonator):
            legacy_qubit = self.legacy_model._map_resonator_to_qubit(legacy_device)
        else:
            legacy_qubit = legacy_device
        qubit = self.model.qubit_with_index(legacy_qubit.index)

        match target.channel_type:
            case PurrChannelType.drive:
                new_id = qubit.drive_pulse_channel.uuid
            case PurrChannelType.second_state:
                new_id = qubit.second_state_pulse_channel.uuid
            case PurrChannelType.freq_shift:
                new_id = qubit.freq_shift_pulse_channel.uuid
            case PurrChannelType.measure:
                new_id = qubit.measure_pulse_channel.uuid
            case PurrChannelType.acquire:
                new_id = qubit.acquire_pulse_channel.uuid
            case PurrChannelType.reset:
                # Reset channels not implemented in pydantic, just pass through
                new_id = None
            case PurrChannelType.cross_resonance:
                other_qubit_index = self._resolve_auxiliary_qubit_index(
                    target.auxiliary_devices
                )
                if other_qubit_index is None:
                    new_id = None
                else:
                    new_id = qubit.cross_resonance_pulse_channels[other_qubit_index].uuid
            case PurrChannelType.cross_resonance_cancellation:
                other_qubit_index = self._resolve_auxiliary_qubit_index(
                    target.auxiliary_devices
                )
                if other_qubit_index is None:
                    new_id = None
                else:
                    new_id = qubit.cross_resonance_cancellation_pulse_channels[
                        other_qubit_index
                    ].uuid
            case _:
                new_id = None

        if new_id is None:
            new_id = target.partial_id()
            logger.warning(
                f"Could not resolve {target.full_id()} to a pulse channel on the "
                "hardware model, using its partial id as the channel id."
            )
        return new_id

    def _resolve_auxiliary_qubit_index(
        self, devices: list[PurrQuantumDevice]
    ) -> int | None:
        if len(devices) != 1:
            logger.warning(f"Expected to see a list of one qubit, found {devices}.")
            return None
        if not isinstance(devices[0], PurrQubit):
            logger.warning(f"Expected a Qubit, got {devices[0]}.")
            return None
        return devices[0].index


class PurrConverter:
    """A converter for PuRR InstructionBuilders.

    The purpose of this converter is to allow us to accept the PuRR InstructionBuilder API
    as an input to the compiler, allowing backwards compatibility. It walks the list of
    instructions, compiling them to QAT IR.

    It takes care of the following:

    * Validation of the physical properties of the hardware model in the IR against the
      model provided to ensure compatibility.
    * Conversion of PuRR pulse channels into QAT IR pulse channels, reconciling with those
      that are created already from the pydantic hardware model. This will also update any
      divergent logical properties, which might appear from device assigns.
    * Conversion of PuRR IR into Pydantic IR.

    .. warning::

        Repeat instructions with a passive reset time or repetition period are not fully
        supported, and these values will be ignored. Please set the passive reset time using
        the compiler config, instead.
    """

    def __init__(self, model: PhysicalHardwareModel):
        self._model = model

    def convert(self, builder: PurrInstructionBuilder) -> QuantumInstructionBuilder:
        """Walks the list of instructions in the PuRR InstructionBuilder, converting them to
        QAT IR.

        :param builder: The quantum program as a legacy (PuRR) instruction builder.
        """
        legacy_model = builder.model
        model_mapper = HardwareModelMapper(legacy_model, self._model)
        model_mapper.validate_physical_properties()

        new_builder: QuantumInstructionBuilder = BuilderFactory.create_builder(self._model)
        for instruction in builder.instructions:
            if isinstance(instruction, PurrInstructions.QuantumInstruction):
                self._register_pulse_channels(instruction, new_builder, model_mapper)
            new_builder.add(
                *self._parse_instruction(
                    instruction,
                    model_mapper=model_mapper,
                )
            )
        return new_builder

    def _register_pulse_channels(
        self,
        instruction: PurrInstructions.QuantumInstruction,
        new_builder: QuantumInstructionBuilder,
        model_mapper: HardwareModelMapper,
    ):
        for target in instruction.quantum_targets:
            if isinstance(target, PurrInstructions.Acquire):
                continue
            if not isinstance(target, PurrPulseChannel):
                raise ValueError(f"Expected target to be a PulseChannel, got {target}.")

            if not model_mapper.pulse_channel_mapping_exists(target):
                new_builder.create_pulse_channel(
                    uuid=model_mapper.get_pulse_channel_id(target),
                    frequency=target.frequency,
                    imbalance=target.imbalance,
                    phase_iq_offset=target.phase_offset,
                    scale=target.scale,
                    physical_channel=model_mapper.physical_channel_map[
                        target.physical_channel_id
                    ],
                )

    @singledispatchmethod
    def _parse_instruction(
        self, instruction: PurrInstructions.Instruction, **kwargs
    ) -> list[Instructions.Instruction]:
        raise NotImplementedError(
            f"Converting of instruction type {type(instruction)} is not supported."
        )

    @_parse_instruction.register(PurrInstructions.Sweep)
    @_parse_instruction.register(PurrInstructions.EndSweep)
    @_parse_instruction.register(PurrInstructions.DeviceUpdate)
    def _(
        self,
        instruction: PurrInstructions.Sweep
        | PurrInstructions.EndSweep
        | PurrInstructions.DeviceUpdate,
        **kwargs,
    ):
        raise NotImplementedError(
            "Converting of builders with sweeps or device updates is not supported. "
            "Builders with sweeps can be unrolled using SweepPipelines."
        )

    @_parse_instruction.register(PurrInstructions.Repeat)
    def _(
        self, instruction: PurrInstructions.Repeat, **kwargs
    ) -> list[Instructions.Repeat]:
        if (
            instruction.repetition_period is not None
            or instruction.passive_reset_time is not None
        ):
            warn(
                "Converting of Repeat instructions with repetition periods or passive reset "
                "times is not supported, and the values will be ignored. Please set the "
                "passive reset time using the compiler config, instead."
            )
        return [Instructions.Repeat(repeat_count=instruction.repeat_count)]

    @_parse_instruction.register(PurrInstructions.EndRepeat)
    def _(self, instruction: PurrInstructions.EndRepeat, **kwargs):
        raise NotImplementedError("Manual use of EndRepeat is not yet supported.")

    @_parse_instruction.register(PurrInstructions.Assign)
    def _(
        self, instruction: PurrInstructions.Assign, **kwargs
    ) -> list[Instructions.Assign]:
        return [
            Instructions.Assign(
                name=instruction.name, value=self._recursively_strip(instruction.value)
            )
        ]

    def _recursively_strip(self, value):
        match value:
            case list():
                value = [self._recursively_strip(val) for val in value]
            case PurrInstructions.IndexAccessor():
                value = (value.name, value.index)
            case PurrInstructions.Variable():
                value = value.name
            case PurrInstructions.BinaryOperator():
                value = str(value).replace("variable ", "")
        return value

    @_parse_instruction.register(PurrInstructions.PostProcessing)
    def _(
        self, instruction: PurrInstructions.PostProcessing, **kwargs
    ) -> list[MeasureInstructions.PostProcessing]:
        return [
            MeasureInstructions.PostProcessing(
                output_variable=instruction.output_variable,
                process_type=instruction.process.value,
                axes=instruction.axes,
                args=instruction.args,
                result_needed=instruction.result_needed,
            )
        ]

    @_parse_instruction.register(PurrInstructions.ResultsProcessing)
    def _(
        self, instruction: PurrInstructions.ResultsProcessing, **kwargs
    ) -> list[Instructions.ResultsProcessing]:
        return [
            Instructions.ResultsProcessing(
                variable=instruction.variable,
                results_processing=instruction.results_processing,
            )
        ]

    @_parse_instruction.register(PurrInstructions.Return)
    def _(
        self, instruction: PurrInstructions.Return, **kwargs
    ) -> list[Instructions.Return]:
        return [Instructions.Return(variables=instruction.variables)]

    @_parse_instruction.register(PurrInstructions.Variable)
    def _(
        self, instruction: PurrInstructions.Variable, **kwargs
    ) -> list[Instructions.Variable]:
        return [
            Instructions.Variable(
                name=instruction.name,
                var_type=instruction.var_type,
                value=instruction.value,
            )
        ]

    @_parse_instruction.register(PurrInstructions.Reset)
    def _(
        self,
        instruction: PurrInstructions.Reset,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.Reset]:
        instructions = []
        for target in instruction.quantum_targets:
            qubits: set[int] = set()
            # api on model mapper
            for device in model_mapper.legacy_model.get_devices_from_pulse_channel(target):
                if isinstance(device, PurrQubit):
                    qubits.add(device.index)
                else:
                    logger.warning(
                        f"Converting of Reset instruction with target {target.full_id()} "
                        f"found a device {device.full_id()} which is not a qubit, this "
                        "device will be ignored."
                    )

            for qubit in qubits:
                instructions.append(Instructions.Reset(qubit_target=qubit))
        return instructions

    @_parse_instruction.register(PurrInstructions.FrequencySet)
    def _(
        self,
        instruction: PurrInstructions.FrequencySet,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.FrequencySet]:
        return [
            Instructions.FrequencySet(
                target=model_mapper.get_pulse_channel_id(target),
                frequency=instruction.frequency,
            )
            for target in instruction.quantum_targets
        ]

    @_parse_instruction.register(PurrInstructions.FrequencyShift)
    def _(
        self,
        instruction: PurrInstructions.FrequencyShift,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.FrequencyShift]:
        return [
            Instructions.FrequencyShift(
                target=model_mapper.get_pulse_channel_id(target),
                frequency=instruction.frequency,
            )
            for target in instruction.quantum_targets
        ]

    @_parse_instruction.register(PurrInstructions.PhaseSet)
    def _(
        self,
        instruction: PurrInstructions.PhaseSet,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.PhaseSet]:
        return [
            Instructions.PhaseSet(
                target=model_mapper.get_pulse_channel_id(target), phase=instruction.phase
            )
            for target in instruction.quantum_targets
        ]

    @_parse_instruction.register(PurrInstructions.PhaseReset)
    def _(
        self,
        instruction: PurrInstructions.PhaseReset,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.PhaseSet]:
        return [
            Instructions.PhaseSet(
                target=model_mapper.get_pulse_channel_id(target), phase=0.0
            )
            for target in instruction.quantum_targets
        ]

    @_parse_instruction.register(PurrInstructions.PhaseShift)
    def _(
        self,
        instruction: PurrInstructions.PhaseShift,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.PhaseShift]:
        return [
            Instructions.PhaseShift(
                target=model_mapper.get_pulse_channel_id(target), phase=instruction.phase
            )
            for target in instruction.quantum_targets
        ]

    @_parse_instruction.register(PurrInstructions.Delay)
    def _(
        self,
        instruction: PurrInstructions.Delay,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.Delay]:
        return [
            Instructions.Delay(
                target=model_mapper.get_pulse_channel_id(target),
                duration=instruction.duration,
            )
            for target in instruction.quantum_targets
        ]

    @_parse_instruction.register(PurrInstructions.Synchronize)
    def _(
        self,
        instruction: PurrInstructions.Synchronize,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[Instructions.Synchronize]:
        targets = set(
            model_mapper.get_pulse_channel_id(target)
            for target in instruction.quantum_targets
        )
        return [Instructions.Synchronize(targets=targets)]

    @_parse_instruction.register(PurrInstructions.Pulse)
    def _(
        self,
        instruction: PurrInstructions.Pulse,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[WaveformInstructions.Pulse]:
        # Pulses are now represented by waveforms with the waveform attributes, and pulse
        # for where that waveform is played.

        waveform_data = dict()
        pulse_data = dict()
        # Pulse contains many attributes, so we iterate using vars...
        for name, var in vars(instruction).items():
            if name == "quantum_targets":
                pulse_data["targets"] = set(
                    model_mapper.get_pulse_channel_id(target) for target in var
                )
            elif name in ("ignore_channel_scale", "duration"):
                pulse_data[name] = var
            else:
                waveform_data[name] = var

        waveform_class = waveform_data.pop("shape")
        if waveform_class not in waveform_map:
            raise ValueError(
                f"Unsupported waveform shape {waveform_class} found when converting "
                "Pulse instruction."
            )

        return [
            WaveformInstructions.Pulse(
                waveform=waveform_map[waveform_class](**waveform_data),
                **pulse_data,
            )
        ]

    @_parse_instruction.register(PurrInstructions.CustomPulse)
    def _(
        self,
        instruction: PurrInstructions.CustomPulse,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[WaveformInstructions.Pulse]:
        waveform = WaveformInstructions.SampledWaveform(samples=instruction.samples)
        return [
            WaveformInstructions.Pulse(
                waveform=waveform,
                targets=set(
                    model_mapper.get_pulse_channel_id(target)
                    for target in instruction.quantum_targets
                ),
                duration=instruction.duration,
            )
        ]

    @_parse_instruction.register(PurrInstructions.Acquire)
    def _(
        self,
        instruction: PurrInstructions.Acquire,
        model_mapper: HardwareModelMapper,
        **kwargs,
    ) -> list[MeasureInstructions.Acquire | Instructions.Delay]:
        instructions = []

        if len(instruction.quantum_targets) != 1:
            raise ValueError(
                f"Expected Acquire instruction to have exactly one quantum target, found "
                f"{len(instruction.quantum_targets)}."
            )
        target = model_mapper.get_pulse_channel_id(next(iter(instruction.quantum_targets)))

        if instruction.delay is not None and instruction.delay > 0:
            instructions.append(
                Instructions.Delay(target=target, duration=instruction.delay)
            )

        filter = (
            self._parse_instruction(instruction.filter, model_mapper=model_mapper)[0]
            if instruction.filter is not None
            else None
        )
        instructions.append(
            MeasureInstructions.Acquire(
                target=target,
                duration=instruction.time,
                mode=instruction.mode,
                output_variable=instruction.output_variable,
                rotation=instruction.rotation,
                threshold=instruction.threshold,
                filter=filter,
            )
        )
        return instructions
