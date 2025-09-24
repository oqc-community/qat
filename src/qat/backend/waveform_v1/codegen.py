# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import List, Optional

import numpy as np

from qat.backend.base import BaseBackend
from qat.backend.passes.analysis import (
    PydIntermediateFrequencyAnalysis,
    PydIntermediateFrequencyResult,
    PydTimelineAnalysis,
    PydTimelineAnalysisResult,
)
from qat.backend.passes.lowering import PydPartitionByPulseChannel
from qat.backend.passes.purr.analysis import (
    IntermediateFrequencyAnalysis,
    IntermediateFrequencyResult,
    TimelineAnalysis,
    TimelineAnalysisResult,
)
from qat.backend.passes.purr.lowering import PartitionByPulseChannel
from qat.backend.passes.purr.validation import NoAcquireWeightsValidation
from qat.backend.passes.validation import PydNoAcquireWeightsValidation
from qat.backend.waveform_v1.executable import (
    WaveformV1ChannelData,
    WaveformV1Executable,
)
from qat.core.pass_base import InvokerMixin, MetricsManager, PassManager
from qat.core.result_base import ResultManager
from qat.executables import AcquireData
from qat.ir.instructions import Assign
from qat.ir.instructions import FrequencyShift as PydFrequencyShift
from qat.ir.instructions import PhaseReset as PydPhaseReset
from qat.ir.instructions import PhaseSet as PydPhaseSet
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.instructions import Reset as PydReset
from qat.ir.lowered import PartitionedIR
from qat.ir.measure import PostProcessing
from qat.ir.waveforms import Pulse as PydPulse
from qat.ir.waveforms import SampledWaveform
from qat.model.device import PhysicalChannel, Qubit
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import DeviceDescription, TargetData
from qat.purr.backends.utilities import UPCONVERT_SIGN, evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    CustomPulse,
    FrequencySet,
    FrequencyShift,
    Instruction,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    Pulse,
    Reset,
    Waveform,
)
from qat.purr.utils.logger import get_default_logger
from qat.utils.waveform import NumericFunction

log = get_default_logger()


class WaveformV1Backend(BaseBackend, InvokerMixin):
    """
    Target-machine code generation from an IR for targets that only require the explicit waveforms.
    """

    # TODO: replacing buffers calculations as passes: waveform generation pass, pulse
    # channel buffers pass, buffer amalgamation pass (COMPILER-413)

    def __init__(self, model: QuantumHardwareModel):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
                    As the emitter is used to generate code for some target machine, the hardware
                    model is needed for context-aware compilation.
        """
        self.model = model

    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        upconvert: bool = True,
        **kwargs,
    ) -> WaveformV1Executable:
        """Compiles :class:`InstructionBuilder` into a :class:`WaveformV1Executable`.

        Translates pulse instructions into explicit waveforms at the required times, and
        combines them across pulse channels to give a composite waveform on the necessary
        physical channels. Deals with low-level details such as pulse scheduling and phase
        shifts.
        """
        res_mgr = res_mgr if res_mgr is not None else ResultManager()
        met_mgr = met_mgr if met_mgr is not None else MetricsManager()
        ir = self.run_pass_pipeline(ir, res_mgr, met_mgr)
        timeline_res: TimelineAnalysisResult = res_mgr.lookup_by_type(
            TimelineAnalysisResult
        )
        if_res = res_mgr.lookup_by_type(IntermediateFrequencyResult)

        buffers = self.create_physical_channel_buffers(ir, timeline_res, upconvert)
        acquire_dict = self.create_acquire_dict(ir, timeline_res)

        channels: dict[str, WaveformV1ChannelData] = {}
        for physical_channel, buffer in buffers.items():
            channels[physical_channel.full_id()] = WaveformV1ChannelData(
                baseband_frequency=if_res.frequencies.get(physical_channel, None),
                buffer=buffer,
                acquires=acquire_dict.get(physical_channel, []),
            )

        returns = []
        for _return in ir.returns:
            returns.extend(_return.variables)

        if ir.passive_reset_time is not None:
            repetition_time = timeline_res.total_duration + ir.passive_reset_time
        else:
            repetition_time = ir.repetition_period

        return WaveformV1Executable(
            channel_data=channels,
            shots=ir.shots,
            compiled_shots=ir.compiled_shots,
            repetition_time=repetition_time,
            post_processing={
                var: [PostProcessing._from_legacy(pp) for pp in pp_list]
                for var, pp_list in ir.pp_map.items()
            },
            results_processing={
                var: rp.results_processing for var, rp in ir.rp_map.items()
            },
            assigns=[Assign._from_legacy(assign) for assign in ir.assigns],
            returns=returns,
            calibration_id=self.model.calibration_id,
        )

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | NoAcquireWeightsValidation()
            | PartitionByPulseChannel()
            | TimelineAnalysis()
            | IntermediateFrequencyAnalysis(self.model)
        )

    @staticmethod
    def create_pulse_channel_buffer(
        pulse_channel: PulseChannel,
        instructions: list[Instruction],
        durations: list[int],
        upconvert: bool = True,
    ):
        """Creates a buffer of waveforms for a single pulse channel.

        :param pulse_channel: The pulse channel to create the buffer for.
        :param instructions: The list of instructions to process.
        :param durations: The list of durations for each instruction.
        :param upconvert: Whether to upconvert the waveforms to the target frequencies.
        """
        context = WaveformContext(pulse_channel, np.sum(durations))
        for i, instruction in enumerate(instructions):
            duration = durations[i]
            if isinstance(instruction, Waveform):
                context.process_pulse(instruction, duration, upconvert)
            elif isinstance(instruction, PhaseShift):
                context.process_phaseshift(instruction.phase)
            elif isinstance(instruction, PhaseSet):
                context.process_phaseset(instruction.phase)
            elif isinstance(instruction, PhaseReset):
                context.process_phasereset()
            elif isinstance(instruction, Reset):
                context.process_reset()
            elif isinstance(instruction, FrequencySet):
                context.process_frequencyset(instruction.frequency)
            elif isinstance(instruction, FrequencyShift):
                context.process_frequencyshift(instruction.frequency)
            else:
                context.process_delay(duration)
        return context.buffer

    def create_physical_channel_buffers(
        self,
        ir: PartitionedIR,
        timeline_res: TimelineAnalysisResult,
        upconvert: bool = True,
    ):
        """Creates a buffer of waveforms for each physical channel.

        :param ir: The partitioned IR containing the instructions.
        :param timeline_res: The timeline analysis result containing the number of samples
            for each instruction.
        :param upconvert: Whether to upconvert the waveforms to the target frequencies.
        """
        # Compute all pulse channel buffers
        pulse_channel_buffers = {}
        for pulse_channel in timeline_res.target_map:
            # The buffer is trivial if there are no waveforms
            if not any(
                [isinstance(inst, Waveform) for inst in ir.target_map[pulse_channel]]
            ):
                continue

            pulse_channel_buffers[pulse_channel] = self.create_pulse_channel_buffer(
                pulse_channel,
                ir.target_map[pulse_channel],
                timeline_res.target_map[pulse_channel].samples,
                upconvert,
            )

        # Organize by physical channels
        phys_to_pulse_map = {}
        for pulse_channel, buffer in pulse_channel_buffers.items():
            phys_to_pulse_map.setdefault(pulse_channel.physical_channel, []).append(buffer)

        # Compute sum of pulse channels
        physical_channel_buffers = {}
        for physical_channel in self.model.physical_channels.values():
            buffer_list = phys_to_pulse_map.get(physical_channel, [])
            if len(buffer_list) == 0:
                length = 0
            else:
                length = max(0, *[len(buffer) for buffer in buffer_list])
            buffer_sum = np.zeros(length, dtype=np.complex128)
            for buffer in buffer_list:
                buffer_sum[0 : len(buffer)] += buffer
            physical_channel_buffers[physical_channel] = buffer_sum

        return physical_channel_buffers

    def create_acquire_dict(self, ir: PartitionedIR, timeline_res: TimelineAnalysisResult):
        """Creates a dictionary of acquire data for each physical channel based on the
        acquire map in the IR and the timeline analysis result.

        :param ir: The partitioned IR containing the acquire map.
        :param timeline_res: The timeline analysis result containing the number of samples
            for each instruction.
        """
        acquire_dict = {}
        for pulse_channel, acquire_list in ir.acquire_map.items():
            acq_list = acquire_dict.setdefault(pulse_channel.physical_channel, [])
            for acquire in acquire_list:
                idx = ir.target_map[pulse_channel].index(acquire)
                acq_list.append(
                    AcquireData(
                        length=timeline_res.target_map[pulse_channel].samples[idx],
                        position=np.sum(
                            timeline_res.target_map[pulse_channel].samples[0:idx]
                        ),
                        mode=acquire.mode,
                        output_variable=acquire.output_variable,
                    )
                )
        return acquire_dict


class WaveformContext:
    """Used for waveform code generation for a particular pulse channel.

    This can be considered to be the dynamical state of a pulse channel which evolves as the
    circuit progresses.
    """

    # TODO: refactor this as a "channel backend" or as passes depending on when this is
    # done (COMPILER-413).

    def __init__(self, pulse_channel: PulseChannel, total_duration: int):
        """
        :param pulse_channel: The pulse channel that this is modelling.
        :param total_duration: The lifetime of the pulse channel in number of samples.
        """
        self.pulse_channel = pulse_channel
        self._buffer = np.zeros(total_duration, dtype=np.complex128)
        self._duration = 0
        self._phase = 0.0
        self._frequency = pulse_channel.frequency

    @property
    def physical_channel(self):
        return self.pulse_channel.physical_channel

    @property
    def buffer(self):
        return self._buffer

    def process_pulse(
        self,
        instruction: PydPulse,
        samples: int,
        sample_time: float,
        do_upconvert: bool = True,
    ):
        """Converts a waveform instruction into a discrete number of samples, handling
        upconversion to the target frequency if specified."""

        # TODO: the evaluate shape is handled in the EvaluatePulses pass, so this needs
        # adjusting to only accept square waveforms and custom pulses. (COMPILER-413)

        length = samples * sample_time
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * sample_time,
            stop=length - centre - 0.5 * sample_time,
            num=samples,
        )
        pulse = evaluate_shape(instruction, t, self._phase)

        scale = self.pulse_channel.scale
        if (
            isinstance(instruction, (Pulse, CustomPulse))
            and instruction.ignore_channel_scale
        ):
            scale = 1
        pulse *= scale
        pulse += self.pulse_channel.bias

        if do_upconvert:
            t += centre - 0.5 * sample_time + self._duration * sample_time
            pulse = self._do_upconvert(pulse, t)

        self._buffer[self._duration : self._duration + samples] = pulse
        self._duration += samples

    def _do_upconvert(
        self,
        buffer: List[float],
        time: List[float],
    ):
        """A virtually NCO to upconvert the waveforms by a frequency that is the difference
        between the target frequency and the baseband frequency."""

        tslip = self.pulse_channel.phase_offset
        imbalance = self.pulse_channel.imbalance
        if self.pulse_channel.fixed_if:
            freq = self.pulse_channel.baseband_if_frequency
        else:
            freq = self._frequency - self.pulse_channel.baseband_frequency
        buffer *= np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * time)
        if not tslip == 0.0:
            buffer_slip = buffer * np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * tslip)
            buffer.imag = buffer_slip.imag
        if not imbalance == 1.0:
            buffer.real /= imbalance**0.5
            buffer.imag *= imbalance**0.5

        return buffer

    def process_phaseshift(self, phase: float):
        self._phase += phase

    def process_phaseset(self, phase: float):
        self._phase = phase

    def process_phasereset(self):
        self._phase = 0

    def process_reset(self):
        log.warning(
            "The WaveformV1Backend uses a `repetition_time` for resetting, so the reset "
            "instruction will be ignored."
        )

    def process_frequencyset(self, frequency: float):
        self._frequency = frequency

    def process_frequencyshift(self, frequency: float):
        self._frequency += frequency

    def process_delay(self, samples: int):
        self._duration += samples


class PydWaveformV1Backend(BaseBackend, InvokerMixin):
    """
    Target-machine code generation from an IR for targets that only require the explicit waveforms.
    """

    def __init__(
        self,
        model: PhysicalHardwareModel,
        target_data: TargetData = TargetData.default(),
    ):
        self.model = model
        self.target_data = target_data

    def emit(
        self,
        ir: InstructionBuilder,
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        upconvert: bool = True,
        **kwargs,
    ) -> WaveformV1Executable:
        """Compiles :class:`InstructionBuilder` into a :class:`WaveformV1Executable`.

        Translates pulse instructions into explicit waveforms at the required times, and
        combines them across pulse channels to give a composite waveform on the necessary
        physical channels. Deals with low-level details such as pulse scheduling and phase
        shifts.
        """
        res_mgr = res_mgr if res_mgr is not None else ResultManager()
        met_mgr = met_mgr if met_mgr is not None else MetricsManager()
        ir = self.run_pass_pipeline(ir, res_mgr, met_mgr)
        timeline_res: PydTimelineAnalysisResult = res_mgr.lookup_by_type(
            PydTimelineAnalysisResult
        )
        if_res = res_mgr.lookup_by_type(PydIntermediateFrequencyResult)

        buffers = self.create_physical_channel_buffers(ir, timeline_res, upconvert)
        acquire_dict = self.create_acquire_dict(ir, timeline_res)

        channels: dict[str, WaveformV1ChannelData] = {}
        for physical_channel, buffer in buffers.items():
            channels[physical_channel.uuid] = WaveformV1ChannelData(
                baseband_frequency=if_res.frequencies.get(physical_channel, None),
                buffer=buffer,
                acquires=acquire_dict.get(physical_channel, []),
            )

        returns = []
        for _return in ir.returns:
            returns.extend(_return.variables)

        if ir.passive_reset_time is not None:
            repetition_time = timeline_res.total_duration + ir.passive_reset_time
        else:
            repetition_time = ir.repetition_period

        return WaveformV1Executable(
            channel_data=channels,
            shots=ir.shots,
            compiled_shots=ir.compiled_shots,
            repetition_time=repetition_time,
            post_processing=ir.pp_map,
            results_processing={
                var: rp.results_processing for var, rp in ir.rp_map.items()
            },
            assigns=ir.assigns,
            returns=returns,
            calibration_id=self.model.calibration_id,
        )

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | PydNoAcquireWeightsValidation()
            | PydPartitionByPulseChannel()
            | PydTimelineAnalysis(self.model, self.target_data)
            | PydIntermediateFrequencyAnalysis(self.model)
        )

    @staticmethod
    def create_pulse_channel_buffer(
        pulse_channel: PulseChannel,
        instructions: list[Instruction],
        durations: list[int],
        device_description: DeviceDescription,
        phys_channel: PhysicalChannel,
        upconvert: bool = True,
    ):
        """Creates a buffer of waveforms for a single pulse channel.

        :param pulse_channel: The pulse channel to create the buffer for.
        :param instructions: The list of instructions to process.
        :param durations: The list of durations for each instruction.
        :param upconvert: Whether to upconvert the waveforms to the target frequencies.
        """
        context = PydWaveformContext(
            pulse_channel, np.sum(durations), device_description, phys_channel
        )
        for i, instruction in enumerate(instructions):
            duration = durations[i]
            if isinstance(instruction, PydPulse):
                context.process_pulse(instruction, duration, upconvert)
            elif isinstance(instruction, PydPhaseShift):
                context.process_phaseshift(instruction.phase)
            elif isinstance(instruction, PydPhaseSet):
                context.process_phaseset(instruction.phase)
            elif isinstance(instruction, PydPhaseReset):
                context.process_phasereset()
            elif isinstance(instruction, PydReset):
                context.process_reset()
            elif isinstance(instruction, PydFrequencyShift):
                context.process_frequencyshift(instruction.frequency)
            else:
                context.process_delay(duration)
        return context.buffer

    def create_physical_channel_buffers(
        self,
        ir: PartitionedIR,
        timeline_res: TimelineAnalysisResult,
        upconvert: bool = True,
    ):
        """Creates a buffer of waveforms for each physical channel.

        :param ir: The partitioned IR containing the instructions.
        :param timeline_res: The timeline analysis result containing the number of samples
            for each instruction.
        :param upconvert: Whether to upconvert the waveforms to the target frequencies.
        """
        # Compute all pulse channel buffers
        pulse_channel_buffers = {}
        for pulse_channel_id in timeline_res.target_map:
            # The buffer is trivial if there are no waveforms
            if not any(
                [isinstance(inst, PydPulse) for inst in ir.target_map[pulse_channel_id]]
            ):
                continue

            pulse_channel = self.model.pulse_channel_with_id(pulse_channel_id)
            device = self.model.device_for_pulse_channel_id(pulse_channel_id)
            pulse_channel_buffers[pulse_channel_id] = self.create_pulse_channel_buffer(
                pulse_channel,
                ir.target_map[pulse_channel_id],
                timeline_res.target_map[pulse_channel_id].samples,
                self.target_data.QUBIT_DATA
                if isinstance(device, Qubit)
                else self.target_data.RESONATOR_DATA,
                self.model.physical_channel_for_pulse_channel_id(pulse_channel.uuid),
                upconvert,
            )

        # Organize by physical channels
        phys_to_pulse_map = {}
        for pulse_channel_id, buffer in pulse_channel_buffers.items():
            phys_to_pulse_map.setdefault(
                self.model.physical_channel_for_pulse_channel_id(pulse_channel_id), []
            ).append(buffer)

        # Compute sum of pulse channels
        physical_channel_buffers = {}
        for device in self.model.quantum_devices:
            physical_channel = device.physical_channel
            buffer_list = phys_to_pulse_map.get(physical_channel, [])
            if len(buffer_list) == 0:
                length = 0
            else:
                length = max(0, *[len(buffer) for buffer in buffer_list])
            buffer_sum = np.zeros(length, dtype=np.complex128)
            for buffer in buffer_list:
                buffer_sum[0 : len(buffer)] += buffer
            physical_channel_buffers[physical_channel] = buffer_sum

        return physical_channel_buffers

    def create_acquire_dict(self, ir: PartitionedIR, timeline_res: TimelineAnalysisResult):
        """Creates a dictionary of acquire data for each physical channel based on the
        acquire map in the IR and the timeline analysis result.

        :param ir: The partitioned IR containing the acquire map.
        :param timeline_res: The timeline analysis result containing the number of samples
            for each instruction.
        """
        acquire_dict = {}
        for pulse_channel_id, acquire_list in ir.acquire_map.items():
            phys_channel = self.model.physical_channel_for_pulse_channel_id(
                pulse_channel_id
            )
            acq_list = acquire_dict.setdefault(phys_channel, [])
            for acquire in acquire_list:
                idx = ir.target_map[pulse_channel_id].index(acquire)
                acq_list.append(
                    AcquireData(
                        length=timeline_res.target_map[pulse_channel_id].samples[idx],
                        position=np.sum(
                            timeline_res.target_map[pulse_channel_id].samples[0:idx]
                        ),
                        mode=acquire.mode,
                        output_variable=acquire.output_variable,
                    )
                )
        return acquire_dict


class PydWaveformContext:
    """Used for waveform code generation for a particular pulse channel.

    This can be considered to be the dynamical state of a pulse channel which evolves as the
    circuit progresses.
    """

    # TODO: refactor this as a "channel backend" or as passes depending on when this is
    # done (COMPILER-413).

    def __init__(
        self,
        pulse_channel: PulseChannel,
        total_duration: int,
        device_description: DeviceDescription,
        phys_channel: PhysicalChannel,
    ):
        """
        :param pulse_channel: The pulse channel that this is modelling.
        :param total_duration: The lifetime of the pulse channel in number of samples.
        """
        self.pulse_channel = pulse_channel
        self.physical_channel = phys_channel
        self._device_desc = device_description
        self._buffer = np.zeros(total_duration, dtype=np.complex128)
        self._duration = 0
        self._phase = 0.0
        self._frequency = pulse_channel.frequency

    @property
    def buffer(self):
        return self._buffer

    def process_pulse(
        self,
        instruction: PydPulse,
        samples: int,
        do_upconvert: bool = True,
    ):
        """Converts a waveform instruction into a discrete number of samples, handling
        upconversion to the target frequency if specified."""

        # TODO: the evaluate shape is handled in the EvaluatePulses pass, so this needs
        # adjusting to only accept square waveforms and custom pulses. (COMPILER-413)

        dt = self._device_desc.sample_time
        length = samples * dt
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * dt, stop=length - centre - 0.5 * dt, num=samples
        )
        waveform = instruction.waveform

        if isinstance(waveform, SampledWaveform):
            amp = 1.0
            scale_factor = 1.0
            drag = 0.0
            amplitude = np.array(waveform.samples, dtype=np.csingle)

            pulse = scale_factor * amp * np.exp(1.0j * self._phase) * amplitude
            if not drag == 0.0:
                amplitude_differential = NumericFunction().derivative(t, amplitude)
                if len(amplitude_differential) < len(pulse):
                    amplitude_differential = np.pad(
                        amplitude_differential,
                        (0, len(pulse) - len(amplitude_differential)),
                        "edge",
                    )
                pulse += (
                    drag
                    * 1.0j
                    * amp
                    * scale_factor
                    * np.exp(1.0j * self._phase)
                    * amplitude_differential
                )
        else:
            pulse = waveform.sample(t, phase_offset=self._phase).samples

        # pulse = evaluate_shape(instruction.waveform, t, self._phase)

        scale = self.pulse_channel.scale
        if instruction.ignore_channel_scale:
            scale = 1
        pulse *= scale
        pulse += self.physical_channel.iq_voltage_bias.bias

        if do_upconvert:
            t += centre - 0.5 * dt + self._duration * dt
            pulse = self._do_upconvert(pulse, t)

        self._buffer[self._duration : self._duration + samples] = pulse
        self._duration += samples

    def _do_upconvert(
        self,
        buffer: List[float],
        time: List[float],
    ):
        """A virtually NCO to upconvert the waveforms by a frequency that is the difference
        between the target frequency and the baseband frequency."""

        tslip = self.pulse_channel.phase_iq_offset
        imbalance = self.pulse_channel.imbalance
        if self.pulse_channel.fixed_if:
            freq = self.physical_channel.baseband.if_frequency
        else:
            freq = self._frequency - self.physical_channel.baseband.frequency
        buffer *= np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * time)
        if not tslip == 0.0:
            buffer_slip = buffer * np.exp(UPCONVERT_SIGN * 2.0j * np.pi * freq * tslip)
            buffer.imag = buffer_slip.imag
        if not imbalance == 1.0:
            buffer.real /= imbalance**0.5
            buffer.imag *= imbalance**0.5

        return buffer

    def process_phaseshift(self, phase: float):
        self._phase += phase

    def process_phaseset(self, phase: float):
        self._phase = phase

    def process_phasereset(self):
        self._phase = 0

    def process_reset(self):
        log.warning(
            "The PydWaveformV1Backend uses a `repetition_time` for resetting, so the reset "
            "instruction will be ignored."
        )

    def process_frequencyset(self, frequency: float):
        self._frequency = frequency

    def process_frequencyshift(self, frequency: float):
        self._frequency += frequency

    def process_delay(self, samples: int):
        self._duration += samples
