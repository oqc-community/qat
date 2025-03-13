# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from typing import List, Optional

import numpy as np

from qat.backend.base import BaseBackend
from qat.backend.passes.analysis import (
    IntermediateFrequencyAnalysis,
    IntermediateFrequencyResult,
    TimelineAnalysis,
    TimelineAnalysisResult,
    TriagePass,
    TriageResult,
)
from qat.backend.passes.transform import RepeatSanitisation, ReturnSanitisation
from qat.backend.passes.validation import FrequencyValidation, NoAcquireWeightsValidation
from qat.backend.waveform_v1.executable import WaveformV1ChannelData, WaveformV1Executable
from qat.core.pass_base import InvokerMixin, MetricsManager, PassManager
from qat.core.result_base import ResultManager
from qat.ir.instructions import Assign
from qat.ir.measure import PostProcessing
from qat.middleend.passes.transform import PostProcessingSanitisation
from qat.purr.backends.utilities import UPCONVERT_SIGN, evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    CustomPulse,
    FrequencyShift,
    Instruction,
    PhaseReset,
    PhaseShift,
    Pulse,
    Repeat,
    Reset,
    Waveform,
)
from qat.runtime.executables import AcquireDataStruct


class WaveformV1Backend(BaseBackend, InvokerMixin):
    """
    Target-machine code generation from an IR for targets that only require the explicit waveforms.
    """

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
        res_mgr = res_mgr if res_mgr else ResultManager()
        met_mgr = met_mgr if met_mgr else MetricsManager()
        self.run_pass_pipeline(ir, res_mgr, met_mgr)
        triage_res: TriageResult = res_mgr.lookup_by_type(TriageResult)
        timeline_res: TimelineAnalysisResult = res_mgr.lookup_by_type(
            TimelineAnalysisResult
        )
        if_res = res_mgr.lookup_by_type(IntermediateFrequencyResult)

        buffers = self.create_physical_channel_buffers(triage_res, timeline_res, upconvert)
        acquire_dict = self.create_acquire_dict(triage_res, timeline_res)

        channels: dict[str, WaveformV1ChannelData] = {}
        for physical_channel, buffer in buffers.items():
            channels[physical_channel.full_id()] = WaveformV1ChannelData(
                baseband_frequency=if_res.frequencies.get(physical_channel, None),
                buffer=buffer,
                acquires=acquire_dict.get(physical_channel, []),
            )

        # TODO: adjust the existing validation/transformation Repeat passes, or add an
        # analysis pass to determine the repeats?
        repeat = [inst for inst in ir.instructions if isinstance(inst, Repeat)][0]
        returns = []
        for _return in triage_res.returns:
            returns.extend(_return.variables)

        return WaveformV1Executable(
            channel_data=channels,
            shots=repeat.repeat_count,
            repetition_time=repeat.repetition_period,
            post_processing={
                var: [PostProcessing._from_legacy(pp) for pp in pp_list]
                for var, pp_list in triage_res.pp_map.items()
            },
            results_processing={
                var: rp.results_processing for var, rp in triage_res.rp_map.items()
            },
            assigns=[Assign._from_legacy(assign) for assign in triage_res.assigns],
            returns=returns,
        )

    def build_pass_pipeline(self, *args, **kwargs):
        return (
            PassManager()
            | RepeatSanitisation(self.model)
            | ReturnSanitisation()
            | PostProcessingSanitisation()
            | FrequencyValidation(self.model)
            | NoAcquireWeightsValidation()
            | TriagePass()
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
        context = WaveformContext(pulse_channel, np.sum(durations))
        for i, instruction in enumerate(instructions):
            duration = durations[i]
            if isinstance(instruction, Waveform):
                context.process_pulse(instruction, duration, upconvert)
            elif isinstance(instruction, PhaseShift):
                context.process_phaseshift(instruction.phase)
            elif isinstance(instruction, PhaseReset):
                context.process_phasereset()
            elif isinstance(instruction, Reset):
                context.process_reset()
            elif isinstance(instruction, FrequencyShift):
                context.process_frequencyshift(instruction.frequency)
            else:
                context.process_delay(duration)
        return context.buffer

    def create_physical_channel_buffers(
        self,
        triage_res: TriageResult,
        timeline_res: TimelineAnalysisResult,
        upconvert: bool = True,
    ):
        # Compute all pulse channel buffers
        pulse_channel_buffers = {}
        for pulse_channel in timeline_res.durations:
            # The buffer is trivial if there are no waveforms
            if not any(
                [
                    isinstance(inst, Waveform)
                    for inst in triage_res.target_map[pulse_channel]
                ]
            ):
                continue

            pulse_channel_buffers[pulse_channel] = self.create_pulse_channel_buffer(
                pulse_channel,
                triage_res.target_map[pulse_channel],
                timeline_res.durations[pulse_channel],
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

    def create_acquire_dict(
        self, triage_res: TriageResult, timeline_res: TimelineAnalysisResult
    ):
        acquire_dict = {}
        for pulse_channel, acquire_list in triage_res.acquire_map.items():
            acq_list = acquire_dict.setdefault(pulse_channel.physical_channel, [])
            for acquire in acquire_list:
                idx = triage_res.target_map[pulse_channel].index(acquire)
                acq_list.append(
                    AcquireDataStruct(
                        length=timeline_res.durations[pulse_channel][idx],
                        position=np.sum(timeline_res.durations[pulse_channel][0:idx]),
                        mode=acquire.mode,
                        output_variable=acquire.output_variable,
                    )
                )
        return acquire_dict


class WaveformContext:

    def __init__(self, pulse_channel: PulseChannel, total_duration: int):
        """Used for waveform code generation for a particular pulse channel.

        This can be considered to be the dynamical state of a pulse channel which evolves as
        the circuit progresses.

        :param PulseChannel pulse_channel: The pulse channel that this is modelling.
        :param int total_duration: The lifetime of the pulse channel in number of samples.
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
        instruction: Waveform,
        samples: int,
        do_upconvert: bool = True,
    ):
        dt = self.pulse_channel.sample_time
        length = samples * dt
        centre = length / 2.0
        t = np.linspace(
            start=-centre + 0.5 * dt, stop=length - centre - 0.5 * dt, num=samples
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
            t += centre - 0.5 * dt + self._duration * dt
            pulse = self._do_upconvert(pulse, t)

        self._buffer[self._duration : self._duration + samples] = pulse
        self._duration += samples

    def _do_upconvert(
        self,
        buffer: List[float],
        time: List[float],
    ):
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

    def process_phasereset(self):
        self._phase = 0

    def process_reset(self):
        raise NotImplementedError(
            "The Reset instruction is not implemented for WaveformV1Executables."
        )

    def process_frequencyshift(self, frequency: float):
        self._frequency += frequency

    def process_delay(self, samples: int):
        self._duration += samples
