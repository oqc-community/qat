from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from qat.core.pass_base import AnalysisPass
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.ir.instructions import QuantumInstruction
from qat.ir.lowered import PartitionedIR
from qat.model.device import (
    PhysicalChannel,
    Qubit,
    Resonator,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.target_data import TargetData
from qat.purr.backends.utilities import UPCONVERT_SIGN


@dataclass
class IntermediateFrequencyResult(ResultInfoMixin):
    frequencies: Dict[PhysicalChannel, float]


class IntermediateFrequencyAnalysis(AnalysisPass):
    """
    Adapted from :meth:`qat.purr.backends.live.LiveDeviceEngine.build_baseband_frequencies`.

    Retrieves intermediate frequencies for all physical channels if they exist,
    and validates that pulse channels that share the same physical channel cannot
    have differing fixed frequencies. This pass should always follow a :class:`PartitionByPulseChannel`,
    as information of pulse channels are needed.
    """

    def __init__(self, model: PhysicalHardwareModel):
        """
        Instantiate the pass with a hardware model.

        :param model: The hardware model.
        """

        # TODO: determine if this pass should be split into an analysis and validation
        #   pass. (COMPILER-610)
        self.channel_data = self._build_channel_data(model)

    def _build_channel_data(
        self, hardware_model: PhysicalHardwareModel | None
    ) -> dict | None:
        """
        Builds a dictionary of channel data based on the provided hardware model.
        The dictionary returned is of the form:
        channel_data = {
            pulse_channel_1_id:  {
                "frequency": pulse_channel_1,
                "physical_channel": physical_channel_1
            },
            ...
        }
        """
        channel_data = {}
        for qubit in hardware_model.qubits.values():
            channel_data.update(self._build_device_channel_data(qubit))
            channel_data.update(self._build_device_channel_data(qubit.resonator))
        return channel_data

    @staticmethod
    def _build_device_channel_data(device: Qubit | Resonator) -> dict:
        physical_channel = device.physical_channel
        channels = {}
        for pulse_channel in device.all_pulse_channels:
            channels[pulse_channel.uuid] = {
                "fixed_if": pulse_channel.fixed_if,
                "baseband_freq": pulse_channel.frequency
                - UPCONVERT_SIGN * physical_channel.baseband.if_frequency,
                "physical_channel": physical_channel.uuid,
                "physical_channel_baseband_freq": physical_channel.baseband.if_frequency,
            }
        return channels

    def run(
        self, ir: PartitionedIR, res_mgr: ResultManager, *args, **kwargs
    ) -> PartitionedIR:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        baseband_freqs = {}
        baseband_freqs_fixed_if = {}

        for pulse_channel in ir.target_map:
            physical_channel = self.channel_data[pulse_channel]["physical_channel"]
            baseband_freq = self.channel_data[pulse_channel]["baseband_freq"]

            if fixed_if := self.channel_data[pulse_channel]["fixed_if"]:
                self._check_fixed_if_not_violated(
                    baseband_freqs, physical_channel, baseband_freq
                )
                baseband_freqs[physical_channel] = baseband_freq
                baseband_freqs_fixed_if[physical_channel] = fixed_if
            elif not baseband_freqs_fixed_if.get(physical_channel, False):
                baseband_freqs_fixed_if[physical_channel] = fixed_if

        res_mgr.add(IntermediateFrequencyResult(frequencies=baseband_freqs))
        return ir

    @staticmethod
    def _check_fixed_if_not_violated(
        baseband_freqs: dict, physical_channel: str, baseband_freq: float
    ) -> None:
        if baseband_freqs.get(physical_channel, baseband_freq) != baseband_freq:
            raise ValueError(
                "Cannot fix the frequency for two pulse channels of different "
                "frequencies on the same physical channel!"
            )


@dataclass
class PulseChannelTimeline:
    """Timeline analysis for instructions on a pulse channel.

    Imagine the timeline for a pulse channel, with an instruction that occurs over samples
    3-7, i.e.,

        samples: 0 1 2 [3 4 5 6 7] 8 9 10.

    The `start_position` would be 3, the `end_position` 7, and the number of `samples` 5.

    :param np.ndarray[int] samples: The number of samples each instruction takes.
    :param np.ndarray[int] start_positions: The sample when the instruction begins.
    :param np.ndarray[int] end_positions: The sample when the instruction ends.
    """

    samples: np.ndarray[int] = field(default_factory=lambda: np.ndarray([]))
    start_positions: np.ndarray[int] = field(default_factory=lambda: np.ndarray([]))
    end_positions: np.ndarray[int] = field(default_factory=lambda: np.ndarray([]))


@dataclass
class TimelineAnalysisResult(ResultInfoMixin):
    """Stores the timeline analysis for all pulse channels.

    :param target_map: The dictionary containing the timeline analysis for all pulse
        channels.
    """

    target_map: dict[str, PulseChannelTimeline] = field(
        default_factory=lambda: defaultdict(PulseChannelTimeline)
    )
    total_duration: float = field(default=0.0)


class TimelineAnalysis(AnalysisPass):
    """Analyses the timeline of each pulse channel.

    Takes the instruction list for each pulse channel retrieved from the the partitioned
    results, and calculates the timeline in units of samples (each sample takes time
    `sample_time`). It calculates the duration of each instruction in units of samples,
    and the start and end times of each instruction in units of samples.

    .. warning::

        The pass will assume that the durations of instructions are sanitised to the
        granularity of the channels. If instructions that do not meet the criteria are
        provided, it might produce incorrect timelines. This can be enforced used the
        :class:`InstructionGranularitySanitisation <qat.middleend.passes.transform.InstructionGranularitySanitisation>`
        pass.
    """

    def __init__(self, model: PhysicalHardwareModel, target_data: TargetData):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        :param target_data: Target-related information.
        """
        self.model = model

        q_sample_time = target_data.QUBIT_DATA.sample_time
        r_sample_time = target_data.RESONATOR_DATA.sample_time
        self.pulse_ch_ids_sample_time = {}
        for qubit in model.qubits.values():
            self.pulse_ch_ids_sample_time.update(
                {
                    pulse_channel.uuid: q_sample_time
                    for pulse_channel in qubit.all_pulse_channels
                }
            )
            self.pulse_ch_ids_sample_time.update(
                {
                    pulse_channel.uuid: r_sample_time
                    for pulse_channel in qubit.resonator.all_pulse_channels
                }
            )

    def run(
        self, ir: PartitionedIR, res_mgr: ResultManager, *args, **kwargs
    ) -> PartitionedIR:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """
        target_map = ir.target_map

        result = TimelineAnalysisResult()
        total_duration = 0
        for pulse_ch_id, instructions in target_map.items():
            pulse_channel_durations = np.array(
                [
                    (inst.duration if isinstance(inst, QuantumInstruction) else 0.0)
                    for inst in instructions
                ]
            )
            total_duration = max(total_duration, np.sum(pulse_channel_durations))

            durations = self.durations_as_samples(pulse_ch_id, pulse_channel_durations)
            cumulative_durations = np.cumsum(durations)
            result.target_map[pulse_ch_id] = PulseChannelTimeline(
                samples=durations,
                start_positions=cumulative_durations - durations,
                end_positions=cumulative_durations - 1,
            )

        result.total_duration = total_duration
        res_mgr.add(result)
        return ir

    def durations_as_samples(self, pulse_ch_id: str, durations: list[float]):
        """Converts a list of durations into a number of samples."""
        sample_time = self.pulse_ch_ids_sample_time[pulse_ch_id]
        block_numbers = np.ceil(np.round(durations / sample_time, decimals=4)).astype(
            np.int64
        )
        return block_numbers


PydIntermediateFrequencyResult = IntermediateFrequencyResult
PydIntermediateFrequencyAnalysis = IntermediateFrequencyAnalysis
PydTimelineAnalysisResult = TimelineAnalysisResult
PydTimelineAnalysis = TimelineAnalysis
