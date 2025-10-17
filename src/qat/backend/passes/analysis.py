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
        self.model = model

    def run(
        self, ir: PartitionedIR, res_mgr: ResultManager, *args, **kwargs
    ) -> PartitionedIR:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        baseband_freqs = {}
        baseband_freqs_fixed_if = {}

        for pulse_channel in ir.pulse_channels.values():
            physical_channel = pulse_channel.physical_channel_id
            if pulse_channel.fixed_if:
                baseband_if_freq = self.model.physical_channel_with_id(
                    physical_channel
                ).baseband.if_frequency
                baseband_freq = pulse_channel.frequency - UPCONVERT_SIGN * baseband_if_freq
                self._check_fixed_if_not_violated(
                    baseband_freqs, physical_channel, baseband_freq
                )
                baseband_freqs[physical_channel] = baseband_freq
                baseband_freqs_fixed_if[physical_channel] = pulse_channel.fixed_if
            elif not baseband_freqs_fixed_if.get(physical_channel, False):
                baseband_freqs_fixed_if[physical_channel] = pulse_channel.fixed_if

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
        self.sample_times_map = dict()
        for qubit in model.qubits.values():
            self.sample_times_map[qubit.physical_channel.uuid] = (
                target_data.QUBIT_DATA.sample_time
            )
            self.sample_times_map[qubit.resonator.physical_channel.uuid] = (
                target_data.RESONATOR_DATA.sample_time
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

            sample_time = self.sample_times_map[
                ir.get_pulse_channel(pulse_ch_id).physical_channel_id
            ]
            durations = self.durations_as_samples(pulse_channel_durations, sample_time)
            cumulative_durations = np.cumsum(durations)
            result.target_map[pulse_ch_id] = PulseChannelTimeline(
                samples=durations,
                start_positions=cumulative_durations - durations,
                end_positions=cumulative_durations - 1,
            )

        result.total_duration = total_duration
        res_mgr.add(result)
        return ir

    def durations_as_samples(self, durations: list[float], sample_time: float):
        """Converts a list of durations into a number of samples."""
        block_numbers = np.ceil(np.round(durations / sample_time, decimals=4)).astype(
            np.int64
        )
        return block_numbers


PydIntermediateFrequencyResult = IntermediateFrequencyResult
PydIntermediateFrequencyAnalysis = IntermediateFrequencyAnalysis
PydTimelineAnalysisResult = TimelineAnalysisResult
PydTimelineAnalysis = TimelineAnalysis
