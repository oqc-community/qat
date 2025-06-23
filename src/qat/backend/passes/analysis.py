from dataclasses import dataclass
from typing import Dict

from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.ir.lowered import PartitionedIR
from qat.model.device import PhysicalChannel, Qubit, Resonator
from qat.model.hardware_model import PhysicalHardwareModel
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


PydIntermediateFrequencyAnalysis = IntermediateFrequencyAnalysis
