# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
from compiler_config.config import CompilerConfig, QuantumResultsFormat, ResultsFormatting

from qat.core.pass_base import ValidationPass
from qat.purr.backends.qiskit_simulator import QiskitBuilder
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import CustomPulse, Pulse, PulseShapeType, Synchronize
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
        if format_flags == None or not ResultsFormatting.BinaryCount in format_flags:
            log.warning(
                "The results formatting `BinaryCount` was not found in the formatting "
                "flags. Please note that the Qiskit runtime only currently supports "
                "results returned as a binary count."
            )
        return ir
