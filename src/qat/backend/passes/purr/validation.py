# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import numpy as np

from qat.core.pass_base import ValidationPass
from qat.ir.lowered import PartitionedIR
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Acquire, CustomPulse, Pulse
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class NCOFrequencyVariability(ValidationPass):
    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        model = next((a for a in args if isinstance(a, QuantumHardwareModel)), None)

        if not model:
            model = kwargs.get("model", None)

        if not model or not isinstance(model, QuantumHardwareModel):
            raise ValueError(
                f"Expected to find an instance of {QuantumHardwareModel} in arguments "
                f"list, but got {model} instead"
            )

        for channel in model.pulse_channels.values():
            if channel.fixed_if:
                raise ValueError("Cannot allow constance of the NCO frequency")
        return ir


class NoAcquireWeightsValidation(ValidationPass):
    """Some target machines do not support :class:`Acquire` instructions that contain weights.
    This pass can be used to validate that this is the case."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        has_filters = [inst.filter for inst in ir.instructions if isinstance(inst, Acquire)]
        if any(has_filters):
            raise NotImplementedError(
                "Acquire filters are not implemented for this target machine."
            )
        return ir


class NoMultipleAcquiresValidation(ValidationPass):
    """Some target machines do not support multiple :class:`Acquire` instructions on the
    same channel. This validation pass should be used to verify this."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        physical_channels = [
            inst.channel.physical_channel_id
            for inst in ir.instructions
            if isinstance(inst, Acquire)
        ]
        if len(physical_channels) != len(set(physical_channels)):
            raise NotImplementedError(
                "Multiple acquisitions on a single channel is not supported for this target machine."
            )
        return ir


class NoAcquiresWithDifferentWeightsValidation(ValidationPass):
    """Some target machines do not support multiple :class:`Acquire` instructions with different
    filters on the same pulse channel. This validation pass should be used to verify this.
    """

    def run(self, ir: PartitionedIR, *args, **kwargs):
        for pulse_ch, acquires in ir.acquire_map.items():
            samples = []
            for acquire in acquires:
                if isinstance(acquire.filter, CustomPulse):
                    samples.append(acquire.filter.samples)
                elif isinstance(acquire, Pulse):
                    raise TypeError(
                        "Type of the acquire filter can only be `CustomPulse` or `None`."
                    )

            samples = np.array(samples)
            if not np.all(samples == samples[0]):
                raise ValueError(
                    f"Cannot have multiple `Acquire`s on the same pulse channel ({pulse_ch.full_id()}) with different weights."
                )

        return ir


# TODO - bring in stuff from verification.py in here in the form of a pass (or a bunch of passes)
