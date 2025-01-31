# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import List

import numpy as np
from compiler_config.config import CompilerConfig

from qat.ir.pass_base import QatIR, ValidationPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Acquire, FrequencyShift, Repeat, Return
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class RepeatSanitisationValidation(ValidationPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Checks if the builder has a repeat instruction and warns if none exists.
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        if not repeats:
            log.warning("Could not find any repeat instructions")


class ReturnSanitisationValidation(ValidationPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Every builder must have a single return instruction
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        returns = [inst for inst in builder.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions")


class NCOFrequencyVariability(ValidationPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        model = next((a for a in args if isinstance(a, QuantumHardwareModel)), None)

        if not model:
            model = kwargs.get("model", None)

        if not model or not isinstance(model, QuantumHardwareModel):
            raise ValueError(
                f"Expected to find an instance of {QuantumHardwareModel} in arguments list, but got {model} instead"
            )

        for channel in model.pulse_channels.values():
            if channel.fixed_if:
                raise ValueError("Cannot allow constance of the NCO frequency")


class HardwareConfigValidity(ValidationPass):
    def __init__(self, hardware_model: QuantumHardwareModel):
        self.hardware_model = hardware_model

    def run(
        self,
        ir: QatIR,
        res_mgr: ResultManager,
        *args,
        compiler_config: CompilerConfig,
        **kwargs,
    ):
        compiler_config.validate(self.hardware_model)


class FrequencyValidation(ValidationPass):
    """
    This validation pass checks two things:

    #. Frequency shifts do not move the frequency of a pulse channel outside of its
       allowed range.
    #. Frequency shifts do not occur on pulse channels that have a fixed IF, or share a
       physical channel with a pulse channel that has a fixed IF.
    """

    def __init__(self, model: QuantumHardwareModel):
        """
        Instantiate the pass with a hardware model.

        :param QuantumHardwareModel model: The hardware model.
        """

        # TODO: replace with new hardware models as our refactors mature.
        self.model = model

    def run(self, ir: QatIR, *args, **kwargs):
        """
        :param QatIR ir: The :class:`InstructionBuilder` wrapped in :class:`QatIR`.
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        freqshifts = [
            inst for inst in builder.instructions if isinstance(inst, FrequencyShift)
        ]
        targets = set([inst.channel for inst in freqshifts])
        self.validate_frequency_ranges(targets, freqshifts)
        self.validate_no_freqshifts_on_fixed_if(targets)

    @staticmethod
    def validate_frequency_ranges(
        targets: List[PulseChannel], freqshifts: List[FrequencyShift]
    ):
        for target in targets:
            freq_shifts = [inst.frequency for inst in freqshifts if inst.channel == target]
            freqs = target.frequency + np.cumsum(freq_shifts)
            violations = np.logical_or(
                freqs > target.max_frequency, freqs < target.min_frequency
            )
            if np.any(violations):
                raise ValueError(
                    f"Frequency shifts will change the pulse channel frequency to fall "
                    f"out of the allowed range between {target.min_frequency} and "
                    f"{target.max_frequency} for pulse channel {target.full_id()}."
                )

    def validate_no_freqshifts_on_fixed_if(self, targets: List[PulseChannel]):
        for target in targets:
            for channel in self.model.get_pulse_channels_from_physical_channel(
                target.physical_channel
            ):
                if channel.fixed_if:
                    raise NotImplementedError(
                        "Hardware does not currently support frequency shifts on the "
                        f"physical channel {target.physical_channel}."
                    )


class NoAcquireWeightsValidation(ValidationPass):
    """
    Some backends do not support :class:`Acquire` instructions that contain weights. This
    pass can be used to validate that this is the case.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        """
        :param QatIR ir: The :class:`InstructionBuilder` wrapped in :class:`QatIR`.
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        has_filters = [
            inst.filter for inst in builder.instructions if isinstance(inst, Acquire)
        ]
        if any(has_filters):
            raise NotImplementedError(
                "Acquire filters are not implemented for this backend."
            )


class NoMultipleAcquiresValidation(ValidationPass):
    """
    Some backends do not support multiple :class:`Acquire` instructions on the same channel.
    This validation pass should be used to verify this.
    """

    def run(self, ir: QatIR, *args, **kwargs):
        """
        :param QatIR ir: The :class:`InstructionBuilder` wrapped in :class:`QatIR`.
        """
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        physical_channels = [
            inst.channel.physical_channel_id
            for inst in builder.instructions
            if isinstance(inst, Acquire)
        ]
        if len(physical_channels) != len(set(physical_channels)):
            raise NotImplementedError(
                "Multiple acquisitions on a single channel is not supported for this backend."
            )


# TODO - bring in stuff from verification.py in here in the form of a pass (or a bunch of passes)
