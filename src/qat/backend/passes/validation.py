# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import List

import numpy as np
from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat import qatconfig
from qat.core.pass_base import ValidationPass
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Acquire, FrequencyShift, Repeat, Return
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class RepeatSanitisationValidation(ValidationPass):
    """Checks if the builder has a :class:`Repeat` instruction and warns if none exists."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        if not repeats:
            log.warning("Could not find any repeat instructions")
        return ir


class ReturnSanitisationValidation(ValidationPass):
    """Validates that the IR has a :class:`Return` instruction."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions")
        return ir


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

    def __init__(self, hardware_model: PydHardwareModel):
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
        self._validate_shots(compiler_config)
        self._validate_error_mitigation(self.hardware_model, compiler_config)
        return ir

    def _validate_shots(self, compiler_config: CompilerConfig):
        if compiler_config.repeats > qatconfig.MAX_REPEATS_LIMIT:
            raise ValueError(
                f"Number of shots in compiler config {compiler_config.repeats} exceeds max "
                f"number of shots {qatconfig.MAX_REPEATS_LIMIT}."
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

    def __init__(self, model: QuantumHardwareModel):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model.
        """

        # TODO: replace with new hardware models as our refactors mature.
        self.model = model

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        """

        freqshifts = [inst for inst in ir.instructions if isinstance(inst, FrequencyShift)]
        targets = set([inst.channel for inst in freqshifts])
        self.validate_frequency_ranges(targets, freqshifts)
        self.validate_no_freqshifts_on_fixed_if(targets)
        return ir

    @staticmethod
    def validate_frequency_ranges(
        targets: List[PulseChannel], freqshifts: List[FrequencyShift]
    ):
        """Validates that a pulse channel remains within its allowed frequency range.

        :param targets: List of pulse channels to validate.
        :param freqshifts: List of frequency shift instructions.
        """
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


# TODO - bring in stuff from verification.py in here in the form of a pass (or a bunch of passes)
