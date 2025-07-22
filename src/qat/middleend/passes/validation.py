# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig, ErrorMitigationConfig, ResultsFormatting

from qat.core.config.configure import get_config
from qat.core.pass_base import ValidationPass
from qat.ir.instruction_builder import InstructionBuilder
from qat.ir.instructions import Return
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    PostProcessing,
    ProcessAxis,
    Pulse,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class NoMidCircuitMeasurementValidation(ValidationPass):
    """
    Validates that there are no mid-circuit measurements by checking that no qubit
    has an acquire instruction that is later followed by a pulse instruction.
    """

    def __init__(
        self,
        model: PhysicalHardwareModel,
        no_mid_circuit_measurement: bool | None = None,
        *args,
        **kwargs,
    ):
        """
        :param model: The hardware model.
        :param no_mid_circuit_measurement: Whether mid-circuit measurements are allowed.
            If None, uses the default from the QatConfig.
        """
        self.model = model
        self.no_mid_circuit_measurement = (
            no_mid_circuit_measurement
            if no_mid_circuit_measurement is not None
            else get_config().INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        )

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The intermediate representation (IR) :class:`InstructionBuilder`.
        """
        consumed_acquire_pc: set[str] = set()

        if not self.no_mid_circuit_measurement:
            return ir

        drive_acq_pc_map = {
            qubit.drive_pulse_channel.uuid: qubit.acquire_pulse_channel.uuid
            for qubit in self.model.qubits.values()
        }

        for instr in ir:
            if isinstance(instr, Acquire):
                consumed_acquire_pc.add(instr.target)

            # Check if we have a measure in the middle of the circuit somewhere.
            elif isinstance(instr, Pulse):
                acq_pc = drive_acq_pc_map.get(instr.target, None)

                if acq_pc and acq_pc in consumed_acquire_pc:
                    raise ValueError(
                        "Mid-circuit measurements currently unable to be used."
                    )
        return ir


class ReadoutValidation(ValidationPass):
    """Validates that the post-processing instructions do not have an invalid sequence.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.validate`.
    """

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        acquire_modes = {}
        for inst in ir:
            if isinstance(inst, Acquire):
                acquire_modes[inst.output_variable] = inst.mode
            if isinstance(inst, PostProcessing):
                acquire_mode = acquire_modes.get(inst.output_variable, None)
                self._post_processing_options_handling(inst, acquire_mode)
        return ir

    @staticmethod
    def _post_processing_options_handling(
        inst: PostProcessing, acquire_mode: AcquireMode | None
    ):
        if acquire_mode == AcquireMode.SCOPE and ProcessAxis.SEQUENCE in inst.axes:
            raise ValueError(
                "Invalid post-processing! Post-processing over SEQUENCE is "
                "not possible after the result is returned from hardware "
                "in SCOPE mode!"
            )
        elif acquire_mode == AcquireMode.INTEGRATOR and ProcessAxis.TIME in inst.axes:
            raise ValueError(
                "Invalid post-processing! Post-processing over TIME is not "
                "possible after the result is returned from hardware in "
                "INTEGRATOR mode!"
            )
        elif acquire_mode == AcquireMode.RAW:
            raise ValueError(
                "Invalid acquire mode! The live hardware doesn't support RAW acquire mode!"
            )
        elif acquire_mode is None:
            raise ValueError(
                f"No AcquireMode found with output variable {inst.output_variable},"
                f"ensure PostProcessing output_variable matches an Acquire output_variable with a"
                f"valid AcquireMode selected."
            )


class HardwareConfigValidity(ValidationPass):
    """Validates the :class:`CompilerConfig` against the hardware model."""

    def __init__(self, hardware_model: PhysicalHardwareModel, max_shots: int | None = None):
        """Instantiate the pass with a hardware model.

        :param hardware_model: The hardware model.
        :param max_shots: The maximum number of shots allowed for a single task.
                If None, uses the default from the QatConfig.
        """
        self.hardware_model = hardware_model
        self.max_shots = (
            max_shots if max_shots is not None else get_config().MAX_REPEATS_LIMIT
        )

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
        if compiler_config.repeats > self.max_shots:
            raise ValueError(
                f"Number of shots in compiler config {compiler_config.repeats} exceeds max "
                f"number of shots {self.max_shots}."
            )

    def _validate_error_mitigation(
        self, hardware_model: PhysicalHardwareModel, compiler_config: CompilerConfig
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


class ReturnSanitisationValidation(ValidationPass):
    """Validates that the IR has a :class:`Return` instruction."""

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        returns = [inst for inst in ir.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions.")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions.")
        return ir


PydReadoutValidation = ReadoutValidation
PydHardwareConfigValidity = HardwareConfigValidity
PydNoMidCircuitMeasurementValidation = NoMidCircuitMeasurementValidation
PydReturnSanitisationValidation = ReturnSanitisationValidation
