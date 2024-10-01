# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import List, Optional

from compiler_config.config import CompilerConfig

from qat.purr.backends.live import LiveHardwareModel, build_lucy_hardware
from qat.purr.backends.live_devices import ControlHardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.instructions import Instruction, QuantumInstruction
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.qat import execute

log = get_default_logger()


class QPUVersion:
    def __init__(self, make: str, version: str = None):
        self.make = make
        self.version = version

    def __repr__(self):
        return f"{self.make}-{self.version if self.version is not None else 'latest'}"

    @staticmethod
    def with_version(version: str = None):
        """Creates a QPU version with an empty QPU make. Only used in very special circumstances."""
        return QPUVersion("", version)


def inject_name(cls):
    """
    Decorator to inject class name into the QPU make/version fields
    in our static make/model naming objects.
    """
    for value in vars(cls).values():
        if isinstance(value, QPUVersion):
            if value.make is None or value.make == "":
                value.make = cls.__name__

    return cls


@inject_name
class Lucy:
    Latest = QPUVersion.with_version()


class VerificationModel(LiveHardwareModel):
    def __init__(
        self,
        qpu_version,
        verification_engine: "VerificationEngine",
        control_hardware: ControlHardware = ControlHardware(),
    ):
        super().__init__(control_hardware)
        self.version = qpu_version
        self.verification_engine = verification_engine

    def create_engine(self) -> "VerificationEngine":
        return self.verification_engine(self)


def verify_instructions(builder: InstructionBuilder, qpu_type: QPUVersion):
    """
    Runs instruction verification for the instructions in this builder.

    Only run this on instructions that will go through no more transformations
    before being sent to the driver, otherwise you cannot rely upon its results as
    being accurate. In most situations this will only be fully truthful on instructions that
    have already gone through the entire pipeline.
    """
    model = get_verification_model(qpu_type)
    if model is None:
        raise ValueError(
            f"Cannot verify instructions, {qpu_type} isn't a valid QPU version."
        )

    engine: VerificationEngine = model.get_engine()
    engine.verify_instructions(builder.instructions)


def get_verification_model(qpu_type: QPUVersion) -> Optional[VerificationModel]:
    """
    Get verification model for a particular QPU make and model. Each make has its own class,
    which has a field that is each individual version available for verification.

    For example, if you wanted to verify our Lucy machine, that'd be done with:
    ``
    get_verification_model(Lucy.Latest)
    ``

    Or with a specific version:
    ``
    get_verification_model(Lucy.XY)
    ``
    """
    if not isinstance(qpu_type, QPUVersion):
        raise ValueError(
            f"{qpu_type} is not a QPU version, can't find verification engine."
        )

    if qpu_type.make == Lucy.__name__:
        return build_lucy_hardware(VerificationModel(qpu_type, LucyVerificationEngine))

    return None


class VerificationEngine(QuantumExecutionEngine, ABC):
    def _execute_on_hardware(
        self, sweep_iterator, package: QatFile, interrupt: Interrupt = NullInterrupt()
    ):
        while not sweep_iterator.is_finished():
            sweep_iterator.do_sweep(package.instructions)

            metadata = {"sweep_iteration": sweep_iterator.get_current_sweep_iteration()}
            interrupt.if_triggered(metadata, throw=True)
            self.verify_instructions(package.instructions, package.meta_instructions)

        return dict()

    def _process_results(self, results, qfile: "QatFile"):
        return results

    def _process_assigns(self, results, qfile: "QatFile"):
        return results

    @abstractmethod
    def verify_instructions(self, instructions: List[Instruction], metadata): ...


class LucyVerificationEngine(VerificationEngine):

    max_circuit_duration = 90000e-9

    def verify_instructions(self, instructions: List[QuantumInstruction], metadata):
        timeline = self.create_duration_timeline(instructions)

        pc2samples = {pc: positions[-1].end for pc, positions in timeline.items()}
        durations = {pc: samples * pc.sample_time for pc, samples in pc2samples.items()}

        circuit_duration = max([duration for duration in durations.values()])

        log.debug(f"The circuit duration is {circuit_duration/1e-6} microseconds.")

        if circuit_duration > self.max_circuit_duration:
            raise VerificationError(
                f"Circuit duration exceeds maximum allowed. Duration {circuit_duration}, max: {self.max_circuit_duration}."
            )


def verify_program(program: str, compiler_config: CompilerConfig, qpu_version: QPUVersion):
    model = get_verification_model(qpu_version)
    if model is None:
        raise ValueError("Unable to find model to verify against.")

    return execute(program, model, compiler_config)


class VerificationError(ValueError): ...
