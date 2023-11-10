# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import List, Optional

from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.instructions import Instruction


class QPUVersion:
    def __init__(self, make: str, version: str = None):
        self.make = make
        self.version = version

    def __repr__(self):
        return f"{self.make}-{self.version if self.version is not None else 'latest'}"

    @staticmethod
    def with_version(version: str = None):
        """ Creates a QPU version with an empty QPU make. Only used in very special circumstances. """
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
    def __init__(self, qpu_version, verification_engine_type: type):
        super().__init__(None, [verification_engine_type], None)
        self.version = qpu_version


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
        # TODO: Should have an apply_setup_to_hardware in live which people can use to build our own architecture.
        model = VerificationModel(qpu_type, LucyVerificationEngine)
        raise NotImplementedError("No lucy-specific model created yet.")

    return None


class VerificationEngine(QuantumExecutionEngine, ABC):
    def _execute_on_hardware(self, sweep_iterator, package: QatFile):
        veri_list = list(package.meta_instructions)
        veri_list.extend(package.instructions)
        self.verify_instructions(veri_list)

    @abstractmethod
    def verify_instructions(self, instructions: List[Instruction]):
        ...


class LucyVerificationEngine(VerificationEngine):
    def verify_instructions(self, instructions: List[Instruction]):
        # TODO: Add actual verification
        raise NotImplementedError("No lucy-specific verification yet.")
