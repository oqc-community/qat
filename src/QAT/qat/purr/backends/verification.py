# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from qat.purr.backends.live import LiveHardwareModel, get_default_lucy_hardware
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.emitter import QatFile, InstructionEmitter
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Instruction
from qat.purr.utils.logger import get_default_logger
from numpy import max as np_max

logger = get_default_logger()


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


def verify_instructions(builder: InstructionBuilder, qpu_type: QPUVersion, max_circuit_duration: int = 90000):
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
    engine.verify_instructions(builder.instructions, max_circuit_duration)

def get_verification_model(qpu_type: QPUVersion) -> Union[VerificationModel, QuantumHardwareModel]:
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
        model = get_default_lucy_hardware()
    else:
        raise NotImplementedError(f"No specific model created yet for {qpu_type}.")

    return model


class VerificationEngine(QuantumExecutionEngine, ABC):

    def __init__(self, model):
        super().__init__(model)

    def _execute_on_hardware(self, sweep_iterator, package: QatFile):
        veri_list = list(package.meta_instructions)
        veri_list.extend(package.instructions)
        self.verify_instructions(veri_list)

    @abstractmethod
    def verify_instructions(self, instructions: List[Instruction], max_circuit_duration: int = 90000):
        ...

class LucyVerificationEngine(VerificationEngine):

    def verify_instructions(self, instructions: Union[List[Instruction], InstructionBuilder], max_circuit_duration: int = 90000) -> bool:
        valid_instructions = True

        duration = self.instruction_duration(instructions)
        if not valid_circuit_length(duration, max_circuit_duration):
            valid_instructions = False

        return valid_instructions


def valid_circuit_length(duration: int, max_circuit_duration: int= 90000) -> bool:
    return duration < max_circuit_duration