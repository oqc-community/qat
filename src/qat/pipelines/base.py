# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod

from qat.backend.base import BaseBackend
from qat.engines.native import NativeEngine
from qat.frontend import BaseFrontend
from qat.middleend.middleends import BaseMiddleend
from qat.model.target_data import TargetData
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.base import BaseRuntime


class AbstractPipeline(ABC):
    """An abstraction of pipelines used in QAT to compile and execute quantum programs.

    A pipeline is composed of many different components used to specify how a quantum
    program is compiled and executed. On the compilation front, there is the:

    #. Model: Contains calibration information surrounding the QPU.
    #. TargetData: Contains information about the target device.
    #. Frontend: Compiles a high-level language-specific, but target-agnostic,
       input (e.g., QASM, QIR, ...) to QAT's intermediate representation (IR), QatIR.
    #. Middleend: Takes the QatIR and performs a sequences of  passes that validate and
       optimise the IR, and prepare it for codegen.
    #. Backend: Handles code generation to allow the program to be executed on the target.

    On the execution front, there is the:

    #. Engine: Communicates the compiled program with the target devices, and returns the
       results.
    #. Runtime: Manages the execution of the program, including the engine and the post-
       processing of the results.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> QuantumHardwareModel: ...

    @property
    @abstractmethod
    def target_data(self) -> TargetData: ...

    @property
    @abstractmethod
    def frontend(self) -> BaseFrontend: ...

    @property
    @abstractmethod
    def middleend(self) -> BaseMiddleend: ...

    @property
    @abstractmethod
    def backend(self) -> BaseBackend: ...

    @property
    @abstractmethod
    def runtime(self) -> BaseRuntime: ...

    @property
    @abstractmethod
    def engine(self) -> NativeEngine: ...
