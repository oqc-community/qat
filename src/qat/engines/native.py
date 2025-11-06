# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from qat.executables import Program


class NativeEngine(ABC, Generic[Program]):
    """:class:`NativeEngine` acts as an interface between some target machine and an
    executable. They are used to connect to the target machine (if applicable), and execute
    and return the results."""

    @abstractmethod
    def execute(self, program: Program, **kwargs) -> dict[str, np.ndarray]:
        """Executes a compiled instruction program and returns results that are processed
        according to the acquires.

        The engine is expected to return the results as a dictionary with the output
        variables as keys.
        """
        ...


class ConnectionMixin(ABC):
    """Specifies a connection requirement for a :class:`NativeEngine`.

    Engines that execute on live hardware or a remote simulator might require a connection
    to the target to be established. This class is used to mix in connection capabilities.
    """

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def disconnect(self): ...

    def __del__(self):
        if self.is_connected:
            self.disconnect()
