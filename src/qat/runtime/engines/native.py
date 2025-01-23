# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import abc
from typing import Dict

import numpy as np

from qat.runtime.executables import Executable


class NativeEngine(abc.ABC):

    def __init__(self, startup: bool = False):
        """
        NativeEngines act as an interface between some target backend and an executable. They
        are used to connect to the backend (if applicable), and execute and return the results.
        """

        if startup:
            self.startup()

    @abc.abstractmethod
    def startup(self): ...

    @abc.abstractmethod
    def shutdown(self): ...

    @abc.abstractmethod
    def execute(self, package: Executable) -> Dict[str, np.ndarray]:
        """
        Executes a compiled instruction executable and returns results that are processed
        according to the acquires.

        The engine is expected to return the results as a dictionary with the output variables
        as keys. This may be changed in future iterations.
        """
        ...

    def __del__(self):
        self.shutdown()
