# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.engines.native import NativeEngine
from qat.executables import AbstractProgram
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class ZeroEngine(NativeEngine):
    """An engine primarily used for testing that simply returns zero readout.

    The engine is designed to be applicable to all programs. It just populates the readout
    data for each output variable with zeros.
    """

    def execute(self, program: AbstractProgram, **kwargs) -> dict[str, np.ndarray]:
        """Execute an :class:`AbstractProgram`, returning zeros for all readouts.

        :param package: The compiled executable containing acquisitions.
        :returns: The zero readout results.
        """
        results = {}
        for acquire, shape in program.acquire_shapes.items():
            results[acquire] = np.zeros(shape, dtype=np.complex128)
        return results
