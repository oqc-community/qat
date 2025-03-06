# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.engines.native import NativeEngine
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.utils.logger import get_default_logger
from qat.runtime.executables import AcquireDataStruct, Executable

log = get_default_logger()


class ZeroEngine(NativeEngine):
    """An engine primarily used for testing that simply returns zero readout.

    The engine is designed to be generally applicable to most executables. It looks into the
    executable to retrieve all acquisitions, and creates appropiate readout data that is
    just filled with zeros.
    """

    def execute(self, package: Executable) -> dict[str, np.ndarray]:
        """Execute an :class:`Executable`, returning zeros for all readouts.

        :param package: The compiled executable containing acquisitions.
        :returns: The zero readout results.
        """
        shots = package.compiled_shots if package.compiled_shots else package.shots
        results = {}
        for acquire in package.acquires:
            results[acquire.output_variable] = np.zeros(
                readout_shape(acquire, shots), dtype=np.complex128
            )
        return results


def readout_shape(acquire: AcquireDataStruct, shots: int) -> tuple[int, ...]:
    """Generates the shape of the readout given the acquire information and number of shots.

    :param acquire: The acquisition information.
    :param shots: The number of shots being executed.
    :raises NotImplementedError: Only `AcquireMode.RAW`, `AcquireMode.INTEGRATOR` and
        `AcquireMode.SCOPE` are supported.
    :return: The shape of the returned array.
    """
    match acquire.mode:
        case AcquireMode.RAW:
            return (
                shots,
                acquire.length,
            )
        case AcquireMode.INTEGRATOR:
            return (shots,)
        case AcquireMode.SCOPE:
            return (acquire.length,)
    raise NotImplementedError(f"Acquire mode {acquire.mode} not currently supported.")
