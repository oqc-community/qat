# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.backend.waveform import WaveformProgram
from qat.engines.native import NativeEngine
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class EchoEngine(NativeEngine[WaveformProgram]):
    """The :class:`EchoEngine` is a minimal execution engine primarily used for testing the
    compilation pipeline.

    It is not connected to any target machine such as live hardware or a simulator, and just
    simply "echos" back the buffers. It only accepts :class:`WaveformProgram` programs.

    It checks the shape of the results generated match the expected shapes from codegen,
    to test consistency across the compilation and execution pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, program: WaveformProgram, **kwargs) -> dict[str, np.ndarray]:
        """
        Execute a :class:`WaveformProgram`.

        :param program: The compiled program.
        :returns: The execution results.
        """
        results = {}
        acquire_shapes = program.acquire_shapes
        for channel_data in program.channel_data.values():
            buffer = np.asarray(channel_data.buffer)
            for acquire in channel_data.acquires:
                results[acquire.output_variable] = process_readout(
                    buffer[acquire.position : acquire.position + acquire.length],
                    program.shots,
                    acquire.mode,
                )

                if (
                    results[acquire.output_variable].shape
                    != acquire_shapes[acquire.output_variable]
                ):
                    raise ValueError(
                        f"Acquired data shape {results[acquire.output_variable].shape} "
                        f"does not match expected shape {acquire.shape} "
                        f"for variable {acquire.output_variable}."
                    )

        return results


def process_readout(readout: np.ndarray, shots: int, mode: AcquireMode):
    """
    Processes a single readout into the expected format for a given acquire mode.

    For :attr:`AcquireMode.RAW`, this means repeating the readout for a given number of
    shots. The :attr:`AcquireMode.INTEGRATOR` emulates the averaging on hardware by
    taking the mean; note that this does not currently include down scaling. Finally, the
    :attr:`AcquireMode.SCOPE` simply returns the readout back in its current form.
    """
    match mode:
        case AcquireMode.RAW:
            return np.tile(readout, shots).reshape((shots, -1))
        case AcquireMode.INTEGRATOR:
            return np.tile(np.mean(readout, axis=0), shots)
        case AcquireMode.SCOPE:
            return readout
    raise NotImplementedError(f"Acquire mode {mode} not currently supported.")
