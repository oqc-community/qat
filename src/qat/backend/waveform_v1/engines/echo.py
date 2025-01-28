# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.backend.waveform_v1 import WaveformV1Executable
from qat.purr.compiler.instructions import AcquireMode
from qat.purr.utils.logger import get_default_logger
from qat.runtime import NativeEngine

log = get_default_logger()


class EchoEngine(NativeEngine):
    """
    The EchoEngine is a minimal execution engine primarily used for testing the compilation
    pipeline. It is not connected to any backend such as live hardware or a simulator, and
    just simply "echos" back the buffers.

    Currently only accepts `WaveformV1Executables`, but this may be changed in the future.
    """

    def execute(self, package: WaveformV1Executable):
        shots = package.compiled_shots if package.compiled_shots else package.shots
        results = {}
        for channel_data in package.channel_data.values():
            buffer = channel_data.buffer
            for acquire in channel_data.acquires:
                results[acquire.output_variable] = process_readout(
                    buffer[acquire.position : acquire.position + acquire.length],
                    shots,
                    acquire.mode,
                )
        return results

    def startup(self):
        log.info("Starting the echo engine.")

    def shutdown(self):
        log.info("Shutting down the echo engine.")


def process_readout(readout: np.ndarray, shots: int, mode: AcquireMode):
    """
    Processes a single readout into the expected format for a given acquire mode.

    For `AcquireMode.RAW`, this means repeating the readout for a given number of shots.
    The `AcquireMode.INTEGRATOR` emulates the averaging on hardware by taking the mean;
    note that this does not currently include down scaling. Finally, the `AcquireMode.SCOPE`
    simply returns the readout back in its current form.
    """
    match mode:
        case AcquireMode.RAW:
            return np.tile(readout, shots).reshape((shots, -1))
        case AcquireMode.INTEGRATOR:
            return np.tile(np.mean(readout, axis=0), shots)
        case AcquireMode.SCOPE:
            return readout
    raise NotImplementedError(f"Acquire mode {mode} not currently supported.")
