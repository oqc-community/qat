# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from warnings import warn

from qat.engines.waveform.echo import EchoEngine as NewEchoEngine


class EchoEngine(NewEchoEngine):
    """The :class:`EchoEngine` is a minimal execution engine primarily used for testing the
    compilation pipeline.

    It is not connected to any target machine such as live hardware or a simulator, and just
    simply "echos" back the buffers. It only accepts :class:`WaveformProgram` programs.

    It checks the shape of the results generated match the expected shapes from codegen,
    to test consistency across the compilation and execution pipeline.
    """

    def __init__(self, *args, **kwargs):
        warn(
            "WaveformV1 support is deprecated and will be removed in v4.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
