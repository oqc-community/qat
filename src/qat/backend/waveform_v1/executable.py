# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
# ruff: noqa F401
from warnings import warn


from qat.backend.waveform.executable import (
    PositionalAcquireData as PositionalAcquireData,
)
from qat.backend.waveform.executable import (
    WaveformChannelData as WaveformV1ChannelData,
)
from qat.backend.waveform.executable import (
    WaveformProgram as WaveformV1Program,
)

warn(
    "WaveformV1 support is deprecated and will be removed in v4.",
    DeprecationWarning,
    stacklevel=2,
)
