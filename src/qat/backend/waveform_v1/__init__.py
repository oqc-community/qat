# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import warnings

# TODO - remove when waveform v1 support is completely removed
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from qat.backend.waveform_v1.executable import (
        PositionalAcquireData as PositionalAcquireData,
        WaveformV1ChannelData as WaveformV1ChannelData,
        WaveformV1Program as WaveformV1Program,
    )

from qat.backend.waveform_v1.purr.codegen import WaveformV1Backend as WaveformV1Backend
