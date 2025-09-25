# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from qat.model.device import PulseChannel


class CustomPulseChannel(PulseChannel):
    """Represents a pulse channel defined at the level of the IR, allowing users to play
    pulses at custom frequencies which might coexist with the standard logical channels.

    In addition to the standard PulseChannel properties seen in the hardware model, a
    physical channel id is stored to link it to a physical channel on hardware.

    TODO: this is a half-step towards treating pulse channels as an IR construct. This
    should eventually be done in full to also treat pulse channels derived from the
    hardware model as IR instructions. (COMPILER-761)
    """

    physical_channel_id: str
