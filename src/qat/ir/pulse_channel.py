# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from uuid import uuid4

from pydantic import BaseModel, Field


class PulseChannel(BaseModel):
    """An IR-level representation of a pulse channel.

    This holds the properties of a pulse channel that are relevant at compile-time, and
    decouples it from the hardware model, allowing dynamic representation and creation of
    hardware models.

    Eventually, this might be replaced with a more formal IR operation the declares pulse
    channels a symbols, with lookups defer to symbol tables.
    """

    uuid: str = Field(default_factory=lambda: str(uuid4()), frozen=True)
    frequency: float
    imbalance: float = 1.0
    phase_iq_offset: float = 0.0
    scale: float | complex = 1.0 + 0.0j
    physical_channel_id: str

    def __hash__(self):
        return hash(self.uuid)
