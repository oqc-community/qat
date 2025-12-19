# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pydantic import BaseModel, Field


class QRPairModel(BaseModel):
    """
    Used to parse JSON/CSV entries. Qubit and Resonator are specified together in pairs.
    """

    label: str = Field(alias="LABEL")
    index: int = Field(frozen=True, alias="INDEX")
    connections: str = Field(alias="CONNECTIONS")

    q_freq: float = Field(alias="Q-FREQ")
    q_freq12: float = Field(alias="Q-FREQ12")
    q_cluster: str = Field(alias="Q-CLUSTER")
    q_slot_idx: int = Field(alias="Q-SLOT-IDX")
    q_output: int = Field(alias="Q-OUTPUT")

    r_freq: float = Field(alias="R-FREQ")
    r_cluster: str = Field(alias="R-CLUSTER")
    r_slot_idx: int = Field(alias="R-SLOT-IDX")
    r_output: int = Field(alias="R-OUTPUT")
    r_input: int = Field(alias="R-INPUT")

    def qubit_physical_channel_id(self):
        return f"{self.q_cluster}-QCM-RF-{self.q_slot_idx}-OUT-{self.q_output}"

    def resonator_physical_channel_id(self):
        return f"{self.r_cluster}-QRM-RF-{self.r_slot_idx}-OUT-{self.r_output}-IN-{self.r_input}"
