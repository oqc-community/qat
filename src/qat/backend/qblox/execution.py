# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd

from pydantic import BaseModel, ConfigDict, Field
from pydantic_extra_types.semantic_version import SemanticVersion

from qat.backend.qblox.config.specification import ModuleConfig, SequencerConfig
from qat.backend.qblox.ir import Sequence
from qat.executables import AbstractProgram
from qat.utils.pydantic import ComplexNDArray

# TODO(COMPILER-1455, COMPILER-1456): Temporary static acquisition-timeout policy.
DEFAULT_TIMEOUT_SECONDS: float = 20 * 60


class QbloxPackage(BaseModel):
    pulse_channel_id: str | None = None
    physical_channel_id: str | None = None
    instrument_id: str | None = None
    seq_idx: int | None = None
    seq_config: SequencerConfig = Field(default_factory=lambda: SequencerConfig())
    slot_idx: int | None = None
    mod_config: ModuleConfig = Field(default_factory=lambda: ModuleConfig())
    sequence: Sequence | None = None
    timeline: ComplexNDArray | None = None


class QbloxProgram(AbstractProgram):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    packages: dict[str, QbloxPackage]

    # COMPILER-1004, COMPILER-1005
    driver_version: SemanticVersion
    fw_version: SemanticVersion

    timeout_seconds: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS, gt=0, allow_inf_nan=False
    )

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}
