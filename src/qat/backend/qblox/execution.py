# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field

from pydantic import ConfigDict

from qat.backend.qblox.config.specification import ModuleConfig, SequencerConfig
from qat.backend.qblox.ir import Sequence
from qat.executables import AbstractProgram
from qat.utils.pydantic import ComplexNDArray


@dataclass
class QbloxPackage:
    pulse_channel_id: str | None = None
    physical_channel_id: str | None = None
    instrument_id: str | None = None
    seq_idx: int | None = None
    seq_config: SequencerConfig = field(default_factory=lambda: SequencerConfig())
    slot_idx: int | None = None
    mod_config: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    sequence: Sequence | None = None
    timeline: ComplexNDArray | None = None


class QbloxProgram(AbstractProgram):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # COMPILER 828: change to pydantic objects (not dataclasses)
    packages: dict[str, QbloxPackage]

    @property
    def acquire_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}
