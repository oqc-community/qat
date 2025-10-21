# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from pydantic import ConfigDict

from qat.backend.passes.purr.analysis import TriageResult
from qat.backend.qblox.config.specification import ModuleConfig, SequencerConfig
from qat.backend.qblox.ir import Sequence
from qat.executables import AcquireData, ChannelExecutable


@dataclass
class QbloxPackage:
    pulse_channel_id: Optional[str] = None
    physical_channel_id: Optional[str] = None
    instrument_id: Optional[str] = None
    seq_idx: Optional[int] = None
    seq_config: SequencerConfig = field(default_factory=lambda: SequencerConfig())
    slot_idx: Optional[int] = None
    mod_config: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    sequence: Optional[Sequence] = None
    timeline: Optional[np.ndarray] = None


class QbloxExecutable(ChannelExecutable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    packages: Dict[str, QbloxPackage]
    triage_result: TriageResult

    @property
    def acquires(self) -> list[AcquireData]:
        return []
