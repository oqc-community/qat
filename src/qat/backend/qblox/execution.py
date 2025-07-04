# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from pydantic import ConfigDict

from qat.backend.passes.purr.analysis import TriageResult
from qat.backend.qblox.config.specification import SequencerConfig
from qat.backend.qblox.ir import Sequence
from qat.model.device import PulseChannel
from qat.runtime.executables import AcquireData, ChannelExecutable


@dataclass
class QbloxPackage:
    sequence: Sequence = None
    sequencer_config: SequencerConfig = field(default_factory=lambda: SequencerConfig())
    timeline: np.ndarray = None


class QbloxExecutable(ChannelExecutable):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    packages: Dict[PulseChannel, QbloxPackage]
    triage_result: TriageResult

    @property
    def acquires(self) -> list[AcquireData]:
        return []
